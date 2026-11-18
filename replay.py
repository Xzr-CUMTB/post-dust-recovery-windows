"""Reconstruct reservations and work starts from the frozen 2022 issue and settlement records."""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def maxval(a):
    a = np.asarray(a)
    return np.nanmax(a) if np.isfinite(a).any() else np.nan


def three(a):
    return np.lib.stride_tricks.sliding_window_view(a, 3)


def run(official, out):
    results = official / "results"
    settlements = pd.read_csv(results / "task_settlement.csv").query(
        "track=='archive' and model=='conditional'"
    )
    ledger = settlements.loc[settlements.rule.eq("RAC")]
    physical = pd.read_csv(results / "physical_windows.csv", keep_default_na=False)
    updates = []
    diagnoses = []
    components = []
    classifications = []
    worked_hours = []
    for (site, clock), df in ledger.groupby(["site", "clock"]):
        lane = f"2022_{site}_{clock}"
        ledger_by_id = df.set_index("i")
        full = (
            pd.read_csv(official / "actions" / (lane + ".csv"))
            .query("track=='archive' and model=='conditional' and rule=='RAC'")
            .set_index("i")
        )
        pd.testing.assert_frame_equal(
            full.sort_index(), ledger_by_id[full.columns].sort_index(), check_dtype=False
        )
        env = np.load(official / "private" / (lane + ".npz"))
        valid_starts = {
            r.i: {int(x) - r.i for x in r.reachable_starts.split("|") if x}
            for r in physical.query("site==@site and clock==@clock").itertuples()
        }
        state = {i: dict(last=-100, mask=0, start=-1, decision=-1, quota=0, crew=0) for i in df.i}
        used = 0
        busy = -1
        local = []
        for f in sorted((official / "predictions" / lane).glob("*.npz"), key=lambda f: int(f.stem)):
            with np.load(f, allow_pickle=False) as z:
                idx = int(z["i"])
                ids = z["ids"]
                labels = z["models"].tolist()
                m = labels.index("conditional")
                all_p = z["start"]
                aux = z["aux"][:, m]
                day = three(z["day"]).all(1)
                gates = z["gate"]
            for j, ident in enumerate(ids):
                i = int(ident)
                t = idx - i
                k = t // 6
                rem = 48 - t
                a = state[i]
                p = all_p[j, m]
                gate = gates[j]
                entry = ledger_by_id.loc[i]
                real = valid_starts[i]
                cap_before = used
                feasible = (np.arange(46) >= 6) & (np.arange(46) < min(rem, 18) - 2) & day
                valid = np.isfinite(gate)
                active = a["start"] < 0
                signal = bool(
                    active and valid and t + 9 <= 48 and feasible.any() and aux[j, 1] >= 0.75
                )
                booking = "none"
                dispatch = False
                top = np.nan
                truth_top = np.nan
                chosen = -1
                if signal and t >= a["last"] + 18:
                    if used >= 11:
                        booking = "quota"
                        a["quota"] += 1
                    elif idx < busy:
                        booking = "crew"
                        a["crew"] += 1
                    else:
                        booking = "granted"
                        a["last"] = t
                        a["mask"] |= 1 << k
                        used += 1
                        busy = idx + min(rem, 18)
                if active and valid and a["last"] + 6 <= t < a["last"] + 18:
                    n = min(6, a["last"] + 18 - t - 2, rem - 2)
                    allowed = np.flatnonzero(day[:n])
                    actual = [l for l in allowed if t + l in real]
                    top = maxval(p[allowed])
                    truth_top = maxval(p[actual])
                    go = allowed[p[allowed] >= 0.75]
                    if len(go):
                        chosen = int(go[np.argmax(3 * p[go] - 2 - 0.001 * go)])
                        a["start"] = t + chosen
                        a["decision"] = t
                        busy = idx + chosen + 3
                        dispatch = True
                local.append(
                    dict(
                        site=site,
                        clock=clock,
                        i=i,
                        issue=idx,
                        t=t,
                        admitted=int(valid),
                        active=int(active),
                        phase=entry.creation_phase,
                        process=entry.creation_process,
                        p_any48=aux[j, 0],
                        p_reach=aux[j, 1],
                        has_booking_daylight=int(feasible.any()),
                        signal=int(signal),
                        booking=booking,
                        quota_used_before=cap_before,
                        feasible_booking_true=int(
                            any(t + l in real for l in np.flatnonzero(feasible))
                        ),
                        max_start_when_prepared=top,
                        max_start_on_covered_true=truth_top,
                        dispatch=int(dispatch),
                        chosen_offset=chosen,
                    )
                )
                if t == 0:
                    row = int(np.flatnonzero(env["ids"] == i)[0])
                    g = three(env["ghi"][row])
                    c = three(env["clear"][row])
                    pm = three(env["pm"][row])
                    pm_fail = (pm > gate).any(1)
                    solar_fail = (g < 0.75 * c).any(1)
                    known = np.isfinite(g).all(1) & np.isfinite(pm).all(1) & valid
                    for off in np.flatnonzero(day):
                        truth = 0 if pm_fail[off] or solar_fail[off] else (1 if known[off] else -1)
                        klass = (
                            "both_fail"
                            if pm_fail[off] and solar_fail[off]
                            else (
                                "pm_exceeds"
                                if pm_fail[off] and np.isfinite(g[off]).all()
                                else (
                                    "solar_below"
                                    if solar_fail[off] and np.isfinite(pm[off]).all()
                                    else "both_pass" if truth == 1 else "incomplete"
                                )
                            )
                        )
                        classifications.append(
                            dict(
                                site=site,
                                clock=clock,
                                i=i,
                                offset=off,
                                phase=entry.creation_phase,
                                physical_state=klass,
                                truth=truth,
                                gate=gate,
                                **{
                                    s: all_p[j, labels.index(s), off]
                                    for s in [
                                        "fixed",
                                        "conditional",
                                        "RAC_raw",
                                        "M1_cal",
                                        "RAC_cal",
                                    ]
                                },
                            )
                        )
        upd = pd.DataFrame(local)
        updates.append(upd)
        for r in df.itertuples():
            a = state[r.i]
            assert (a["mask"], a["start"], a["decision"], a["quota"], a["crew"]) == (
                r.order_mask,
                r.start,
                r.decision,
                r.quota_denied,
                r.crew_denied,
            )
            d = upd[upd.i == r.i]
            if r.start >= 0:
                j = list(env["ids"]).index(r.i)
                with np.load(official / "predictions" / lane / (str(r.i) + ".npz")) as receipt:
                    gate = float(receipt["gate"][list(receipt["ids"]).index(r.i)])
                for h in range(3):
                    col = r.start + h
                    worked_hours.append(
                        dict(
                            site=site,
                            clock=clock,
                            i=r.i,
                            start_utc=r.start_utc,
                            h=h,
                            pm=env["pm"][j, col],
                            gate=gate,
                            ghi=env["ghi"][j, col],
                            clear_ghi=env["clear"][j, col],
                            solar_alt=env["sun"][j, col],
                            creation_phase=r.creation_phase,
                            execution_phase=r.execution_phase,
                        )
                    )
            if r.reachable != 1:
                reason = "unknown_opportunity" if r.reachable == -1 else "no_opportunity"
            elif not r.preparations:
                if a["quota"] and a["crew"]:
                    reason = "quota_and_crew"
                elif a["quota"]:
                    reason = "quota"
                elif a["crew"]:
                    reason = "crew"
                elif not d.admitted.any():
                    reason = "missing_input"
                else:
                    reason = "no_booking_signal"
            elif r.start >= 0:
                reason = {1: "success", 0: "wrong", -1: "unknown_dispatch"}[r.outcome]
            elif r.covered != 1:
                reason = "booking_did_not_cover_confirmed_start"
            else:
                reason = "covered_below_start_threshold"
            diagnoses.append(
                dict(
                    **r._asdict(),
                    reason=reason,
                    creation_any=d.iloc[0].p_any48,
                    creation_reach=d.iloc[0].p_reach,
                    max_eligible_booking_reach=maxval(
                        d.loc[
                            (d.t <= 39) & (d.has_booking_daylight == 1) & (d.active == 1), "p_reach"
                        ]
                    ),
                    max_start_prepared=maxval(d.max_start_when_prepared),
                    max_start_on_true=maxval(d.max_start_on_covered_true),
                    signal_updates=d.signal.sum(),
                    true_booking_updates=d.feasible_booking_true.sum(),
                )
            )
    tasks = pd.DataFrame(diagnoses).drop(columns="Index")
    updates = pd.concat(updates, ignore_index=True)
    cases = pd.DataFrame(classifications)
    out.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(out / "native_tasks.csv", index=False)
    updates.to_csv(
        out / "issue_path.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )
    pd.DataFrame(worked_hours).to_csv(out / "dispatch_observations.csv", index=False)
    cases.to_csv(
        out / "start_cases.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )
    groups = ["site", "clock"]
    rows = []
    for (s, c), v in tasks.groupby(groups):
        pos = v[v.reachable == 1]
        pw = physical.query("site==@s and clock==@c")
        starts = set(x for col in pw.reachable_starts for x in col.split("|") if x)
        reasons = pos.reason.value_counts()
        rows.append(
            dict(
                site=s,
                clock=c,
                tasks=len(v),
                reachable=len(pos),
                unknown_opportunity=sum(v.reachable == -1),
                distinct_true_utc=len(starts),
                input_admitted_updates=v.input_updates.sum(),
                bookings=v.preparations.sum(),
                booked_tasks=sum(v.preparations > 0),
                positive_booked=sum(pos.preparations > 0),
                covered=sum(v.covered == 1),
                quota_denied_updates=v.quota_denied.sum(),
                crew_denied_updates=v.crew_denied.sum(),
                start=sum(v.start >= 0),
                success=sum(v.outcome == 1),
                wrong=sum(v.outcome == 0),
                unknown_dispatch=sum(v.outcome == -1),
                abstain=sum(v.start < 0),
                crew_hours=v.crew_hours.sum(),
                total_cost=v.cost.sum(),
                utility_per_task=v.utility.mean(),
                **{
                    "positive_" + k: int(reasons.get(k, 0))
                    for k in [
                        "missing_input",
                        "no_booking_signal",
                        "quota",
                        "crew",
                        "quota_and_crew",
                        "booking_did_not_cover_confirmed_start",
                        "covered_below_start_threshold",
                        "wrong",
                        "success",
                        "unknown_dispatch",
                    ]
                },
            )
        )
    pd.DataFrame(rows).to_csv(out / "2022_ACTION_ATTRITION.csv", index=False)
    for keys, sub in cases.groupby(groups + ["physical_state"]):
        use = sub.truth >= 0
        for model in ["fixed", "conditional", "RAC_raw", "M1_cal", "RAC_cal"]:
            ok = use & sub[model].notna()
            components.append(
                dict(
                    zip(groups + ["physical_state"], keys),
                    model=model,
                    day_starts=len(sub),
                    known=ok.sum(),
                    positive=sum(sub.truth == 1),
                    mean_probability=sub.loc[ok, model].mean(),
                    brier=((sub.loc[ok, model] - sub.loc[ok, "truth"]) ** 2).mean(),
                )
            )
    pd.DataFrame(components).to_csv(out / "component_errors.csv", index=False)
    tasks.groupby(groups + ["creation_phase", "reason"]).size().rename(
        "tasks"
    ).reset_index().to_csv(out / "phase_attrition.csv", index=False)
    ref = pd.read_csv(results / "paired_prediction.csv").query(
        "track=='archive' and endpoint=='day_start' and left=='RAC_cal' and right=='M1_cal'"
    )
    for r in ref.itertuples():
        d = cases[(cases.site == r.site) & (cases.clock == r.clock) & (cases.truth >= 0)]
        assert len(d) == r.known
        np.testing.assert_allclose(
            ((d.RAC_cal - d.truth) ** 2 - (d.M1_cal - d.truth) ** 2).mean(), r.mean_delta, atol=1e-7
        )
    assert (
        len(tasks) == 1104
        and len(updates) == 8832
        and len(cases) == 20304
        and sum(cases.truth >= 0) == 19978
    )
    utilities = settlements.loc[settlements.rule.isin(["RAC", "P02D", "never"])].pivot(
        index=["site", "clock", "i"], columns="rule", values="utility"
    )
    if utilities.isna().any().any():
        raise ValueError("Missing matched-task utility for a comparison policy.")
    comparisons = pd.DataFrame({
        "main_minus_P02D": utilities["RAC"] - utilities["P02D"],
        "main_minus_never": utilities["RAC"] - utilities["never"],
    })
    comparisons.groupby(level=["site", "clock"]).mean().to_csv(out / "paired_mean_utility.csv")
    print(
        json.dumps(
            dict(
                native_tasks=len(tasks),
                native_updates=len(updates),
                creation_day_starts=len(cases),
                verified_lanes=len(rows),
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "archive", type=Path, help="Directory with predictions/, actions/, private/ and results/"
    )
    p.add_argument("out", type=Path, help="Output directory for the task and update ledgers")
    a = p.parse_args()
    run(a.archive, a.out)
