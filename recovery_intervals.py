"""Hourly recovery intervals, recurrence and joint observational windows."""

import numpy as np
import pandas as pd

HR = H = pd.Timedelta(hours=1)


def runs(mask, hours):
    if len(mask) < hours:
        return []
    assert mask.index.is_unique and mask.index.is_monotonic_increasing
    flags = mask.fillna(False).to_numpy(dtype=bool)
    consecutive = np.lib.stride_tricks.sliding_window_view(flags, hours).all(axis=1)
    clocks = (mask.index[hours - 1 :] - mask.index[: len(mask) - hours + 1]) == (hours - 1) * HR
    return list(mask.index[: len(consecutive)][consecutive & clocks])


def endpoint(series, expected, start, core_end, stop, bad, good, kind, duration):
    # Series indexed by interval_end. Index supports (previous hour, end].
    v = series.where(expected)
    core = v[(v.index > start) & (v.index <= core_end)]
    mask = (expected.index > start) & (expected.index <= core_end)
    count = int(core.notna().sum())
    den = int(expected[mask].sum())
    res = dict(
        state="insufficient",
        lower=pd.NaT,
        upper=pd.NaT,
        confirmed=pd.NaT,
        last_failure=pd.NaT,
        extreme_time=pd.NaT,
        extreme_value=np.nan,
        core_valid=count,
        core_expected=den,
        rebound=False,
        post_confirm_valid=0,
        tail_missing=False,
    )
    low = kind != "pm"
    if count < max(3, np.ceil(0.5 * den)) or not np.isfinite(bad + good):
        if count:
            extreme = core.idxmin() if low else core.idxmax()
            res.update(extreme_time=extreme, extreme_value=float(core[extreme]))
        return res
    disturbed = core < bad if low else core > bad
    meets = v >= good if low else v <= good
    if not disturbed.any():
        res["state"] = "no_disturbance"
        pre = v[(v.index > start - pd.Timedelta(days=1)) & (v.index <= start)]
        prebad = pre < bad if low else pre > bad
        if prebad.any():
            lastbad = pre[prebad].index[-1]
            prepass = runs(meets[(meets.index > lastbad) & (meets.index <= start)], duration)
            if (
                prepass
                and not ((~meets) & v.notna() & (v.index >= prepass[0]) & (v.index <= start)).any()
            ):
                res.update(
                    state="left_censored",
                    upper=prepass[0],
                    confirmed=prepass[0] + (duration - 1) * HR,
                    last_failure=lastbad,
                )
        return res
    extreme = core.idxmin() if low else core.idxmax()
    eligible = v[(v.index >= extreme) & (v.index <= stop)]
    res.update(extreme_time=extreme, extreme_value=float(core[extreme]))
    r = runs(meets.reindex(eligible.index), duration)
    if not r:
        failures = eligible.loc[eligible.notna() & ~meets.reindex(eligible.index)]
        res.update(
            state="right_censored", lower=failures.index[-1], last_failure=failures.index[-1]
        )
        exp = expected.loc[(expected.index > extreme) & (expected.index <= stop)]
        if exp.any():
            res["tail_missing"] = pd.isna(v.loc[exp[exp].index[-1]])
        return res
    first = r[0]
    confirmed = first + (duration - 1) * HR
    fail = v.loc[(v.index >= extreme) & (v.index < first) & v.notna() & ~meets].index[-1]
    after = eligible[eligible.index > confirmed].dropna()
    res.update(
        state="hour_bracket" if first - fail == HR else "interval_censored",
        lower=fail,
        upper=first,
        confirmed=confirmed,
        last_failure=fail,
        rebound=bool((after < bad if low else after > bad).any()),
        post_confirm_valid=len(after),
    )
    return res


def order(rad, pm):
    lo = (
        (pm["lower"] - rad["upper"]) / HR
        if pd.notna(pm["lower"]) and pd.notna(rad["upper"])
        else np.nan
    )
    hi = (
        (pm["upper"] - rad["lower"]) / HR
        if pd.notna(pm["upper"]) and pd.notna(rad["lower"])
        else np.nan
    )
    if rad["state"] == "left_censored" or pm["state"] == "left_censored":
        cl = "unresolved"
    elif lo > 0:
        cl = "radiation_first"
    elif hi < 0:
        cl = "pm_first"
    else:
        cl = "unresolved"
    return dict(lag_lower_h=lo, lag_upper_h=hi, order=cl)


def opportunity(v, expected, confirmation, severe, stop, length):
    wanted = pd.date_range(confirmation + H, periods=length, freq="h")
    exp = expected.reindex(wanted).fillna(False)
    scope = v.reindex(wanted).where(exp & (wanted <= stop))
    encountered = severe.reindex(wanted).fillna(False) & scope.notna()
    full = (
        wanted[-1] <= stop
        and expected.reindex(wanted).notna().all()
        and exp.any()
        and scope.notna().sum() == exp.sum()
    )
    state = (
        "observed_rebound" if encountered.any() else ("observed_no_rebound" if full else "unknown")
    )
    return dict(
        state=state,
        time=encountered[encountered].index.min(),
        observed=int(scope.notna().sum()),
        expected=int(exp.sum()),
        truncated=wanted[-1] > stop,
    )


def stability(v, expected, first, bad, good, kind, stop, length):
    result = dict(
        horizon_h=length,
        first_confirmation=first["confirmed"],
        rebound_state="not_applicable",
        rebound_time=pd.NaT,
        opportunities=0,
        observed=0,
        truncated=False,
        stable_state=first["state"],
        stable_lower=pd.NaT,
        stable_upper=pd.NaT,
        stable_confirmed=pd.NaT,
        candidate_count=0,
        unknown_candidates=0,
    )
    if first["state"] not in ["hour_bracket", "interval_censored"]:
        if first["state"] == "right_censored":
            result.update(stable_lower=first["lower"], rebound_state="unknown")
        if first["state"] == "left_censored":
            result.update(stable_state="entry_left_followup_unknown", rebound_state="unknown")
        return result
    severe = (v < bad if kind == "rad" else v > bad) & expected
    goodmask = (v >= good if kind == "rad" else v <= good) & expected
    o = opportunity(v, expected, first["confirmed"], severe, stop, length)
    result.update(
        rebound_state=o["state"],
        rebound_time=o["time"],
        opportunities=o["expected"],
        observed=o["observed"],
        truncated=o["truncated"],
    )
    valid = runs(goodmask.loc[first["upper"] : stop], 3)
    result["candidate_count"] = len(valid)
    # The latest severe observation excludes any candidate whose follow-up includes that recurrence.
    floor = first["lower"]
    possible_lower = None
    for t in valid:
        follow = opportunity(v, expected, t + 2 * H, severe, stop, length)
        failures = v.loc[first["extreme_time"] : t - H].loc[
            lambda x: x.notna() & ~goodmask.reindex(x.index)
        ]
        bound = failures.index.max() if len(failures) else floor
        if follow["state"] == "unknown":
            result["unknown_candidates"] += 1
            if possible_lower is None:
                possible_lower = bound
        if follow["state"] == "observed_no_rebound":
            lower = min(bound, possible_lower) if possible_lower is not None else bound
            result.update(
                stable_state="hour_bracket" if t - lower == H else "interval_censored",
                stable_lower=lower,
                stable_upper=t,
                stable_confirmed=t + (2 + length) * H,
            )
            return result
    result.update(
        stable_state="followup_unknown" if result["unknown_candidates"] else "right_censored",
        stable_lower=possible_lower if possible_lower is not None else floor,
    )
    return result


def shared(v, expected, first, bad, good, kind, end, stop):
    if first["state"] not in ["hour_bracket", "interval_censored", "right_censored"]:
        return dict(state=first["state"], lower=pd.NaT, upper=pd.NaT, confirmed=pd.NaT)
    s = v.where(expected)
    s = s[(s.index > end) & (s.index <= stop)]
    passes = s >= good if kind == "rad" else s <= good
    hits = runs(passes, 3)
    fail = s[s.notna() & ~passes]
    if not hits:
        return dict(
            state="right_censored",
            lower=fail.index.max() if len(fail) else end,
            upper=pd.NaT,
            confirmed=pd.NaT,
        )
    upper = hits[0]
    fail = fail[fail.index < upper]
    lower = fail.index.max() if len(fail) else end
    return dict(
        state="hour_bracket" if upper - lower == H else "interval_censored",
        lower=lower,
        upper=upper,
        confirmed=upper + 2 * H,
    )


def component(g, s, exp, start, end, stop, bad, good, kind):
    first = endpoint(s, exp, start, end, stop, bad, good, kind, 3)
    common = shared(s, exp, first, bad, good, kind, end, stop)
    return first, common, [stability(s, exp, first, bad, good, kind, stop, k) for k in [24, 48]]


def earliest(series, expected, thresh, kind, anchor, stop):
    if pd.isna(anchor) or pd.isna(thresh):
        return pd.NaT, pd.NaT
    sub = series.loc[anchor:stop]
    e = expected.reindex(sub.index)
    ok = sub.ge(thresh) if kind == "rad" else sub.le(thresh)
    st = np.where(~e, 0, np.where(sub.isna(), -1, ok.astype(int)))
    if len(st) < 3:
        return pd.NaT, pd.NaT
    w = np.lib.stride_tricks.sliding_window_view(st, 3)
    poss = np.flatnonzero((w != 0).all(1))
    sure = np.flatnonzero((w == 1).all(1))
    return (
        sub.index[poss[0]] - H if len(poss) else pd.NaT,
        sub.index[sure[0]] - H if len(sure) else pd.NaT,
    )


def tri(g, gate):
    fails = (g.sun < 10) | (g.ratio < 0.75) | (g.pm > gate)
    meets = (g.sun >= 10) & (g.ratio >= 0.75) & (g.pm <= gate)
    return np.select([fails, meets], [0, 1], default=-1)


def windows(g, gate):
    xx = tri(g, gate)
    if len(xx) < 3:
        return []
    return [
        (t - H, 0 if 0 in w else 1 if min(w) == 1 else -1)
        for t, w in zip(g.index, np.lib.stride_tricks.sliding_window_view(xx, 3))
    ]
