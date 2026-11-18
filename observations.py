"""Estimate recovery intervals, renewed disturbances and matched ordinary-day contrasts."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from recovery_intervals import H, component, earliest, order, stability, windows

DAY = 24 * H
ZONE = "Asia/Shanghai"
CLOCKS = ("start_pm0", "start_pm1", "end_pm0", "end_pm1")


def read_settings(site, hours_path, events_path):
    hours = pd.read_parquet(hours_path)
    events = pd.read_csv(events_path)
    if site == "xianghe":
        events = events.loc[events["split"].eq("development")].rename(columns={"group": "event"})
        for col in ("start", "end", "stop"):
            events[col] = pd.to_datetime(events[col], utc=True)
        hours.index = hours.index.tz_convert("UTC")
        frame = hours.rename(
            columns={
                "sun_deg": "sun",
                "ratio_valid": "ratio",
                "pm10_1068A": "pm",
                "wind_ms": "wind",
                "rh_pct": "rh",
            }
        )
        yield "Xianghe", "primary", frame, events
    else:
        events = events.rename(columns={"start_utc": "start", "end_utc": "end"})
        for col in ("start", "end"):
            events[col] = pd.to_datetime(events[col], utc=True)

        def followup(row):
            subsequent = events.loc[(events.year == row.year) & (events.start >= row.end), "start"]
            return min([row.end + 7 * DAY, *subsequent])

        events["stop"] = [followup(row) for row in events.itertuples()]
        for station in ("ZhangyeWetland", "Daman"):
            for clock in CLOCKS:
                aws, pm = clock.split("_")
                group = hours.loc[(hours.site == station) & (hours.aws == aws)].set_index("end_utc")
                group = group.rename(columns={pm.upper(): "pm"})
                yield station, clock, group, events


def background(events, is_xianghe):
    excluded = set()
    for row in events.itertuples():
        start = row.start.tz_convert(ZONE).normalize()
        end = row.end.tz_convert(ZONE).normalize()
        excluded.update(pd.date_range(start, end + DAY, freq="D"))
    if is_xianghe:
        excluded.update(pd.date_range("2018-04-25", "2018-04-26", tz=ZONE))
    return excluded


def analyse_event(g, event, excluded, key):
    start, end, stop = event.start, event.end, event.stop
    local = start.tz_convert(ZONE)
    dates = (g.index - H).tz_convert(ZONE).normalize()
    prior = [
        t for t in pd.date_range(local - 42 * DAY, local - DAY, freq="D") if t not in excluded
    ][-14:]
    baseline = g.loc[dates.isin(prior)].copy()
    baseline["date"] = (baseline.index - H).tz_convert(ZONE).normalize()
    pm = baseline.pm.dropna()
    enough = len(pm) >= 72 and baseline.loc[pm.index, "date"].nunique() >= 5
    severe, good = pm.quantile([0.90, 0.75]).to_numpy() if enough else (np.nan, np.nan)
    core = g.loc[(g.index > start) & (g.index <= end)]
    flags = [
        (g.ratio, g.sun.ge(10), 0.65, 0.75, "rad"),
        (g.pm, pd.Series(True, index=g.index), severe, good, "pm"),
    ]
    parts = [
        component(g, values, expected, start, end, stop, bad, threshold, kind)
        for values, expected, bad, threshold, kind in flags
    ]
    joint = (core.ratio.notna() & core.pm.notna()).sum()
    qualifying = (
        enough
        and joint >= max(3, np.ceil(core.sun.ge(10).sum() / 2))
        and all(p[0]["state"] != "insufficient" for p in parts)
    )
    disturbed = tuple(
        p[0]["state"] in {"hour_bracket", "interval_censored", "right_censored"} for p in parts
    )
    classification = (
        {
            (True, True): "dual_disturbed",
            (True, False): "ghi_only",
            (False, True): "pm_only",
            (False, False): "neither_disturbed",
        }[disturbed]
        if qualifying
        else "insufficient_common"
    )
    key = dict(key, core_class=classification)
    recovery, follow = [], []
    for idx, anchor in enumerate(("own", "common")):
        clocks = [p[idx] for p in parts]
        row = dict(**key, anchor=anchor, **order(*clocks))
        if classification != "dual_disturbed":
            row["order"] = "not_dual"
        for engaged, meta, part, record, channel in zip(
            disturbed, flags, parts, clocks, ("ghi", "pm")
        ):
            values, expected, bad, threshold, kind = meta
            for field in ("state", "lower", "upper", "confirmed"):
                row[f"{channel}_{field}"] = record[field]
            base = part[0]["extreme_time"] if anchor == "own" else end + H
            possible, confirmed = (
                earliest(values, expected, threshold, kind, base, stop)
                if engaged
                else (pd.NaT, pd.NaT)
            )
            row[f"{channel}_possible_start"], row[f"{channel}_observed_start"] = possible, confirmed
            checks = (
                part[2]
                if anchor == "own"
                else [
                    stability(
                        values,
                        expected,
                        dict(record, extreme_time=end),
                        bad,
                        threshold,
                        kind,
                        stop,
                        horizon,
                    )
                    for horizon in (24, 48)
                ]
            )
            follow.extend(
                dict(**key, anchor=anchor, component=channel, **check) for check in checks
            )
        recovery.append(row)
    weather = baseline.groupby("date")[["wind", "rh"]].mean()
    distances = ((weather.wind - core.wind.mean()) / 3) ** 2 + (
        (weather.rh - core.rh.mean()) / 20
    ) ** 2
    control_date = distances.dropna().sort_index().idxmin() if distances.notna().any() else pd.NaT
    ordinary = baseline.loc[baseline.date.eq(control_date)]
    contrast = dict(
        **key,
        control_start_utc=control_date.tz_convert("UTC") if pd.notna(control_date) else pd.NaT,
        core_ghi_ratio=core.ratio.mean(),
        control_ghi_ratio=ordinary.ratio.mean(),
        core_pm=core.pm.mean(),
        control_pm=ordinary.pm.mean(),
    )
    tail = g.loc[(g.index > end) & (g.index <= min(stop, end + 48 * H))]
    opportunity = [
        dict(**key, physical_start_utc=t, truth=truth) for t, truth in windows(tail, good)
    ]
    return recovery, follow, [contrast], opportunity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", choices=["xianghe", "zhangye"])
    parser.add_argument(
        "hours", type=Path, help="Study hourly observation Parquet for the selected site"
    )
    parser.add_argument(
        "events",
        type=Path,
        help="Registered weather groups (Xianghe) or UTC event catalogue (Zhangye), CSV",
    )
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    names = ("recovery", "followup", "matched_days", "window_starts")
    outputs = [[] for _ in names]
    for station, clock, hours, events in read_settings(args.site, args.hours, args.events):
        excluded = background(events, args.site == "xianghe")
        for event in events.itertuples():
            g = hours.loc[event.start - 42 * DAY : event.stop + 48 * H].sort_index()
            if not g.index.is_unique or not g.index.to_series().diff().dropna().eq(H).all():
                raise ValueError(
                    "Hourly records require unique, consecutive hour ends; retain missing values as rows."
                )
            key = dict(site=station, clock=clock, event=event.event)
            for output, rows in zip(outputs, analyse_event(g, event, excluded, key)):
                output.extend(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(names, outputs):
        pd.DataFrame(rows).to_csv(args.out / (name + ".csv"), index=False)


if __name__ == "__main__":
    main()
