"""Calculate three-hour start probabilities and the union of crew-reachable starts on paired paths."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit


def member_windows(paths, clear, daylight, gate):
    if paths.shape != (64, 48, 2) or clear.shape != (48,) or daylight.shape != (48,):
        raise ValueError("Expected paths[64,48,2], clear[48] and daylight[48].")
    if (
        daylight.dtype != bool
        or not np.isfinite(gate)
        or not np.isfinite(clear[daylight]).all()
        or not np.isfinite(paths[:, daylight, :]).all()
    ):
        raise ValueError(
            "A joint probability needs complete forecast values during daylight and a finite PM10 threshold."
        )
    hourly = daylight & (paths[..., 0] >= 0.75 * clear) & (paths[..., 1] <= gate)
    return np.lib.stride_tricks.sliding_window_view(hourly, 3, axis=1).all(-1)


def mapped(raw, c):
    return expit(c[0] * logit(np.clip(raw, 1 / 128, 127 / 128)) + c[1])


def legacy_map(raw, c):
    return raw if tuple(c) == (1.0, 0.0) else mapped(raw, c)


def calculate(paths, clear, daylight, gate, remaining, calibration):
    if remaining not in range(6, 49, 6):
        raise ValueError(
            "The remaining task deadline must be 6 through 48 hours, in six-hour steps."
        )
    windows = member_windows(paths, clear, daylight, gate)
    solar = np.lib.stride_tricks.sliding_window_view(clear, 3).mean(-1) / 1000
    sun = np.lib.stride_tricks.sliding_window_view(daylight, 3).all(-1)
    offsets = np.arange(remaining - 2)
    raw_start = windows.mean(axis=0)
    start = raw_start.copy()
    if calibration:
        if "coefficients" in calibration:
            legacy = calibration["coefficients"]
            start = np.where(sun, legacy_map(raw_start, legacy["s3"]), 0)
        elif calibration["support"]["start"] > 0:
            c = calibration["coeff"]["start"]
            fitted = (
                c[0] * logit(np.clip(raw_start, 1 / 128, 127 / 128))
                + c[1]
                + c[2] * (solar - 0.5)
                + c[3] * (np.arange(46) / 45 - 0.5)
            )
            start = np.where(sun, expit(fitted), 0)
    reach = anywhere = 0.0
    raw_reach = raw_any = 0.0
    for horizon in range(6, remaining + 1, 6):
        end = horizon - 2
        stop = max(6, min(horizon, 18) - 2)
        allowed = slice(6, stop)
        raw_any = windows[:, :end].any(axis=1).mean()
        raw_reach = windows[:, allowed].any(axis=1).mean()
        fitted_any, fitted_reach = raw_any, raw_reach
        if calibration:
            if "coefficients" in calibration:
                fitted_any = legacy_map(raw_any, legacy[f"a3:{horizon}"])
                fitted_reach = legacy_map(raw_reach, legacy[f"r3:6:12:{horizon}"])
            else:
                for label in ("any", "reach"):
                    if calibration["support"][label] > 0:
                        if label == "any":
                            fitted_any = mapped(raw_any, calibration["coeff"][label])
                        else:
                            fitted_reach = mapped(raw_reach, calibration["coeff"][label])
        reach = max(
            reach, fitted_reach if sun[allowed].any() else 0, start[allowed].max(initial=0.0)
        )
        anywhere = max(anywhere, reach, fitted_any if sun[:end].any() else 0, start[:end].max())
    return pd.DataFrame(
        dict(
            start_offset=offsets,
            reachable_start=(offsets >= 6) & (offsets + 3 <= min(18, remaining)),
            raw_start=raw_start[offsets],
            raw_reach=raw_reach,
            raw_any=raw_any,
            start_probability=start[offsets],
            reachable_probability=reach,
            any_probability=anywhere,
        )
    )


def read_paths(path, gate):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return (
                archive["paths"],
                archive["clear"],
                archive["daylight"],
                float(archive["gate"]) if gate is None else gate,
            )
    data = pd.read_csv(path)
    if gate is None:
        if "gate" not in data or data.gate.nunique(dropna=False) != 1:
            raise ValueError("Provide one PM10 gate for the task.")
        gate = float(data.gate.iloc[0])
    if (
        data[["member", "lead_end"]].duplicated().any()
        or set(data.member) != set(range(64))
        or set(data.lead_end) != set(range(1, 49))
        or len(data) != 64 * 48
    ):
        raise ValueError("Provide one forecast for every member 0–63 and hour end 1–48.")
    if (
        data.daylight.dtype != bool
        or not data.groupby("lead_end")[["clear", "daylight"]]
        .nunique(dropna=False)
        .eq(1)
        .all()
        .all()
    ):
        raise ValueError(
            "Clear-sky irradiance and the Boolean daylight mask must be shared across members."
        )
    data = data.sort_values(["member", "lead_end"])
    return (
        data[["GHI", "PM"]].to_numpy().reshape(64, 48, 2),
        data.clear.to_numpy()[:48],
        data.daylight.to_numpy()[:48],
        gate,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        type=Path,
        help="Paired path NPZ, or CSV with member, lead_end, GHI, PM, clear, daylight and gate",
    )
    parser.add_argument(
        "output", type=Path, help="Output probability table, one row per physical start"
    )
    parser.add_argument("--gate", type=float, help="Task-creation PM10 threshold in µg/m³")
    parser.add_argument("--remaining", type=int, default=48)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Frozen fitted coefficient JSON for this model and issue year",
    )
    args = parser.parse_args()
    cal = json.loads(args.calibration.read_text("utf-8")) if args.calibration else None
    calculate(*read_paths(args.paths, args.gate), args.remaining, cal).to_csv(
        args.output, index=False
    )


if __name__ == "__main__":
    main()
