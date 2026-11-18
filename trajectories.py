"""Construct 64 paired 48-hour irradiance and PM10 trajectories from fitted marginal quantiles."""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import ndtri

LEVELS = np.array([0.05, 0.25, 0.50, 0.75, 0.95], dtype="f4")
MEMBERS = 64


def ranks(values):
    return np.argsort(np.argsort(values, axis=0, kind="stable"), axis=0, kind="stable")


def conditional_ranks(library, state, hour, day_of_year):
    """Select earlier paired donor blocks from the frozen hour- and season-matched library."""
    candidates = np.flatnonzero(library["hour"] == hour)
    if not len(candidates):
        raise ValueError("No donor blocks are available for the issue hour.")
    gap = np.abs(library["day"][candidates] - day_of_year)
    gap = np.minimum(gap, 366 - gap)
    nearby = candidates[gap <= 46]
    if len(nearby) < MEMBERS:
        nearby = candidates[np.argsort(gap, kind="stable")[:MEMBERS]]
    distance = (((library["state"][nearby] - state) / library["scale"]) ** 2).mean(axis=1)
    selected = np.resize(nearby[np.argsort(distance, kind="stable")[:MEMBERS]], MEMBERS)
    return ranks(library["blocks"][selected])


def ordered_members(quantiles, clear, daylight):
    """Quantile columns are normalized irradiance and log1p(PM10)/5, respectively."""
    q = np.sort(quantiles, axis=-1)
    if q.shape != (48, 2, 5) or np.shape(clear) != (48,) or np.shape(daylight) != (48,):
        raise ValueError("Expected quantiles[48,2,5], clear[48] and daylight[48].")
    grid = np.r_[0, LEVELS, 1]
    levels = (np.arange(MEMBERS, dtype="f4") + 0.5) / MEMBERS
    left = np.clip(np.searchsorted(grid, levels, side="right") - 1, 0, len(grid) - 2)
    weight = ((levels - grid[left]) / (grid[left + 1] - grid[left]))[:, None, None]
    lower = q[..., :1] - 0.25 * (q[..., 1:2] - q[..., :1])
    upper = q[..., -1:] + 0.25 * (q[..., -1:] - q[..., -2:-1])
    knots = np.concatenate((lower, q, upper), axis=-1)
    values = (
        knots[..., left].transpose(2, 0, 1) * (1 - weight)
        + knots[..., left + 1].transpose(2, 0, 1) * weight
    ).astype("f4")
    values[..., 0] = np.where(daylight, np.maximum(values[..., 0], 0) * clear, 0)
    values[..., 1] = np.expm1(np.maximum(values[..., 1], 0) * 5)
    return values


def couple(members, fixed_ranks, state_ranks, choice):
    for template in (fixed_ranks, state_ranks):
        if (
            template.shape != members.shape
            or not np.equal(np.sort(template, axis=0), np.arange(MEMBERS)[:, None, None]).all()
        ):
            raise ValueError("Each member rank column must be a permutation of 0 through 63.")
    fixed = np.take_along_axis(members, fixed_ranks, axis=0)
    conditioned = np.take_along_axis(members, state_ranks, axis=0)
    if choice == "fixed":
        return fixed
    if choice == "conditional":
        return conditioned
    if choice != "half_rank":
        raise ValueError("Unknown dependence template.")
    z = ndtri((np.arange(MEMBERS) + 0.5) / MEMBERS)
    hybrid = ranks(z[ranks(fixed)] + z[ranks(conditioned)])
    return np.take_along_axis(members, hybrid, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        help="One issue, NPZ: quantiles, clear, daylight, fixed_ranks, conditional_ranks (or state, hour, day_of_year with --library)",
    )
    parser.add_argument(
        "output", type=Path, help="Output NPZ: paths[member,hour,component], clear, daylight"
    )
    parser.add_argument(
        "--library", type=Path, help="Frozen donor NPZ with blocks, state, scale, day, hour"
    )
    parser.add_argument(
        "--template", choices=["conditional", "fixed", "half_rank"], default="conditional"
    )
    args = parser.parse_args()
    with np.load(args.inputs, allow_pickle=False) as inputs:
        members = ordered_members(inputs["quantiles"], inputs["clear"], inputs["daylight"])
        if args.library:
            with np.load(args.library, allow_pickle=False) as library:
                selected = conditional_ranks(
                    library, inputs["state"], inputs["hour"].item(), inputs["day_of_year"].item()
                )
        else:
            selected = inputs["conditional_ranks"]
        paths = couple(members, inputs["fixed_ranks"], selected, args.template)
        np.savez_compressed(
            args.output, paths=paths, clear=inputs["clear"], daylight=inputs["daylight"]
        )


if __name__ == "__main__":
    main()
