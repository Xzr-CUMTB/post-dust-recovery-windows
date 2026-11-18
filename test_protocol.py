"""Small deterministic checks of the study definitions. Run with python -m unittest."""
import unittest

import numpy as np
import pandas as pd

from recovery_intervals import H, component
from trajectories import couple
from window_probabilities import calculate, member_windows


class ProtocolTests(unittest.TestCase):
    def test_recurrence_after_confirmed_recovery(self):
        hours = pd.date_range("2000-01-01", periods=40, freq="h", tz="UTC")
        ratio = pd.Series(0.85, index=hours)
        ratio.iloc[1:4] = 0.50
        ratio.iloc[10] = 0.60
        expected = pd.Series(True, index=hours)
        first, _, follow = component(
            None, ratio, expected, hours[0], hours[3], hours[-1], 0.65, 0.75, "rad"
        )
        self.assertEqual(first["upper"], hours[4])
        self.assertEqual(first["confirmed"], hours[4] + 2 * H)
        self.assertEqual(follow[0]["rebound_state"], "observed_rebound")
        self.assertEqual(follow[0]["rebound_time"], hours[10])

    def test_decisions_use_pathwise_unions(self):
        daylight = np.ones(48, dtype=bool)
        clear = np.ones(48) * 600
        paths = np.zeros((64, 48, 2))
        paths[:32, 6:9, 0] = 600
        paths[32:, 13:16, 0] = 600
        table = calculate(paths, clear, daylight, 50, 48, None)
        self.assertEqual(table.raw_start.max(), 0.5)
        self.assertEqual(table.raw_reach.iloc[0], 1.0)
        daylight[7] = False
        self.assertFalse(member_windows(paths, clear, daylight, 50)[:, 6].any())

    def test_reordering_preserves_hourly_marginals(self):
        rng = np.random.default_rng(31)
        members = np.sort(rng.normal(size=(64, 48, 2)), axis=0)
        fixed = np.argsort(rng.uniform(size=members.shape), axis=0)
        conditional = np.argsort(rng.uniform(size=members.shape), axis=0)
        for template in ("fixed", "conditional", "half_rank"):
            paths = couple(members, fixed, conditional, template)
            np.testing.assert_array_equal(np.sort(paths, axis=0), members)


if __name__ == "__main__":
    unittest.main()
