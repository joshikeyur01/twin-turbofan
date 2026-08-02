"""Tests for the RUL metrics.

The asymmetry of ``phm_score`` is the whole point of the metric — a maintenance
prediction that comes in LATE means the part failed in service, so it must score
strictly worse than an equally-sized early prediction. These tests pin that down.
"""

import numpy as np
import pytest

from src.evaluate import phm_score, rmse


class TestRMSE:
    def test_zero_when_perfect(self):
        y = np.array([10.0, 50.0, 125.0])
        assert rmse(y, y) == 0.0

    def test_known_value(self):
        # errors of -3 and +4 -> sqrt((9+16)/2) = 3.5355...
        assert rmse([10, 20], [7, 24]) == pytest.approx(np.sqrt(12.5))

    def test_symmetric(self):
        """RMSE must NOT care about direction — that is phm_score's job."""
        early = rmse([50.0], [40.0])
        late = rmse([50.0], [60.0])
        assert early == pytest.approx(late)

    def test_accepts_lists_and_arrays(self):
        assert rmse([1, 2, 3], [1, 2, 3]) == rmse(np.array([1, 2, 3]), np.array([1, 2, 3]))


class TestPHMScore:
    def test_zero_when_perfect(self):
        y = np.array([10.0, 50.0, 125.0])
        assert phm_score(y, y) == pytest.approx(0.0)

    @pytest.mark.parametrize("magnitude", [1, 5, 10, 20, 40])
    def test_late_strictly_worse_than_equally_early(self, magnitude):
        """The core asymmetry: +d must penalise more than -d, for every d > 0."""
        true = np.array([60.0])
        late = phm_score(true, true + magnitude)
        early = phm_score(true, true - magnitude)
        assert late > early, f"late ({late:.3f}) should exceed early ({early:.3f})"

    def test_penalty_is_positive_in_both_directions(self):
        """Any error is a penalty; only an exact hit scores zero."""
        true = np.array([60.0])
        assert phm_score(true, true + 5) > 0
        assert phm_score(true, true - 5) > 0

    def test_monotonic_in_error_magnitude(self):
        true = np.array([60.0])
        late = [phm_score(true, true + d) for d in [1, 2, 5, 10, 20]]
        early = [phm_score(true, true - d) for d in [1, 2, 5, 10, 20]]
        assert late == sorted(late), "penalty must grow with lateness"
        assert early == sorted(early), "penalty must grow with earliness"

    def test_matches_challenge_formula(self):
        """Explicit check against the PHM'08 definition (13 early / 10 late)."""
        true, pred = np.array([50.0]), np.array([57.0])
        assert phm_score(true, pred) == pytest.approx(np.exp(7 / 10.0) - 1.0)
        assert phm_score(np.array([50.0]), np.array([43.0])) == pytest.approx(
            np.exp(7 / 13.0) - 1.0
        )

    def test_sums_over_engines(self):
        """Score is a sum, not a mean — more engines means a larger score."""
        one = phm_score([50.0], [55.0])
        three = phm_score([50.0] * 3, [55.0] * 3)
        assert three == pytest.approx(3 * one)

    def test_dominated_by_large_late_errors(self):
        """Exponential growth means one big late miss outweighs many small ones.

        This is why the aggregate score can worsen while RMSE improves, and it is
        the behaviour the error analysis relies on.
        """
        many_small = phm_score([60.0] * 10, [63.0] * 10)
        one_large = phm_score([60.0], [95.0])
        assert one_large > many_small
