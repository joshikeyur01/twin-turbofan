"""Small-sample confidence intervals, shared by `src/rerank.py` and `src/compare.py`.

**Why this replaced ±half-range.** Both scripts used to report `mean ±(max−min)/2`, on the
argument that with three seeds a standard deviation invites more precision than three samples
support. That is true as far as it goes, but half-range answers the wrong question: it describes
*the runs that happened*, and grows without bound as seeds are added. What the comparison
actually needs is an interval on the **mean** — which tightens as ~1/√n and can therefore settle
a ranking that half-range never will.

So this module computes a Student-t interval on the mean. The t distribution (not the normal) is
the point: at n=5 the 95% multiplier is 2.776 rather than 1.96, a 42% penalty for not knowing
the population variance. Using 1.96 at these sample sizes would manufacture the very confidence
the module exists to measure.

**A caveat that must travel with every overlap claim.** Non-overlapping 95% CIs imply a
significant difference; *overlapping* CIs do not imply the absence of one — two means can overlap
and still differ at p<0.05 (the classic threshold is ~1.4 SE, not 2 SE). So `overlaps()` is a
**conservative** separation test: it is trustworthy when it says "separated", and only suggestive
when it says "tied". Callers should render that as "indistinguishable at 95% CI" — a statement
about the evidence — rather than "equivalent", a statement about the models.

No scipy: nothing else in `src/` needs it, and a 30-row lookup table is cheaper to trust than a
new dependency for one number.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

# Two-sided 95% Student-t critical values by degrees of freedom (n−1). Beyond df=30 the value
# is within 2% of the normal 1.96, which is far inside the noise these intervals describe.
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}
_T95_LARGE = 1.960


def t_crit_95(df: int) -> float:
    """Two-sided 95% t multiplier for ``df`` degrees of freedom."""
    if df < 1:
        raise ValueError(f"need df >= 1 (i.e. n >= 2), got {df}")
    return _T95.get(df, _T95_LARGE)


@dataclass(frozen=True)
class Interval:
    """A mean with its 95% confidence interval.

    ``half`` is None when n == 1: one sample carries no information about its own spread, and
    printing ``±0`` there would be a lie in the most misleading available direction.
    """

    n: int
    mean: float
    sd: float | None
    half: float | None
    lo: float | None
    hi: float | None
    range: float

    def fmt(self, dp: int = 3) -> str:
        """``mean ±ci`` as the reports render it; bare mean when undefined."""
        if self.half is None:
            return f"{self.mean:.{dp}f}"
        return f"{self.mean:.{dp}f} ±{self.half:.{dp}f}"


def mean_ci95(values: list[float]) -> Interval:
    """Mean of ``values`` with a Student-t 95% interval on that mean."""
    if not values:
        raise ValueError("mean_ci95 needs at least one value")
    n = len(values)
    mean = statistics.fmean(values)
    rng = max(values) - min(values)
    if n == 1:
        return Interval(n=1, mean=mean, sd=None, half=None, lo=None, hi=None, range=0.0)
    sd = statistics.stdev(values)
    half = t_crit_95(n - 1) * sd / math.sqrt(n)
    return Interval(n=n, mean=mean, sd=sd, half=half, lo=mean - half, hi=mean + half, range=rng)


def overlaps(a: Interval, b: Interval) -> bool:
    """Do two 95% intervals overlap?

    Conservative by construction — see the module docstring. An interval with no defined
    half-width (n == 1) overlaps everything, because one seed can rule nothing out.
    """
    if a.lo is None or a.hi is None or b.lo is None or b.hi is None:
        return True
    return a.lo <= b.hi and b.lo <= a.hi


def as_dict(iv: Interval, prefix: str, dp: int = 3) -> dict[str, float | int | None]:
    """Serialise an interval into flat, rounded JSON keys under ``prefix``."""

    def r(x: float | None) -> float | None:
        return None if x is None else round(x, dp)

    return {
        f"{prefix}_mean": r(iv.mean),
        f"{prefix}_sd": r(iv.sd),
        f"{prefix}_ci95": r(iv.half),
        f"{prefix}_lo": r(iv.lo),
        f"{prefix}_hi": r(iv.hi),
        f"{prefix}_range": r(iv.range),
    }
