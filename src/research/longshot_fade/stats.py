"""Inference for the longshot-fade study (protocol section 4).

- event-clustered mean test: SE = sqrt(G/(G-1) * sum_g (sum_{i in g} (r_i - rbar))^2) / n, t = rbar / SE,
  one-sided p from Student t with G-1 degrees of freedom;
- market-clustered (heteroskedasticity-robust) SE: the same formula with singleton clusters;
- cluster bootstrap percentile CI of the mean (events resampled with replacement);
- Bonferroni adjustment and the deflated Sharpe ratio (Bailey & Lopez de Prado, 2014).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class MeanTest:
    mean: float
    se: float
    t: float
    p_one_sided: float
    n: int
    n_clusters: int

    def to_dict(self) -> dict:
        return asdict(self)


def _codes(clusters) -> tuple[np.ndarray, int]:
    _, codes = np.unique(np.asarray(clusters), return_inverse=True)
    codes = codes.ravel()
    return codes, int(codes.max()) + 1 if codes.size else 0


def clustered_mean_test(r, clusters) -> MeanTest:
    """One-sided test of mean(r) > 0 with cluster-robust SE (CR1-style G/(G-1) correction)."""
    r = np.asarray(r, dtype=np.float64)
    n = r.size
    if n == 0:
        return MeanTest(math.nan, math.nan, math.nan, math.nan, 0, 0)
    codes, g = _codes(clusters)
    if codes.size != n:
        raise ValueError("r and clusters must have the same length")
    mean = float(r.mean())
    if g < 2:
        return MeanTest(mean, math.nan, math.nan, math.nan, n, g)
    s = np.bincount(codes, weights=r - mean, minlength=g)
    se = math.sqrt(g / (g - 1) * float(np.dot(s, s))) / n
    if se == 0:
        t = math.copysign(math.inf, mean) if mean != 0 else math.nan
    else:
        t = mean / se
    p = float(sps.t.sf(t, df=g - 1)) if not math.isnan(t) else math.nan
    return MeanTest(mean, se, t, p, n, g)


def market_clustered_test(r) -> MeanTest:
    """Same estimator with every observation its own cluster (one entry per market)."""
    r = np.asarray(r, dtype=np.float64)
    return clustered_mean_test(r, np.arange(r.size))


def cluster_bootstrap_means(r, clusters, iterations: int = 1000, seed: int = 20251001) -> np.ndarray:
    """Bootstrap distribution of the mean, resampling whole clusters with replacement."""
    r = np.asarray(r, dtype=np.float64)
    codes, g = _codes(clusters)
    if g == 0:
        return np.full(iterations, np.nan)
    sums = np.bincount(codes, weights=r, minlength=g)
    counts = np.bincount(codes, minlength=g).astype(np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(iterations, dtype=np.float64)
    for b in range(iterations):
        idx = rng.integers(0, g, size=g)
        out[b] = sums[idx].sum() / counts[idx].sum()
    return out


def cluster_bootstrap_ci(
    r, clusters, iterations: int = 1000, seed: int = 20251001, level: float = 0.95
) -> tuple[float, float]:
    """Percentile CI of the mean from the cluster bootstrap."""
    boot = cluster_bootstrap_means(r, clusters, iterations=iterations, seed=seed)
    if np.isnan(boot).all():
        return math.nan, math.nan
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(boot, [a, 100 - a])
    return float(lo), float(hi)


def bonferroni(p: float, m: int = 12) -> float:
    if p is None or math.isnan(p):
        return math.nan
    return min(1.0, p * m)


def sharpe(r) -> float:
    """Per-observation Sharpe ratio mean / sd (ddof=1); not annualised."""
    r = np.asarray(r, dtype=np.float64)
    if r.size < 2:
        return math.nan
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else math.nan


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """E[max SR] under the null across n_trials independent trials (Bailey & Lopez de Prado 2014, eq. 2)."""
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    z1 = sps.norm.ppf(1 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int, skew: float, kurtosis: float) -> float:
    """PSR: P(true SR > benchmark) given estimated SR, sample length, skewness and (non-excess) kurtosis."""
    denom = 1 - skew * sr + (kurtosis - 1) / 4 * sr**2
    if n_obs < 2 or denom <= 0 or math.isnan(sr):
        return math.nan
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(sps.norm.cdf(z))


def deflated_sharpe(r, trial_sharpes, n_trials: int = 12) -> dict:
    """Deflated Sharpe ratio of return series `r`, deflating by the dispersion of `trial_sharpes`."""
    r = np.asarray(r, dtype=np.float64)
    trial_sharpes = np.asarray([s for s in trial_sharpes if not math.isnan(s)], dtype=np.float64)
    sr = sharpe(r)
    var_sr = float(trial_sharpes.var(ddof=1)) if trial_sharpes.size > 1 else 0.0
    sr0 = expected_max_sharpe(var_sr, n_trials)
    skew = float(sps.skew(r)) if r.size > 2 else math.nan
    kurt = float(sps.kurtosis(r, fisher=False)) if r.size > 3 else math.nan
    return {
        "sharpe_per_entry": sr,
        "n_obs": int(r.size),
        "skewness": skew,
        "kurtosis": kurt,
        "n_trials": n_trials,
        "trial_sharpe_variance": var_sr,
        "expected_max_sharpe_sr0": sr0,
        "psr_vs_zero": probabilistic_sharpe(sr, 0.0, r.size, skew, kurt),
        "deflated_sharpe_ratio": probabilistic_sharpe(sr, sr0, r.size, skew, kurt),
    }
