"""Tests for clustered inference, the cluster bootstrap and multiple-testing helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.research.longshot_fade import stats as S


def test_clustered_se_hand_computed_toy():
    # r = [1, 2, 3, 6], clusters [a, a, b, b]: rbar = 3, deviations -2,-1,0,3,
    # cluster sums -3 and 3, sum of squares 18, G/(G-1) = 2 -> sqrt(36) = 6, / n=4 -> SE = 1.5, t = 2.
    res = S.clustered_mean_test([1, 2, 3, 6], ["a", "a", "b", "b"])
    assert res.mean == pytest.approx(3.0)
    assert res.se == pytest.approx(1.5)
    assert res.t == pytest.approx(2.0)
    assert res.n == 4 and res.n_clusters == 2


def test_market_clustered_equals_hc1():
    r = np.array([0.1, -0.2, 0.05, 0.3, -0.1])
    res = S.market_clustered_test(r)
    n = r.size
    hc1 = math.sqrt(n / (n - 1) * np.sum((r - r.mean()) ** 2)) / n
    assert res.se == pytest.approx(hc1)


def test_t_sign_follows_mean():
    rng = np.random.default_rng(0)
    r = rng.normal(0.02, 0.1, size=200)
    cl = rng.integers(0, 40, size=200)
    pos = S.clustered_mean_test(r, cl)
    neg = S.clustered_mean_test(-r, cl)
    assert np.sign(pos.t) == np.sign(pos.mean)
    assert neg.t == pytest.approx(-pos.t)
    assert neg.se == pytest.approx(pos.se)
    assert pos.p_one_sided + neg.p_one_sided == pytest.approx(1.0)


def test_p_value_uses_t_with_g_minus_1_df():
    res = S.clustered_mean_test([1, 2, 3, 6], ["a", "a", "b", "b"])
    from scipy import stats as sps

    assert res.p_one_sided == pytest.approx(sps.t.sf(2.0, df=1))


def test_degenerate_inputs():
    assert math.isnan(S.clustered_mean_test([], []).mean)
    one = S.clustered_mean_test([1.0, 2.0], ["a", "a"])
    assert one.mean == 1.5 and math.isnan(one.se)


def test_bootstrap_reproducible_with_seed():
    rng = np.random.default_rng(1)
    r = rng.normal(0.01, 0.05, size=500)
    cl = rng.integers(0, 120, size=500)
    a = S.cluster_bootstrap_ci(r, cl, iterations=1000, seed=20251001)
    b = S.cluster_bootstrap_ci(r, cl, iterations=1000, seed=20251001)
    c = S.cluster_bootstrap_ci(r, cl, iterations=1000, seed=7)
    assert a == b
    assert a != c
    assert a[0] < r.mean() < a[1]


def test_bootstrap_resamples_whole_clusters():
    # Two clusters with constant values: every replicate mean is a weighted mix of the two cluster
    # means, so it can only take values from resampling whole clusters.
    r = np.array([0.0, 0.0, 1.0])
    cl = np.array(["a", "a", "b"])
    boot = S.cluster_bootstrap_means(r, cl, iterations=200, seed=3)
    allowed = {0.0, 1.0, 1 / 3}
    assert set(np.round(boot, 12)) <= {round(x, 12) for x in allowed}


def test_bonferroni():
    assert S.bonferroni(0.001, 12) == pytest.approx(0.012)
    assert S.bonferroni(0.2, 12) == 1.0
    assert math.isnan(S.bonferroni(math.nan))


def test_deflated_sharpe_reduces_to_psr_without_dispersion():
    rng = np.random.default_rng(2)
    r = rng.normal(0.05, 1.0, size=1000)
    out = S.deflated_sharpe(r, trial_sharpes=[0.05] * 12, n_trials=12)
    assert out["expected_max_sharpe_sr0"] == pytest.approx(0.0, abs=1e-12)
    assert out["deflated_sharpe_ratio"] == pytest.approx(out["psr_vs_zero"], abs=1e-9)
    wide = S.deflated_sharpe(r, trial_sharpes=np.linspace(-0.2, 0.2, 12), n_trials=12)
    assert wide["expected_max_sharpe_sr0"] > 0
    assert wide["deflated_sharpe_ratio"] < out["deflated_sharpe_ratio"]
