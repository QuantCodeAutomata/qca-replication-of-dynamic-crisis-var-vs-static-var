# AGENTS.md — repository memory

## Purpose
Replication of the paper "Dynamic Crisis VaR vs Static VaR" that constructs
correlation dynamics as row-wise Brownian bridges on the Cholesky manifold of
the correlation matrix.

## Key implementation choices (paper is silent)
* Returns: log; Weights: equal; EWMA λ = 0.94, burn-in 60 days
* Monte-Carlo path count: 5000 (the paper does not specify its VaR path count)
* Scenario convention: bridge ENDS at the crisis correlation peak
  (GFC = 2008-09-15, Black Monday = 2011-08-08, COVID = 2020-03-16);
  as-of/EWMA snapshot = peak − 30 calendar days; horizon T = 21 trading days;
  the final discrete grid point is pinned exactly to the crisis target
* σ_bridge is recovered per scenario from the pre-as-of EWMA geodesic-QV
  slope α via σ = sqrt(α / dim(Chol_n)), dim = n(n-1)/2. The paper's fig. 4
  slope ~0.11 is the QV *rate* of the full manifold, not per-coordinate σ;
  at n=10 that conversion gives ~0.05.
* Missing data: Massive→Yahoo fallback; GOOGL/META/EOG/MPC are Yahoo-only caches;
  per-scenario assets with <200 prior returns excluded
* Random seed: 20250101

## Modules to touch first for typical tasks
* Add a portfolio → `src/data.py::PORTFOLIOS`
* Change the geometry → `src/manifold.py` (rows on the S^{i-1} hemisphere)
* Change VaR engine → `src/var_engine.py`

## Testing
`pytest tests/ -q` → 40 tests. `tests/test_manifold.py` verifies PSD + row-norm
preservation and `σ = sqrt(α / dim(Chol_n))`. `tests/test_ewma_and_var.py`
verifies static == dynamic when `ρ_crisis = ρ_start`.

## Data access
GOOGL, META, EOG and MPC are Yahoo-adjusted closes; other names are Massive-first
with a Yahoo fill. `MASSIVE_API_KEY` / `MASSIVE_TOKEN` (environment or `.env`)
is needed to refresh the Massive-backed names.
