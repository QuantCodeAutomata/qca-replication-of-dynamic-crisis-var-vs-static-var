# AGENTS.md — repository memory

## Purpose
Replication of the paper "Dynamic Crisis VaR vs Static VaR" that constructs
correlation dynamics as row-wise Brownian bridges on the Cholesky manifold of
the correlation matrix.

## Key implementation choices (paper is silent)
* Returns: log; Weights: equal; EWMA λ = 0.94, burn-in 60 days
* Monte-Carlo path count: 5000 (paper suggests up to 50k)
* Scenario dates: GFC = 2008-09-15, Downgrade = 2011-08-05, COVID = 2020-02-20
* Missing data: listwise deletion after latest IPO — pre-IPO scenarios skipped
* Random seed: 20250101

## Modules to touch first for typical tasks
* Add a portfolio → `src/data.py::PORTFOLIOS`
* Change the geometry → `src/manifold.py` (rows on the S^{i-1} hemisphere)
* Change VaR engine → `src/var_engine.py`
* Change sensitivity method (AD vs FD) → `src/sensitivity.py`

## Testing
`pytest tests/ -q` → 28 tests. `tests/test_manifold.py` verifies PSD + row-norm
preservation, `tests/test_var_engine.py` verifies static == dynamic when
`ρ_crisis = ρ_start`, `tests/test_sensitivity.py` verifies ES-more-stable-than
-VaR.

## Data access
`MASSIVE_API_KEY` env var required; fallback to yfinance is silent.
