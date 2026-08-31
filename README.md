# Replication of *Dynamic Crisis VaR vs Static VaR* on the Cholesky manifold

This repository implements the four experiments defined in the task
specification, with corrected scenario dating, horizon, bridge endpoint, EWMA
path, data coverage, and bridge-volatility calibration.

| # | Script                                       | Purpose |
|---|----------------------------------------------|---------|
| 1 | `exp/exp_1_dynamic_vs_static_var.py`         | 21-trading-day (≈ 30-calendar) 95 % dynamic-crisis vs static VaR over 4 portfolios × 3 crises |
| 2 | `exp/exp_2_manifold_validation.py`           | Manifold diffusion + bridge sanity checks |
| 3 | `exp/exp_3_qv_linearity.py`                  | Empirical quadratic-variation linearity test |
| 4 | `exp/exp_4_distance_vs_uplift.py`            | Cross-sectional geodesic distance ⇒ VaR uplift regression |

## Repository layout

```
src/
├── data.py           Massive-API / yfinance loader with parquet caching
├── ewma.py           RiskMetrics EWMA recursion (λ = 0.94)
├── manifold.py       Cholesky-manifold geometry: projection, log/exp maps,
│                     Brownian-bridge and pure-diffusion simulators, geodesic
│                     distance, QV-slope → σ conversion.
└── var_engine.py     Monte-Carlo dynamic and static VaR / ES engines
exp/                  End-to-end experiment scripts
tests/                pytest suite (unit + integration)
results/
├── figures/          PNG output
├── *.csv, *.json     Tables / metrics
└── RESULTS.md        Full experiment write-up
```

## Environment

The Massive key can be supplied as `MASSIVE_API_KEY` / `MASSIVE_TOKEN` in the
environment or repository `.env`. If Massive fails or does not cover the
requested date range, `src/data.py` supplements the series with `yfinance`.

Python ≥ 3.10.  Key libraries: `numpy`, `pandas`, `scipy`, `statsmodels`,
`matplotlib`, `seaborn`, `massive`, `yfinance`, `python-dotenv`, `pytest`.

## Running

```bash
# Full pipeline (runtime depends strongly on CPU):
python -m exp.exp_1_dynamic_vs_static_var
python -m exp.exp_2_manifold_validation
python -m exp.exp_3_qv_linearity
python -m exp.exp_4_distance_vs_uplift

# Tests:
pytest tests/ -q
```

Results and figures are written under `results/`. See
[`results/RESULTS.md`](results/RESULTS.md) for the write-up and comparison with
the paper.

## Key assumptions

The paper does not fully specify weights, return type, exact scenario dates,
MC path count, or pre-IPO handling. Choices:

* log returns; equal weights (100 % gross); EWMA λ = 0.94; MC = 5000 paths;
* scenario convention: the bridge *ends* at each crisis correlation peak
  (GFC = 2008-09-15, Black Monday = 2011-08-08, COVID = 2020-03-16); the
  as-of date (EWMA snapshot) is 30 calendar days earlier, matching the paper's
  fig. 3; horizon T = 21 trading days; the last grid point is pinned exactly
  to the crisis target;
* `σ_bridge` is recovered per scenario as `sqrt(QV_slope / dim(Chol_n))`
  with `dim = n(n-1)/2`. The fig. 4 QV slope ~0.11 is an aggregate manifold
  rate; at n=10 that implies ~0.05 per coordinate. Dynamic APIs require this
  calibrated σ explicitly and do not default to 0.11;
* Massive-first data with Yahoo history fallback; GOOGL, META, EOG and MPC
  are stored as Yahoo-adjusted closes for the full cache window; each scenario
  excludes assets with fewer than 200 prior returns, then applies listwise
  deletion to the remaining historical universe.

See `results/RESULTS.md` for the full assumption memo.
