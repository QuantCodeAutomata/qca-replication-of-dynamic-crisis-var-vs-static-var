# Replication of *Dynamic Crisis VaR vs Static VaR* on the Cholesky manifold

This repository implements the five experiments defined in the task
specification:

| # | Script                                       | Purpose |
|---|----------------------------------------------|---------|
| 1 | `exp/exp_1_dynamic_vs_static_var.py`         | 30-day 95 % dynamic-crisis vs static VaR over 4 portfolios × 3 crises |
| 2 | `exp/exp_2_manifold_validation.py`           | Manifold diffusion + bridge sanity checks |
| 3 | `exp/exp_3_qv_linearity.py`                  | Empirical quadratic-variation linearity test |
| 4 | `exp/exp_4_distance_vs_uplift.py`            | Cross-sectional geodesic distance ⇒ VaR uplift regression |
| 5 | `exp/exp_5_sensitivities.py`                 | VaR / ES pathwise sensitivities and stability comparison |

## Repository layout

```
src/
├── data.py           Massive-API / yfinance loader with parquet caching
├── ewma.py           RiskMetrics EWMA recursion (λ = 0.94)
├── manifold.py       Cholesky-manifold geometry: projection, log/exp maps,
│                     Brownian-bridge and pure-diffusion simulators, geodesic
│                     distance.
├── var_engine.py     30-day Monte-Carlo dynamic and static VaR / ES engines
└── sensitivity.py    Central-finite-difference sensitivity engine with
                      common random numbers
exp/                  End-to-end experiment scripts
tests/                pytest suite (unit + integration)
results/
├── figures/          PNG output
├── *.csv, *.json     Tables / metrics
└── RESULTS.md        Full experiment write-up
```

## Environment

Requires `MASSIVE_API_KEY` in the environment. If absent, `src/data.py`
falls back to `yfinance`.

Python ≥ 3.10.  Key libraries: `numpy`, `pandas`, `scipy`, `statsmodels`,
`matplotlib`, `seaborn`, `massive`, `pytest`. All are pre-installed in the
container spec.

## Running

```bash
# Full pipeline (≈ 15 min total on a laptop):
python -m exp.exp_1_dynamic_vs_static_var
python -m exp.exp_2_manifold_validation
python -m exp.exp_3_qv_linearity
python -m exp.exp_4_distance_vs_uplift
python -m exp.exp_5_sensitivities

# Tests:
pytest tests/ -q
```

Results and figures are written under `results/`. See
[`results/RESULTS.md`](results/RESULTS.md) for the full write-up with tables,
comparison to the paper's headline numbers, and reproducibility notes.

## Key assumptions

The paper does not fully specify weights, return type, exact scenario dates,
MC path count, or pre-IPO handling. Choices:

* log returns; equal weights (100 % gross); EWMA λ = 0.94; MC = 5000 paths;
* scenarios: GFC 2008 = 2008-09-15, US Downgrade 2011 = 2011-08-05,
  COVID 2020 = 2020-02-20;
* listwise deletion after the latest IPO date — scenarios where the start
  date precedes all-names inception are logged and skipped.

See `results/RESULTS.md` for the full assumption memo.
