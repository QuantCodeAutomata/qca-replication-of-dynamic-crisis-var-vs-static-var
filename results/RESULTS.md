# Results — Replication of Dynamic Crisis VaR vs Static VaR

This document summarises the numerical output of the package's four
experiments. Every replication number comes from an artefact in this
directory; values attributed to the paper come from *Stochastic Correlations
and Crisis VaR* (Davis, 8 July 2026).

## Implementation settings and source conventions

* **Returns** — daily log returns $`\log(P_t/P_{t-1})`$.
* **EWMA** — RiskMetrics recursion with $\lambda = 0.94$ (paper §5), a 60-observation
  burn-in and initial $\Sigma$ equal to the burn-in sample covariance. The scalar
  and path estimators include the same final return and therefore have the
  same terminal covariance.
* **Weights** — equal weights, gross notional = 100 %.
* **Missing data / pre-IPO** — Massive is primary; Yahoo Finance supplements
  incomplete history. GOOGL, META, EOG and MPC are cached as Yahoo-adjusted
  closes over the full requested window (Massive’s GOOGL series starts only
  in 2014; META includes the Facebook listing from 2012-05-18). Assets with
  fewer than 200 pre-scenario returns are excluded
  before listwise alignment. META is excluded from Technology in 2008/2011
  and MPC from Commodities in 2008/2011.
* **Bridge** — $T = 21$ trading days, corresponding approximately to one
  calendar month. The paper states a “30-day horizon” in §5.5, while fig. 3
  spans 14 February–17 March 2020. The crisis target is $\rho_{\mathrm{crisis}} = 0.9$
  (paper eq. (9) and table 7); the Monte Carlo uses 5 000 paths. The final
  grid point is pinned exactly to that target.
* **Bridge volatility** — the paper uses $\sigma = 0.11$ in §5.5 / table 7. Here $\sigma$
  is estimated separately for every scenario from the pre-as-of EWMA
  geodesic-QV slope $\alpha$:

```math
\dim(\mathrm{Chol}_{n}) = n(n-1)/2, \qquad \sigma_{\mathrm{bridge}} = \sqrt{\alpha / \dim(\mathrm{Chol}_{n})}.
```

Experiment 3 slopes of 0.07–0.12 at $n = 10$ (dim = 45) give
$\sigma_{\mathrm{bridge}}$ near 0.04–0.05; the paper’s 0.11 is not used as $\sigma$.
* **Random seed** — fixed globally at `20250101`.

Each scenario's calendar window ends at a crisis correlation peak and begins
30 calendar days earlier:
GFC 2008 = 2008-08-16 → 2008-09-15; Downgrade 2011 =
2011-07-09 → 2011-08-08; COVID 2020 = 2020-02-15 → 2020-03-16.
This follows the paper’s fig. 3 (COVID bridge in the weeks before the March
peak) and §5.5 (Finance initialized at its February 2020 correlation matrix).
Peaks: Lehman 15 Sep 2008, Black Monday 8 Aug 2011, COVID 16 Mar 2020
(paper §5.1–5.3 and table 5).

These dates define the start snapshot and horizon. The simulated terminal
matrix is the equicorrelation target with $\rho_{\mathrm{crisis}} = 0.9$, not the empirical
EWMA matrix observed on the peak date.

---

## Experiment 1 — Dynamic Crisis VaR vs Static VaR

All twelve scenarios run under the historical-availability rule. The complete
output, including each QV slope and $\sigma$, is in
[`exp_1_var_table.csv`](exp_1_var_table.csv).

Values in parentheses are from paper Table 6; the first value in each cell
is from `exp_1_var_table.csv` (5 000 paths, seed 20250101).

| Portfolio | Crisis | n | σ | Static VaR (paper) | Dynamic VaR (paper) | ΔVaR (paper) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Technology | GFC 2008 | 9 | 0.0549 | 13.11% (12.33%) | 17.58% (14.64%) | +447 (+231) bps |
| Old Economy | GFC 2008 | 10 | 0.0512 | 8.29% (8.07%) | 10.97% (10.45%) | +267 (+238) bps |
| Commodities | GFC 2008 | 9 | 0.0451 | 22.08% (23.42%) | 25.27% (24.58%) | +319 (+116) bps |
| Technology | Downgrade 2011 | 9 | 0.0484 | 8.94% (9.78%) | 11.46% (10.79%) | +252 (+101) bps |
| Old Economy | Downgrade 2011 | 10 | 0.0467 | 5.89% (6.06%) | 6.69% (6.73%) | +80 (+67) bps |
| Finance | Downgrade 2011 | 10 | 0.0399 | 10.23% (9.85%) | 10.96% (10.25%) | +73 (+40) bps |
| Commodities | Downgrade 2011 | 9 | 0.0430 | 10.10% (11.38%) | 12.26% (12.18%) | +215 (+80) bps |
| Technology | COVID 2020 | 10 | 0.0491 | 8.01% (7.90%) | 9.91% (9.73%) | +190 (+183) bps |
| Old Economy | COVID 2020 | 10 | 0.0501 | 4.82% (4.65%) | 6.96% (6.76%) | +214 (+211) bps |
| Finance | COVID 2020 | 10 | 0.0396 | 7.47% (7.14%) | 8.08% (7.74%) | +60 (+60) bps |
| Commodities | COVID 2020 | 10 | 0.0503 | 10.81% (10.29%) | 12.82% (12.00%) | +201 (+171) bps |
| Finance | GFC 2008 | 10 | 0.0446 | 27.18% (24.99%) | 27.59% (25.52%) | +41 (+53) bps |

**Comparison with the paper's Table 6.**

* All 12 uplift signs are positive, matching Table 6.
* Static VaR MAE vs Table 6 is 0.69 percentage points. Uplift MAE is 69 bps
  across all 12 cells and 16 bps across the eight full 10-asset cells.
* Discrepancies in the eight complete 10-asset cells are much smaller than in
  the four 9-asset cells (uplift MAE 16 bps vs 176 bps). The four largest
  Table 6 errors are Technology/GFC, Technology/2011, Commodities/GFC and
  Commodities/2011. The portfolios published in tables 1–4 contain ten names.
  One possible explanation—unverifiable from the paper—is that the author's
  historical runs replaced pre-IPO META or MPC with another name in the same
  category without documenting the substitute.
* Finance has the smallest uplifts here and in Table 6, consistent with
  starting correlations already closest to the crisis target (paper fig. 6).

Technology/GFC and Technology/2011 exclude pre-IPO META; Commodities/GFC
and Commodities/2011 exclude pre-IPO MPC; Technology/COVID uses all ten
names from table 1.

Figures:
[`exp1_loss_histograms.png`](figures/exp1_loss_histograms.png),
[`exp1_uplift_bars.png`](figures/exp1_uplift_bars.png).

---

## Experiment 2 — Manifold / Bridge Validation

The experiment evolves 500 random starting correlation matrices for 30
diffusion steps. Full diagnostics are in
[`exp_2_manifold_validation.json`](exp_2_manifold_validation.json). The
paper’s §2 check uses LKJ starts and 500 paths; this implementation samples
starts on the Cholesky hemispheres and evolves one path per start.

| Diagnostic | Value |
| --- | --- |
| min eigenvalue across all simulated C(t) | −6.6e-16 (numerical) |
| mean min-eigenvalue | 1.7e-4 |
| max deviation of C(t) diagonal from 1 | 4.4e-16 |
| max symmetry error | 0.0 (exact) |
| off-diag histogram mean / std / median | −0.003 / 0.395 / −0.005 |

Structural correlation constraints are preserved to machine precision.

**Brownian-bridge check** (200 paths, 30 days, $\rho_{\mathrm{crisis}} = 0.9$):

* Mean average pairwise correlation: 0.012 → **0.900**.
* Mean geodesic distance to target: 3.82 → **4.2e-6** (numerical zero).

The last discrete grid point is pinned to the crisis target, so the simulated
bridge ends at $C_T = C_{\mathrm{crisis}}$ (equicorrelation 0.9), as required
by the Brownian-bridge construction in paper eq. (7).

Figures:
[`exp2_diffusion_offdiag_hist.png`](figures/exp2_diffusion_offdiag_hist.png),
[`exp2_bridge_paths.png`](figures/exp2_bridge_paths.png).

---

## Experiment 3 — Empirical QV Linearity

The experiment regresses cumulative quadratic variation of daily
Cholesky-geodesic increments of EWMA correlation matrices on the time index.
The estimates are in
[`exp_3_qv_regression.csv`](exp_3_qv_regression.csv). Paper §5.4 reports
QV-linearity $R^2$ of 0.997 (technology), 0.993 (old economy), 0.995
(commodities) and 0.998 (financial services).

| Portfolio | n-obs | slope | implied σ | R² | Paper R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Technology | 3215 | 0.1018 | 0.0476 | **0.9982** | 0.997 |
| Old Economy | 4467 | 0.1178 | 0.0512 | **0.9975** | 0.993 |
| Finance | 4467 | 0.0710 | 0.0397 | **0.9988** | 0.998 |
| Commodities | 3442 | 0.1090 | 0.0492 | **0.9988** | 0.995 |

All four $R^2$ values are within 0.005 of the paper. Technology’s shorter
window (n-obs = 3215, start 2012-08-16) is the listwise panel after META’s
2012 listing. The slope is the QV rate of all 45 tangent coordinates; the
displayed $\sigma$ is $\sqrt{\mathrm{slope}/45}$.

Figure: [`exp3_qv_linearity.png`](figures/exp3_qv_linearity.png).

---

## Experiment 4 — Geodesic Distance ⇒ VaR Uplift

The primary cross-sectional OLS regression uses the eight Experiment 1 cells
that retain all ten assets. This keeps the geodesic-distance dimension
consistent with the paper's ten-name portfolios. Define the fitted
coefficients

```math
\widehat{\alpha}_{\mathrm{OLS}}
:= \texttt{intercept\_bps} = -50.6\,\mathrm{bps},
\qquad
\widehat{\beta}_{\mathrm{OLS}}
:= \texttt{slope\_bps\_per\_unit\_distance}
= 98.8\,\mathrm{bps}.
```

The fitted model is therefore

```math
\Delta\mathrm{VaR}_{i}
= \widehat{\alpha}_{\mathrm{OLS}}
+ \widehat{\beta}_{\mathrm{OLS}}\,
d(C_{\mathrm{start},i}, C_{\mathrm{crisis}})
+ \varepsilon_i.
```

The unrounded estimates are stored in
[`exp_4_summary.json`](exp_4_summary.json). They are sample estimates, not
parameters specified by the paper. The title of figure 6 on page 13 reports
$R^2 = 0.839$ for the paper's 12 ten-asset scenario cells and states that the
VaR increase is governed by the geodesic distance in eq. (2).

* **Primary 10-asset fit:** $R^2 = 0.900$, slope p-value = 0.00032
  on eight cells.
* **Secondary pooled fit:** $R^2 = 0.200$, slope p-value = 0.145
  on all 12 cells.

The dimension-consistent 10-name fit is strong, while the secondary pooled
slope is not statistically significant at the 5 % level. The four 9-name
cells materially weaken the pooled relationship. This is the same split seen
in Experiment 1 and is consistent with—but does not prove—the possibility
that the paper’s historical runs used undocumented same-category substitutes
for META and MPC.

Figure:
[`exp4_distance_vs_uplift.png`](figures/exp4_distance_vs_uplift.png).

---

## Reproducibility

* Every experiment is reproducible via `python -m exp.exp_N_<name>`.
* Random seeds are fixed at 20250101.
* The 40 paper tickers (tables 1–4) are fetched via Massive with a Yahoo
  history fill. GOOGL, META, EOG and MPC are Yahoo-adjusted series for the
  full requested window.
* Dynamic APIs require an explicit calibrated `sigma_bridge`; they do not
  silently substitute the paper’s $\sigma = 0.11$.
* The 40 tests cover manifold constraints, exact bridge endpoint, QV-to-σ
  conversion, scalar/path EWMA terminal equality, fallback/cache handling and
  scenario universes.
* Remaining deviations from the published setup are four 9-name pre-IPO
  cells, QV-calibrated $\sigma$ rather than 0.11, $T = 21$ trading steps rather than
  the paper’s stated 30-day horizon, and assumed equal weights and log
  returns. The simulation uses 5 000 paths; the paper does not state its path
  count.
