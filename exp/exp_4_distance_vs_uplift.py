"""Experiment 4 — Cross-sectional regression of VaR uplift on geodesic distance.

Consumes ``results/exp_1_var_table.csv`` produced by exp_1 and fits
    DeltaVaR (bps) = alpha + beta * d(C_start, C_crisis) + eps

The paper reports an R^2 of about 0.839 across the 12 scenario cells.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIG_DIR = RESULTS / "figures"


def main() -> Dict:
    src_csv = RESULTS / "exp_1_var_table.csv"
    if not src_csv.exists():
        raise FileNotFoundError(
            "Run exp_1 first — its CSV feeds the distance-vs-uplift regression."
        )
    df = pd.read_csv(src_csv)
    df = df.dropna(subset=["geodesic_distance", "var_uplift_bps"]).copy()

    x = df["geodesic_distance"].to_numpy()
    y = df["var_uplift_bps"].to_numpy()
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    # Cook's distance / leave-one-out sensitivity
    r2_loo = []
    for i in range(len(df)):
        idx = np.arange(len(df)) != i
        Xi = sm.add_constant(x[idx])
        mi = sm.OLS(y[idx], Xi).fit()
        r2_loo.append(float(mi.rsquared))

    stats = {
        "n_obs": int(len(df)),
        "slope_bps_per_unit_distance": float(model.params[1]),
        "intercept_bps": float(model.params[0]),
        "r_squared": float(model.rsquared),
        "slope_stderr": float(model.bse[1]),
        "slope_pvalue": float(model.pvalues[1]),
        "r_squared_loo_min": float(np.min(r2_loo)),
        "r_squared_loo_max": float(np.max(r2_loo)),
    }
    (RESULTS / "exp_4_summary.json").write_text(json.dumps(stats, indent=2))

    plt.figure(figsize=(7.5, 5))
    for pname, grp in df.groupby("portfolio"):
        plt.scatter(grp["geodesic_distance"], grp["var_uplift_bps"],
                    label=pname, s=70)
    xs = np.linspace(x.min(), x.max(), 100)
    plt.plot(xs, model.params[0] + model.params[1] * xs, "k--",
             label=fr"OLS  $R^2$={model.rsquared:.3f}")
    plt.xlabel(r"Geodesic distance $d(C_{start}, C_{crisis})$")
    plt.ylabel(r"$\Delta\mathrm{VaR}$  (bps)")
    plt.title("Cross-sectional distance vs. VaR uplift")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "exp4_distance_vs_uplift.png", dpi=150)
    plt.close()

    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    main()
