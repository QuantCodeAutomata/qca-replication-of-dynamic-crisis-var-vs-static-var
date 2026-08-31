"""RiskMetrics-style EWMA covariance and correlation estimation.

Implemented from scratch — Context7 confirms no mainstream lib provides the
exact ``S_t = lambda S_{t-1} + (1 - lambda) r_{t-1} r_{t-1}^T`` recursion
in the shape needed here (`arch` is univariate-focused; `sklearn` covariance
estimators do not implement the RiskMetrics recursion).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    """Compute daily returns from an adjusted-price panel.

    Parameters
    ----------
    prices : DataFrame indexed by date, columns are tickers.
    kind : ``'log'`` (default) or ``'simple'``. Choice documented as an
        implementation assumption.
    """
    if kind == "log":
        rets = np.log(prices).diff()
    elif kind == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError(f"Unknown return kind: {kind}")
    return rets.dropna(how="all")


def ewma_covariance(
    returns: np.ndarray,
    lam: float = 0.94,
    burn_in: int = 60,
) -> np.ndarray:
    """RiskMetrics EWMA recursion on a returns matrix (T x n).

    Initialization: sample covariance over the first ``burn_in`` observations.
    """
    T, n = returns.shape
    if T <= burn_in:
        raise ValueError(f"Not enough observations ({T}) for burn-in={burn_in}.")
    S = np.cov(returns[:burn_in].T, ddof=0)
    if S.ndim == 0:  # single asset edge case
        S = np.array([[float(S)]])
    for t in range(burn_in, T):
        r = returns[t - 1].reshape(-1, 1)
        S = lam * S + (1.0 - lam) * (r @ r.T)
    return S


def ewma_covariance_path(
    returns: np.ndarray,
    lam: float = 0.94,
    burn_in: int = 60,
) -> np.ndarray:
    """Return a path of EWMA covariance matrices, shape (T - burn_in, n, n).

    ``S_t`` at output index ``k`` corresponds to input index ``burn_in + k``.
    """
    T, n = returns.shape
    if T <= burn_in:
        raise ValueError(f"Not enough observations ({T}) for burn-in={burn_in}.")
    S = np.cov(returns[:burn_in].T, ddof=0)
    if S.ndim == 0:
        S = np.array([[float(S)]])
    out = np.zeros((T - burn_in, n, n))
    out[0] = S
    for t in range(burn_in, T - 1):
        r = returns[t].reshape(-1, 1)
        S = lam * S + (1.0 - lam) * (r @ r.T)
        out[t - burn_in + 1] = S
    return out


def cov_to_corr(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose ``S`` into volatilities ``sigma`` and correlation matrix ``C``."""
    sigma = np.sqrt(np.diag(S))
    inv = np.where(sigma > 0, 1.0 / sigma, 0.0)
    C = (S * inv[:, None]) * inv[None, :]
    # enforce symmetry and unit diagonal
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return sigma, C
