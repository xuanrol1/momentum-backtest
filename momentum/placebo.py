"""Monte Carlo placebo test: pure-noise returns should produce no momentum premium."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .backtest import build_momentum_panel, form_long_short


def build_lookahead_panel(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """Replicate the original bug: the 12-month signal includes holding month t."""
    panel = build_momentum_panel(monthly_returns)
    panel["momentum_12_1"] = panel.groupby("symbol", sort=False)["ret"].transform(
        lambda s: np.expm1(np.log1p(s).rolling(12, min_periods=12).sum())
    )
    return panel


def random_returns(rng: np.random.Generator, n_stocks: int, n_months: int,
                   mean: float = 0.01, vol: float = 0.08) -> pd.DataFrame:
    """IID normal monthly returns, so the true momentum premium is exactly zero."""
    dates = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    ret = np.clip(rng.normal(mean, vol, size=(n_months, n_stocks)), -0.95, None)
    return pd.DataFrame({"date": np.repeat(dates, n_stocks),
        "symbol": np.tile([f"S{i:03}" for i in range(n_stocks)], n_months), "ret": ret.ravel()})


def _sharpe(panel: pd.DataFrame, quantiles: int) -> float:
    r = form_long_short(panel, quantiles)[0].long_short_return
    return float(r.mean() / r.std(ddof=1) * np.sqrt(12))


def monte_carlo_placebo(n_sims: int = 200, n_stocks: int = 50, n_months: int = 134,
                        quantiles: int = 10, seed: int = 0) -> pd.DataFrame:
    """Annualized long-short Sharpe of the biased and corrected pipelines on each simulated sample."""
    rng = np.random.default_rng(seed); rows = []
    for sim in range(n_sims):
        data = random_returns(rng, n_stocks, n_months)
        rows.append({"simulation": sim,
            "sharpe_lookahead": _sharpe(build_lookahead_panel(data), quantiles),
            "sharpe_corrected": _sharpe(build_momentum_panel(data), quantiles)})
    return pd.DataFrame(rows)
