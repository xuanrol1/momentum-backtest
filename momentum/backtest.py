"""Pure backtest functions with explicit month-t information timing."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def build_momentum_panel(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "ret"}
    if missing := required.difference(monthly_returns.columns):
        raise ValueError(f"monthly_returns is missing columns: {sorted(missing)}")
    panel = monthly_returns.loc[:, ["date", "symbol", "ret"]].copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.to_period("M").dt.to_timestamp("M")
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    # shift(2) excludes both holding month t and skipped month t-1.
    panel["momentum_12_1"] = panel.groupby("symbol", sort=False)["ret"].transform(
        lambda s: np.expm1(np.log1p(s).shift(2).rolling(11, min_periods=11).sum())
    )
    return panel


def _assign_quantiles(values: pd.Series, quantiles: int) -> pd.Series:
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = values.dropna()
    if len(valid) < quantiles:
        return result
    ranks = valid.rank(method="first")
    result.loc[valid.index] = np.floor((ranks - 1) * quantiles / len(valid)).astype(int) + 1
    return result


def form_long_short(panel: pd.DataFrame, quantiles: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    ranked = panel.dropna(subset=["momentum_12_1", "ret"]).copy()
    ranked["bucket"] = ranked.groupby("date", group_keys=False)["momentum_12_1"].apply(
        _assign_quantiles, quantiles=quantiles
    )
    ranked = ranked.dropna(subset=["bucket"])
    legs = ranked.groupby(["date", "bucket"], observed=True)["ret"].agg(
        [("return", "mean"), ("n_stocks", "size")]
    ).reset_index()
    low = legs.loc[legs.bucket == 1, ["date", "return", "n_stocks"]].rename(
        columns={"return": "short_return", "n_stocks": "n_short"})
    high = legs.loc[legs.bucket == quantiles, ["date", "return", "n_stocks"]].rename(
        columns={"return": "long_return", "n_stocks": "n_long"})
    strategy = high.merge(low, on="date", how="inner").sort_values("date")
    strategy["long_short_return"] = strategy.long_return - strategy.short_return
    strategy["cumulative_return"] = (1 + strategy.long_short_return).cumprod() - 1
    return strategy.reset_index(drop=True), ranked.reset_index(drop=True)


def add_market_regime(strategy: pd.DataFrame, market_returns: pd.DataFrame) -> pd.DataFrame:
    market = market_returns.loc[:, ["date", "market_return"]].copy().sort_values("date")
    market["date"] = pd.to_datetime(market.date).dt.to_period("M").dt.to_timestamp("M")
    market["market_trailing_12m"] = np.expm1(
        np.log1p(market.market_return).shift(1).rolling(12, min_periods=12).sum())
    market["regime"] = np.where(market.market_trailing_12m.isna(), pd.NA,
        np.where(market.market_trailing_12m > 0, "Bull", "Bear"))
    return strategy.merge(market, on="date", how="left")


def performance_statistics(data: pd.DataFrame) -> dict[str, float | int]:
    r = data.long_short_return.dropna()
    if len(r) < 2:
        raise ValueError("at least two portfolio returns are required")
    vol = r.std(ddof=1)
    wealth = (1 + r).cumprod()
    test = stats.ttest_1samp(r, 0.0, nan_policy="omit")
    out = {"months": int(len(r)), "mean_monthly_return": float(r.mean()),
        "annualized_return": float((1+r).prod() ** (12/len(r)) - 1),
        "annualized_volatility": float(vol*np.sqrt(12)),
        "annualized_sharpe_zero_rf": float(r.mean()/vol*np.sqrt(12)),
        "maximum_drawdown": float((wealth/wealth.cummax()-1).min()),
        "t_statistic": float(test.statistic), "p_value_two_sided": float(test.pvalue)}
    reg = data[["long_short_return", "market_return"]].dropna()
    if len(reg) >= 3:
        fit = sm.OLS(reg.long_short_return, sm.add_constant(reg.market_return)).fit()
        out["market_beta"] = float(fit.params["market_return"])
        out["market_beta_p_value"] = float(fit.pvalues["market_return"])
    return out


def regime_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, group in data.dropna(subset=["regime"]).groupby("regime"):
        r = group.long_short_return.dropna(); vol = r.std(ddof=1)
        rows.append({"regime": regime, "months": len(r), "mean_monthly_return": r.mean(),
            "annualized_sharpe_zero_rf": r.mean()/vol*np.sqrt(12) if vol > 0 else np.nan,
            "t_statistic": r.mean()/(vol/np.sqrt(len(r))) if vol > 0 else np.nan})
    return pd.DataFrame(rows)

