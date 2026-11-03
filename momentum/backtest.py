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
    if panel.duplicated(["symbol", "date"]).any():
        raise ValueError("monthly_returns has duplicate (symbol, date) rows")
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    # Work on a complete calendar-month grid so a missing month cannot shift the window.
    # A missing month inside the 11-month window makes the signal NaN (strict, no silent misalignment).
    wide = panel.pivot(index="date", columns="symbol", values="ret")
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="ME"))
    # shift(2) excludes both holding month t and skipped month t-1.
    signal = np.expm1(np.log1p(wide).shift(2).rolling(11, min_periods=11).sum())
    # A stock can be traded at the start of month t only if it had a price at the end of
    # month t-1, which is known when the portfolio is formed.
    tradable = wide.shift(1).notna()
    grid = pd.DataFrame({
        "ret": wide.stack(future_stack=True),
        "momentum_12_1": signal.stack(future_stack=True),
        "tradable": tradable.stack(future_stack=True),
    }).rename_axis(["date", "symbol"]).reset_index()
    # Keep a row whenever the stock has a return or a signal, so a stock with a signal but
    # no holding-month return (it stopped trading during month t) is still ranked.
    grid = grid[grid.ret.notna() | grid.momentum_12_1.notna()]
    return grid.sort_values(["symbol", "date"]).reset_index(drop=True)


def _assign_quantiles(values: pd.Series, quantiles: int) -> pd.Series:
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = values.dropna()
    if len(valid) < quantiles:
        return result
    ranks = valid.rank(method="first")
    result.loc[valid.index] = np.floor((ranks - 1) * quantiles / len(valid)).astype(int) + 1
    return result


def form_long_short(panel: pd.DataFrame, quantiles: int = 10,
                    missing_return: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort stocks into quantiles on information available at the start of month t.

    Eligibility uses only the signal and whether the stock was tradable at the end of
    month t-1. Whether the stock still has a return in month t is not known at formation,
    so a stock that stops trading during month t stays in its portfolio and earns
    ``missing_return`` for that month (a delisting-return assumption, 0 by default).
    """
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    eligible = panel.momentum_12_1.notna()
    if "tradable" in panel.columns:
        eligible &= panel.tradable.fillna(False).astype(bool)
    ranked = panel.loc[eligible].copy()
    ranked["missing_return"] = ranked.ret.isna()
    ranked["ret"] = ranked.ret.fillna(missing_return)
    ranked["bucket"] = ranked.groupby("date", group_keys=False)["momentum_12_1"].apply(
        _assign_quantiles, quantiles=quantiles
    )
    ranked = ranked.dropna(subset=["bucket"])
    legs = ranked.groupby(["date", "bucket"], observed=True).agg(
        **{"return": ("ret", "mean"), "n_stocks": ("ret", "size"),
           "n_missing": ("missing_return", "sum")}
    ).reset_index()
    low = legs.loc[legs.bucket == 1, ["date", "return", "n_stocks", "n_missing"]].rename(
        columns={"return": "short_return", "n_stocks": "n_short", "n_missing": "n_short_missing"})
    high = legs.loc[legs.bucket == quantiles, ["date", "return", "n_stocks", "n_missing"]].rename(
        columns={"return": "long_return", "n_stocks": "n_long", "n_missing": "n_long_missing"})
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

