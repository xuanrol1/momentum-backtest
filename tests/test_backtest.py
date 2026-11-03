import numpy as np
import pandas as pd
from momentum.backtest import (
    build_momentum_panel,
    form_long_short,
    performance_statistics,
    turnover_statistics,
)


def synthetic_returns(n_stocks=20, n_months=18):
    dates = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    return pd.DataFrame(
        [
            {"date": d, "symbol": f"S{s:02}", "ret": s / 1000 + m / 10000}
            for s in range(n_stocks)
            for m, d in enumerate(dates)
        ]
    )


def test_signal_window():
    data = synthetic_returns(1, 14)
    panel = build_momentum_panel(data)
    assert np.isclose(panel.loc[12, "momentum_12_1"], np.prod(1 + data.loc[0:10, "ret"]) - 1)


def test_missing_month_does_not_misalign_window():
    data = synthetic_returns(1, 20)
    dates = sorted(data.date.unique())
    gap = data[data.date != dates[5]].reset_index(drop=True)  # drop calendar month 5
    panel = build_momentum_panel(gap).set_index("date")
    # Windows that contain the missing month must be NaN, not silently shifted.
    for i in (12, 13, 14, 15, 16, 17):
        assert np.isnan(panel.loc[dates[i], "momentum_12_1"])
    # First window clear of the gap (months 6..16 for holding month 18) is exact.
    expected = np.prod(1 + data.loc[data.date.isin(dates[6:17]), "ret"]) - 1
    assert np.isclose(panel.loc[dates[18], "momentum_12_1"], expected)


def test_duplicate_rows_rejected():
    data = synthetic_returns(1, 14)
    try:
        build_momentum_panel(pd.concat([data, data.iloc[[0]]]))
    except ValueError:
        return
    raise AssertionError("duplicate rows should raise")


def test_holding_return_cannot_change_membership():
    base = synthetic_returns()
    shocked = base.copy()
    date = shocked.date.max()
    shocked.loc[(shocked.date == date) & (shocked.symbol == "S00"), "ret"] = 50
    _, a = form_long_short(build_momentum_panel(base))
    _, b = form_long_short(build_momentum_panel(shocked))
    cols = ["symbol", "momentum_12_1", "bucket"]
    pd.testing.assert_frame_equal(
        a[a.date == date][cols].reset_index(drop=True),
        b[b.date == date][cols].reset_index(drop=True),
    )


def test_statistics():
    out = performance_statistics(
        pd.DataFrame(
            {
                "long_short_return": [0.01, -0.02, 0.03, 0.01],
                "market_return": [0.02, -0.01, 0.01, 0],
            }
        )
    )
    assert out["months"] == 4 and all(np.isfinite(x) for x in out.values())


def test_placebo_detects_lookahead_bias():
    from momentum.placebo import monte_carlo_placebo

    sims = monte_carlo_placebo(n_sims=5, n_stocks=50, n_months=60, seed=1)
    assert sims.sharpe_lookahead.min() > 3 and sims.sharpe_corrected.abs().max() < 1.5


def test_missing_holding_return_cannot_change_membership():
    """A stock that stops trading during month t must stay ranked: survival is not known ex ante."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-31", periods=16, freq="ME")
    base = pd.DataFrame(
        [
            {"date": d, "symbol": f"S{s:02}", "ret": rng.normal(0.01, 0.08)}
            for s in range(20)
            for d in dates
        ]
    )
    last = dates[-1]
    _, ranked = form_long_short(build_momentum_panel(base))
    top = ranked[ranked.date == last].sort_values("momentum_12_1").symbol.iloc[-1]
    delisted = base[~((base.date == last) & (base.symbol == top))]
    strategy, ranked2 = form_long_short(build_momentum_panel(delisted), missing_return=-0.3)
    cols = ["symbol", "momentum_12_1", "bucket"]
    pd.testing.assert_frame_equal(
        ranked[ranked.date == last][cols].reset_index(drop=True),
        ranked2[ranked2.date == last][cols].reset_index(drop=True),
    )
    # The delisted winner earns the assumed delisting return.
    row = ranked2[(ranked2.date == last) & (ranked2.symbol == top)].iloc[0]
    assert row.ret == -0.3 and row.missing_return
    assert strategy.loc[strategy.date == last, "n_long_missing"].item() == 1


def test_stock_without_price_at_formation_is_not_ranked():
    """No price at the end of month t-1 means the stock cannot be bought at the start of t."""
    data = synthetic_returns(20, 16)
    dates = sorted(data.date.unique())
    gone = data[~((data.symbol == "S05") & (data.date >= dates[-2]))]  # stops after month -3
    _, ranked = form_long_short(build_momentum_panel(gone))
    assert "S05" not in set(ranked[ranked.date == dates[-1]].symbol)


def test_newey_west_matches_ordinary_t_for_iid_returns():
    rng = np.random.default_rng(3)
    r = rng.normal(0.01, 0.05, 600)
    out = performance_statistics(
        pd.DataFrame({"long_short_return": r, "market_return": rng.normal(size=600)})
    )
    assert out["newey_west_lags"] == 5  # floor(4 · 6^(2/9)) = floor(5.96)
    assert abs(out["t_statistic_newey_west"] - out["t_statistic"]) < 0.25 * abs(out["t_statistic"])


def test_turnover_full_replacement_and_no_change():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    rows = []
    # Long leg: {A, B} -> {A, B} -> {C, D}; short leg stays {X, Y}.
    for d, longs in zip(dates, (["A", "B"], ["A", "B"], ["C", "D"])):
        rows += [{"date": d, "symbol": s, "bucket": 10} for s in longs]
        rows += [{"date": d, "symbol": s, "bucket": 1} for s in ("X", "Y")]
    ranked = pd.DataFrame(rows)
    strategy = pd.DataFrame({"date": dates, "long_short_return": [0.0, 0.01, 0.01]})
    out = turnover_statistics(strategy, ranked)
    assert np.isclose(out["mean_monthly_turnover_long"], 1.0)  # 0 then 2
    assert np.isclose(out["mean_monthly_turnover_short"], 0.0)
    assert np.isclose(out["break_even_cost_bps"], 0.01 / 1.0 * 1e4)


FRENCH_SAMPLE = """This file was created using the 202609 CRSP database.
It contains a momentum factor, constructed from six value-weight portfolios.

,Mom
202001,   1.50
202002,  -2.25
202003, -99.99
202004,   0.75

 Annual Factors: January-December
,Mom
2020,   5.00
"""


def test_parse_french_monthly_keeps_only_monthly_rows_in_decimals():
    from momentum.benchmark import parse_french_monthly

    umd = parse_french_monthly(FRENCH_SAMPLE)
    assert list(umd.date.dt.strftime("%Y-%m")) == [
        "2020-01",
        "2020-02",
        "2020-04",
    ]  # -99.99 dropped
    assert np.allclose(umd.umd, [0.015, -0.0225, 0.0075])


def test_compare_with_umd_recovers_beta_and_correlation():
    from momentum.benchmark import compare_with_umd

    rng = np.random.default_rng(2)
    dates = pd.date_range("2010-01-31", periods=120, freq="ME")
    umd = pd.DataFrame({"date": dates, "umd": rng.normal(0.005, 0.04, 120)})
    strategy = pd.DataFrame(
        {"date": dates, "long_short_return": 1.5 * umd.umd + rng.normal(0, 0.01, 120)}
    )
    out, both = compare_with_umd(strategy, umd)
    assert out["months"] == 120 and abs(out["beta_on_umd"] - 1.5) < 0.1 and out["correlation"] > 0.9
