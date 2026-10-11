import numpy as np
import pandas as pd
from momentum.backtest import build_momentum_panel, form_long_short, performance_statistics


def synthetic_returns(n_stocks=20, n_months=18):
    dates = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    return pd.DataFrame([{"date":d,"symbol":f"S{s:02}","ret":s/1000+m/10000}
        for s in range(n_stocks) for m,d in enumerate(dates)])


def test_signal_window():
    data=synthetic_returns(1,14); panel=build_momentum_panel(data)
    assert np.isclose(panel.loc[12,"momentum_12_1"], np.prod(1+data.loc[0:10,"ret"])-1)


def test_holding_return_cannot_change_membership():
    base=synthetic_returns(); shocked=base.copy(); date=shocked.date.max()
    shocked.loc[(shocked.date==date)&(shocked.symbol=="S00"),"ret"]=50
    _,a=form_long_short(build_momentum_panel(base)); _,b=form_long_short(build_momentum_panel(shocked))
    cols=["symbol","momentum_12_1","bucket"]
    pd.testing.assert_frame_equal(a[a.date==date][cols].reset_index(drop=True), b[b.date==date][cols].reset_index(drop=True))


def test_statistics():
    out=performance_statistics(pd.DataFrame({"long_short_return":[.01,-.02,.03,.01],"market_return":[.02,-.01,.01,0]}))
    assert out["months"]==4 and all(np.isfinite(x) for x in out.values())



def test_placebo_detects_lookahead_bias():
    from momentum.placebo import monte_carlo_placebo
    sims=monte_carlo_placebo(n_sims=5, n_stocks=50, n_months=60, seed=1)
    assert sims.sharpe_lookahead.min() > 3 and sims.sharpe_corrected.abs().max() < 1.5
