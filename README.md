# Cross-Sectional Momentum in U.S. Equities

A clean, reproducible Python backtest of an equal-weighted **12–1 cross-sectional momentum** strategy. This is the corrected version of an earlier analysis whose Sharpe ratio was inflated by look-ahead bias.

## What was fixed

For a portfolio held in month **t**, the signal uses only returns from **t−12 through t−2**:

```text
signal(t) = product(1 + return[t-12:t-2]) - 1
portfolio return(t) = return[t]
```

Month `t−1` is skipped, and month `t` is never used to rank stocks for month `t`. An automated test changes the holding-month return and verifies that the signal and portfolio membership do not change.

The script also:

- averages simple returns within portfolios;
- compounds the long–short return geometrically;
- labels regimes ex ante using market returns through `t−1`;
- reports annualized Sharpe, maximum drawdown, a t-test, and market beta;
- saves the exact universe, monthly returns, statistics, and chart.

## Method

Each month, stocks are ranked by their 12–1 signal. The strategy buys the top decile and shorts the bottom decile with equal weights. The monthly return is the mean winner return minus the mean loser return.

The default sample uses the first 50 symbols from the current S&P 500 table to remain comparable with the original coursework. Use all current constituents with `--max-tickers 0`, or provide a researched point-in-time universe with `--tickers-file`.

## Important limitations

- **Survivorship bias:** the default universe uses today's S&P 500 membership, not historical membership.
- **Small portfolios:** with 50 stocks, each decile contains only about five names.
- **Data source:** Yahoo Finance data accessed through `yfinance` is not a research-grade point-in-time database.
- **Trading frictions:** costs, short borrow, slippage, taxes, and capacity are omitted.
- **Inference:** the ordinary t-statistic does not correct for every form of dependence or data snooping.

These limitations are intentionally disclosed rather than hidden behind an attractive statistic.

## Run it

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_backtest.py --start 2016-04-01 --end 2026-04-01 --max-tickers 50
```

The end date is exclusive. Prices are cached under `data/`; generated tables and figures go to `results/`.

For a custom universe, create a text file containing one ticker per line or comma-separated tickers:

```bash
python run_backtest.py --tickers-file my_universe.txt
```

Run the checks locally with `pytest`.

## Repository layout

```text
momentum/
tests/
run_backtest.py
requirements.txt
README.md
```

The corrected Sharpe ratio should be materially lower than the previously reported value near 2.9. That is a feature of a credible research process: correct timing, transparent assumptions, and reproducibility matter more than a visually impressive number.
