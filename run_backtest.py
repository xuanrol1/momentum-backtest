"""Run the corrected cross-sectional momentum study."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from momentum.backtest import add_market_regime, build_momentum_panel, form_long_short, performance_statistics, regime_statistics
from momentum.data import current_sp500_tickers, download_adjusted_prices, prices_to_monthly_returns, read_tickers


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2016-04-01"); p.add_argument("--end", default="2026-04-01")
    p.add_argument("--max-tickers", type=int, default=50); p.add_argument("--tickers-file")
    p.add_argument("--data-dir", type=Path, default=Path("data")); p.add_argument("--results-dir", type=Path, default=Path("results"))
    return p.parse_args()


def main():
    a = parse_args(); start = pd.Timestamp(a.start); end = pd.Timestamp(a.end)
    if start >= end: raise ValueError("--start must be earlier than --end")
    tickers = read_tickers(a.tickers_file) if a.tickers_file else current_sp500_tickers(None if a.max_tickers == 0 else a.max_tickers)
    download_start = (start-pd.DateOffset(months=14)).strftime("%Y-%m-%d")
    a.data_dir.mkdir(parents=True, exist_ok=True); a.results_dir.mkdir(parents=True, exist_ok=True)
    (a.results_dir/"universe.txt").write_text("\n".join(tickers)+"\n", encoding="utf-8")
    key = hashlib.sha256((",".join(tickers)+f"|{download_start}|{a.end}").encode()).hexdigest()[:12]
    prices = download_adjusted_prices(tickers, download_start, a.end, a.data_dir/f"stock_adjusted_close_{key}.csv")
    strategy, ranked = form_long_short(build_momentum_panel(prices_to_monthly_returns(prices)))
    strategy = strategy.loc[(strategy.date >= start) & (strategy.date < end)].copy()
    mp = download_adjusted_prices(["^GSPC"], download_start, a.end, a.data_dir/f"market_{key}.csv")
    market = prices_to_monthly_returns(mp).rename(columns={"ret":"market_return"})
    final = add_market_regime(strategy, market); summary = performance_statistics(final)
    final.to_csv(a.results_dir/"monthly_strategy_returns.csv", index=False)
    ranked.to_csv(a.results_dir/"stock_month_assignments.csv", index=False)
    regime_statistics(final).to_csv(a.results_dir/"regime_statistics.csv", index=False)
    (a.results_dir/"performance_statistics.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    fig, ax = plt.subplots(figsize=(10,5.5)); ax.plot(final.date, final.cumulative_return, color="#1f4e79")
    ax.axhline(0,color="black",linewidth=.7); ax.set(title="12–1 Momentum: Winner Minus Loser", xlabel="Month", ylabel="Cumulative return"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(a.results_dir/"cumulative_return.png", dpi=180); plt.close(fig)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()

