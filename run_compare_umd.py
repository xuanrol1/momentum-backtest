"""Compare the backtest's long-short returns with the Ken French momentum factor (UMD).

Run run_backtest.py first; this script reads its monthly_strategy_returns.csv.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from momentum.benchmark import compare_with_umd, download_umd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    a = p.parse_args()
    strategy = pd.read_csv(a.results_dir / "monthly_strategy_returns.csv", parse_dates=["date"])
    summary, both = compare_with_umd(strategy, download_umd())
    (a.results_dir / "umd_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        both.date,
        (1 + both.long_short_return).cumprod() - 1,
        color="#1f4e79",
        label=f"This backtest (Sharpe {summary['sharpe_backtest']:.2f})",
    )
    ax.plot(
        both.date,
        (1 + both.umd).cumprod() - 1,
        color="#c0504d",
        label=f"Ken French UMD (Sharpe {summary['sharpe_umd']:.2f})",
    )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(
        title=f"12–1 momentum vs Ken French UMD (correlation {summary['correlation']:.2f})",
        xlabel="Month",
        ylabel="Cumulative return",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(a.results_dir / "umd_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
