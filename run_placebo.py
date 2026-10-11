"""Run the Monte Carlo placebo test that exposes the original look-ahead bias."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from momentum.placebo import monte_carlo_placebo


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sims", type=int, default=200); p.add_argument("--stocks", type=int, default=50)
    p.add_argument("--months", type=int, default=134); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    return p.parse_args()


def main():
    a = parse_args(); a.results_dir.mkdir(parents=True, exist_ok=True)
    sims = monte_carlo_placebo(a.sims, a.stocks, a.months, seed=a.seed)
    sims.to_csv(a.results_dir/"placebo_simulations.csv", index=False)
    summary = {"simulations": a.sims, "stocks": a.stocks, "months": a.months, "seed": a.seed}
    for col in ["sharpe_lookahead", "sharpe_corrected"]:
        s = sims[col]
        summary[col] = {"mean": s.mean(), "std": s.std(ddof=1), "p05": s.quantile(.05), "p95": s.quantile(.95)}
    (a.results_dir/"placebo_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    fig, ax = plt.subplots(figsize=(10,5.5))
    ax.hist(sims.sharpe_corrected, bins=30, alpha=.75, color="#1f4e79", label="Corrected (t−12 to t−2)")
    ax.hist(sims.sharpe_lookahead, bins=30, alpha=.75, color="#c0504d", label="Look-ahead (includes month t)")
    ax.axvline(0, color="black", linewidth=.7)
    ax.set(title="Annualized Sharpe on Pure-Noise Returns", xlabel="Annualized Sharpe", ylabel="Simulations")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(a.results_dir/"placebo_sharpe.png", dpi=180); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
