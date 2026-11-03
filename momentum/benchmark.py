"""Ken French momentum factor (UMD) as an external benchmark for the backtest."""

from __future__ import annotations
import io
import re
import zipfile
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
import statsmodels.api as sm

UMD_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/" "F-F_Momentum_Factor_CSV.zip"
)


def parse_french_monthly(text: str) -> pd.DataFrame:
    """Parse the monthly block of a Ken French factor CSV.

    The file has a free-text header, a monthly block of rows "YYYYMM, value" in percent,
    and then an annual block of rows "YYYY, value". Only the monthly block is kept;
    -99.99 and -999 mark missing values.
    """
    rows, started = [], False
    for line in text.splitlines():
        m = re.match(r"^\s*(\d{6})\s*,\s*(-?\d+(?:\.\d+)?)\s*$", line)
        if m:
            started = True
            rows.append((m.group(1), float(m.group(2))))
        elif started:
            break  # first non-monthly line after the block: annual data follow
    if not rows:
        raise ValueError("no monthly rows found")
    out = pd.DataFrame(rows, columns=["yyyymm", "umd"])
    out["date"] = pd.to_datetime(out.yyyymm, format="%Y%m").dt.to_period("M").dt.to_timestamp("M")
    out["umd"] = out.umd.where(out.umd > -99, np.nan) / 100
    return out[["date", "umd"]].dropna()


def download_umd(url: str = UMD_URL) -> pd.DataFrame:
    request = Request(url, headers={"User-Agent": "momentum-research-project/1.0"})
    with urlopen(request, timeout=30) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
    return parse_french_monthly(archive.read(name).decode("latin-1"))


def compare_with_umd(strategy: pd.DataFrame, umd: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Correlation, Sharpe ratios and regression of the backtest on UMD over common months."""
    s = strategy.loc[:, ["date", "long_short_return"]].copy()
    s["date"] = pd.to_datetime(s.date).dt.to_period("M").dt.to_timestamp("M")
    both = s.merge(umd, on="date", how="inner").dropna()
    if len(both) < 12:
        raise ValueError("fewer than 12 overlapping months")
    sharpe = lambda r: float(r.mean() / r.std(ddof=1) * np.sqrt(12))
    fit = sm.OLS(both.long_short_return, sm.add_constant(both.umd)).fit()
    out = {
        "months": int(len(both)),
        "first_month": both.date.min().strftime("%Y-%m"),
        "last_month": both.date.max().strftime("%Y-%m"),
        "correlation": float(both.long_short_return.corr(both.umd)),
        "sharpe_backtest": sharpe(both.long_short_return),
        "sharpe_umd": sharpe(both.umd),
        "mean_monthly_backtest": float(both.long_short_return.mean()),
        "mean_monthly_umd": float(both.umd.mean()),
        "beta_on_umd": float(fit.params["umd"]),
        "alpha_monthly": float(fit.params["const"]),
        "alpha_t_statistic": float(fit.tvalues["const"]),
    }
    return out, both
