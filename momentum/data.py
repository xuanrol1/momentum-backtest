"""Market-data acquisition helpers."""
from __future__ import annotations
from io import StringIO
from pathlib import Path
import re
from urllib.request import Request, urlopen
import pandas as pd
import yfinance as yf

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def current_sp500_tickers(max_tickers: int | None = 50) -> list[str]:
    request = Request(SP500_URL, headers={"User-Agent": "momentum-research-project/1.0"})
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8")
    table = pd.read_html(StringIO(html), attrs={"id": "constituents"})[0]
    tickers = table.Symbol.astype(str).str.replace(".", "-", regex=False).tolist()
    return tickers if not max_tickers else tickers[:max_tickers]


def read_tickers(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    items = [x.strip().upper().replace(".", "-") for x in re.split(r"[\s,]+", text)]
    return list(dict.fromkeys(x for x in items if x))


def download_adjusted_prices(tickers, start, end, cache_path) -> pd.DataFrame:
    cache = Path(cache_path)
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True).sort_index()
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False,
                      group_by="column", threads=True)
    if raw.empty:
        raise RuntimeError("No price data were downloaded")
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close = close.dropna(axis=1, how="all").sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True); close.to_csv(cache)
    return close


def prices_to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.resample("ME").last().pct_change(fill_method=None)
    return returns.rename_axis(index="date", columns="symbol").stack(future_stack=True).rename(
        "ret").dropna().reset_index()

