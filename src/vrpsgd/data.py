"""Market data loading: Yahoo Finance prices -> log-return matrix, with disk caching."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path.home() / ".cache" / "vrpsgd"


def _cache_key(tickers: list[str], start: str, end: str) -> str:
    raw = "|".join(sorted(tickers)) + f"|{start}|{end}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def fetch_prices(tickers: list[str], start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Download adjusted close prices for ``tickers`` between ``start`` and ``end``.

    Cached to ~/.cache/vrpsgd so repeated API calls / tests don't hammer Yahoo Finance.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{_cache_key(tickers, start, end)}.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.dropna(how="all")

    if use_cache:
        prices.to_parquet(cache_path)
    return prices


def prices_to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a price DataFrame to daily log returns, dropping incomplete rows."""
    returns = np.log(prices / prices.shift(1)).dropna(how="any")
    return returns


def load_return_matrix(
    tickers: list[str], start: str, end: str, use_cache: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Convenience wrapper: tickers/date range -> (T x n float64 return matrix, column names)."""
    prices = fetch_prices(tickers, start, end, use_cache=use_cache)
    returns = prices_to_log_returns(prices)
    return returns.values.astype(np.float64), list(returns.columns)
