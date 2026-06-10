"""
utils.py — shared data + statistics layer for the SPCX report studies.

Responsibilities
----------------
1. Robust price downloads (yfinance is flaky / rate-limited): retry with
   exponential backoff, cache to local CSV, and log every ticker we drop.
2. Return / market-adjustment math used by Study 1 and Study 2.
3. t-stats (mean / (sd / sqrt(n))) used for significance reporting.

Design notes
------------
* Prices are pulled with ``auto_adjust=True`` so Open/High/Low/Close are all
  split- and dividend-adjusted on the SAME basis. That keeps multi-month return
  paths (e.g. day-1 High -> +180d Close) internally consistent across any split.
* We cache each ticker's FULL history once (period='max') to data/raw/<tkr>.csv,
  then slice the windows we need. This minimises calls to Yahoo and makes runs
  resumable after a rate-limit interruption.
* The earliest cached row is treated as the first-trade date (per the spec).
"""

from __future__ import annotations

import os
import sys
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Windows consoles default to cp1252, which cannot encode characters like
# U+2212 (−) or U+2265 (≥) that appear in our labels. Force UTF-8 so prints
# never crash a run. (Charts/CSVs are already UTF-8.)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
OUT_CHARTS = ROOT / "outputs" / "charts"
OUT_TABLES = ROOT / "outputs" / "tables"
for _d in (RAW_DIR, CLEAN_DIR, OUT_CHARTS, OUT_TABLES):
    _d.mkdir(parents=True, exist_ok=True)

_DROP_LOG = OUT_TABLES / "dropped_tickers.csv"
_FAILED_CACHE = RAW_DIR / "_unavailable_tickers.csv"  # negative cache (truly no data)


# ----------------------------------------------------------------------------
# Drop / survivorship logging  (Methodology rule #4)
# ----------------------------------------------------------------------------
def begin_study(name: str) -> None:
    """Point the survivorship log at a clean per-study file and truncate it."""
    global _DROP_LOG
    _DROP_LOG = OUT_TABLES / f"dropped_{name}.csv"
    if _DROP_LOG.exists():
        _DROP_LOG.unlink()


def log_drop(ticker: str, reason: str) -> None:
    """Append a dropped/missing ticker + reason to the survivorship log."""
    print(f"  [DROP] {ticker}: {reason}")
    header = not _DROP_LOG.exists()
    with open(_DROP_LOG, "a", encoding="utf-8") as fh:
        if header:
            fh.write("ticker,reason\n")
        fh.write(f"{ticker},\"{reason}\"\n")


def reset_drop_log() -> None:
    if _DROP_LOG.exists():
        _DROP_LOG.unlink()


# ---- negative cache: tickers Yahoo returns no data for (delisted/unknown) ----
def _load_unavailable() -> set:
    if _FAILED_CACHE.exists():
        try:
            return set(pd.read_csv(_FAILED_CACHE)["ticker"].astype(str))
        except Exception:
            return set()
    return set()


def _mark_unavailable(ticker: str) -> None:
    header = not _FAILED_CACHE.exists()
    with open(_FAILED_CACHE, "a", encoding="utf-8") as fh:
        if header:
            fh.write("ticker\n")
        fh.write(f"{ticker}\n")


# ----------------------------------------------------------------------------
# Price downloads (cached + retried)
# ----------------------------------------------------------------------------
def _cache_path(ticker: str, raw: bool = False) -> Path:
    safe = ticker.replace("/", "-").replace("^", "_")
    suffix = "__unadj" if raw else ""
    return RAW_DIR / f"{safe}{suffix}.csv"


def fetch_history(
    ticker: str,
    force: bool = False,
    max_retries: int = 5,
    base_pause: float = 1.5,
    raw: bool = False,
) -> pd.DataFrame | None:
    """
    Return the full daily OHLCV history for ``ticker``.

    ``raw=False`` (default): split/dividend-adjusted OHLC (auto_adjust=True) —
    used for multi-month return *paths* so they are split-consistent.

    ``raw=True``: UNADJUSTED OHLC plus an ``AdjClose`` column (auto_adjust=False).
    Needed so the first-day IPO pop can divide the *raw* day-1 close by the *raw*
    offer price (dividing an adjusted close by a raw offer would invent a large
    fake "pop"/drop for dividend payers like GM). Cached separately (``__unadj``).

    On-disk CSV cache; network only on a miss or ``force``. Retries with backoff
    on genuine network errors; empty (delisted) responses fail fast + are
    negative-cached.
    """
    import yfinance as yf  # local import so utils imports even if yf missing

    path = _cache_path(ticker, raw=raw)
    if path.exists() and not force:
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if len(df) > 0:
                return df
        except Exception:
            pass  # corrupt cache -> re-download

    # Negative cache: Yahoo has already told us this symbol has no data. Skip the
    # (pointless, slow) re-attempt unless the caller forces it. This is what keeps
    # the bulk Study-2 run fast despite many delisted/acquired index members.
    if not force and ticker in _load_unavailable():
        return None

    last_err = None
    for attempt in range(max_retries):
        try:
            dl = yf.download(
                ticker,
                period="max",
                interval="1d",
                auto_adjust=not raw,
                progress=False,
                threads=False,
            )
            if dl is not None and len(dl) > 0:
                if isinstance(dl.columns, pd.MultiIndex):
                    dl.columns = dl.columns.get_level_values(0)
                dl = dl.rename(columns={"Adj Close": "AdjClose"})
                want = ["Open", "High", "Low", "Close", "Volume"]
                if raw:
                    want = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
                keep = [c for c in want if c in dl.columns]
                dl = dl[keep].dropna(how="all")
                dl.index.name = "Date"
                dl.to_csv(path)
                return dl
            # Empty, non-error response => symbol genuinely has no data (delisted/
            # unknown). Retrying won't help, so fail FAST and negative-cache it —
            # no exponential backoff (the bug that made bulk runs take ~45s/name).
            _mark_unavailable(ticker)
            log_drop(ticker, "no data (delisted/unknown symbol)")
            return None
        except Exception as e:  # genuine network / parse / rate-limit error
            last_err = repr(e)
            time.sleep(base_pause * (2 ** attempt) + random.uniform(0, 1.0))

    log_drop(ticker, f"download failed after {max_retries} network errors ({last_err})")
    return None


def split_factor_since(ticker: str, since_date, force: bool = False) -> float:
    """
    Product of Yahoo split/spin ratios occurring AFTER ``since_date`` (1.0 if none).

    Yahoo's ``auto_adjust=False`` 'Close' is still SPLIT-adjusted (only dividends
    are left raw), so for a stock that split since its IPO the historical close is
    off by the split factor (e.g. Groupon's 1:20 reverse split scales its 2011
    close x20). Multiplying the cached split-adjusted close by this factor recovers
    the original-dollar day-1 close, so the offer-to-close pop is correct. Cached.
    """
    import yfinance as yf

    path = RAW_DIR / f"{ticker.replace('/', '-').replace('^', '_')}__splits.csv"
    s = None
    if path.exists() and not force:
        try:
            tmp = pd.read_csv(path, index_col=0, parse_dates=True)
            s = tmp.iloc[:, 0] if tmp.shape[1] else pd.Series(dtype=float)
        except Exception:
            s = None
    if s is None:
        try:
            s = yf.Ticker(ticker).splits
            s.to_csv(path)
        except Exception:
            return 1.0
    if s is None or len(s) == 0:
        return 1.0
    idx = pd.to_datetime(s.index)
    try:
        idx = idx.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    vals = np.asarray(s.values)[np.asarray(idx > pd.Timestamp(since_date))]
    factor = float(np.prod(vals)) if len(vals) else 1.0
    return factor if factor > 0 else 1.0


def get_panel(tickers, polite_pause: float = 0.7, force: bool = False) -> dict:
    """
    Fetch many tickers -> {ticker: DataFrame}. Missing names are skipped (logged).
    A small pause between *fresh* downloads keeps us under Yahoo's rate limit.
    """
    out = {}
    for t in tickers:
        cached = _cache_path(t).exists()
        df = fetch_history(t, force=force)
        if df is not None and len(df) > 0:
            out[t] = df
            if not cached:
                time.sleep(polite_pause)  # only throttle on real network hits
    return out


# ----------------------------------------------------------------------------
# Date / window helpers
# ----------------------------------------------------------------------------
def first_trade_date(df: pd.DataFrame) -> pd.Timestamp:
    """Earliest row = first trade date (spec: verify IPO dates from this)."""
    return df.index.min()


def resolve_first_trade(df: pd.DataFrame, expected=None,
                        tolerance_days: int = 7) -> pd.Timestamp:
    """
    The real first *public* trading day, used to anchor the analysis window.

    Defaults to the earliest cached row (the spec's rule). But Yahoo sometimes
    prepends pre-IPO *when-issued / placeholder* rows that sit weeks-to-months
    before the real first trade (e.g. ALLY's flat 2014-01-28 rows before its
    2014-04-10 IPO; PECO 2021-02-25 vs 2021-07-15; VICI 2018-01-02 vs 2018-02-01).
    Those artifacts corrupt day-1 prices, the fade window, AND the split factor
    (a phantom split dated on the true IPO day gets counted as "after" the wrong,
    earlier anchor — ALLY's 310.0). When an audited offer/IPO date is supplied
    (from build_universe) and the earliest row precedes it by more than
    ``tolerance_days``, anchor to the first trading row on/after that audited date
    instead. A normal name (earliest row at/after the offer date, or within the
    tolerance) is unchanged.
    """
    earliest = df.index.min()
    if expected not in (None, "", "nan"):
        exp = pd.Timestamp(expected)
        if pd.notna(exp) and earliest < exp - pd.Timedelta(days=tolerance_days):
            sub = df.loc[exp:]
            if len(sub):
                return sub.index.min()
    return earliest


def window_from(df: pd.DataFrame, start_date, n_trading_days: int) -> pd.DataFrame:
    """Rows from ``start_date`` (inclusive) for the next ``n_trading_days`` rows."""
    sub = df.loc[pd.Timestamp(start_date):]
    return sub.iloc[: n_trading_days + 1]


def value_at_offset(window: pd.DataFrame, col: str, h: int):
    """Value of ``col`` h trading days after the window start (0 = first row)."""
    if h < len(window):
        return float(window[col].iloc[h])
    return np.nan


def align_benchmark(bench: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Benchmark Close aligned to the given trading dates (ffill any gaps)."""
    s = bench["Close"].reindex(dates)
    return s.ffill()


# ----------------------------------------------------------------------------
# Return math
# ----------------------------------------------------------------------------
def cumret(level_now: float, level_base: float) -> float:
    """Simple cumulative return level_now / level_base - 1."""
    if level_base in (0, np.nan) or pd.isna(level_base) or pd.isna(level_now):
        return np.nan
    return level_now / level_base - 1.0


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def market_adjust(stock_ret, bench_ret):
    """Subtract benchmark return from stock return (element-wise or scalar)."""
    return stock_ret - bench_ret


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def tstat(series) -> dict:
    """
    One-sample stats vs 0 using the spec's formula t = mean / (sd / sqrt(n)).
    Returns mean, median, sd, n, se, t, and a two-sided p-value.
    """
    x = pd.Series(series, dtype="float64").dropna()
    n = int(x.shape[0])
    if n == 0:
        return dict(mean=np.nan, median=np.nan, sd=np.nan, n=0, se=np.nan, t=np.nan, p=np.nan)
    mean = float(x.mean())
    median = float(x.median())
    sd = float(x.std(ddof=1)) if n > 1 else np.nan
    se = sd / np.sqrt(n) if (n > 1 and sd > 0) else np.nan
    t = mean / se if (se and not np.isnan(se) and se > 0) else np.nan
    p = float(2 * stats.t.sf(abs(t), df=n - 1)) if (n > 1 and not np.isnan(t)) else np.nan
    return dict(mean=mean, median=median, sd=sd, n=n, se=se, t=t, p=p)


def stderr(series) -> float:
    x = pd.Series(series, dtype="float64").dropna()
    n = x.shape[0]
    if n < 2:
        return np.nan
    return float(x.std(ddof=1) / np.sqrt(n))


def sig_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""
