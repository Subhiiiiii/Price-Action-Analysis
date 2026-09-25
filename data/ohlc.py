# market_analysis/data/ohlc.py
"""
OHLC data fetching and caching.

Fetches 1D OHLC bars from the internal API, caches to CSV under DATA_DIR.
Multi-day (2d/3d/5d) bars are built via trailing-window aggregation from 1D bars.
If CLEAN_PREVIOUS_DATA is True the cache is wiped before fetching.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    DATA_DIR, CLEAN_PREVIOUS_DATA, OHLC_START_DATE, OHLC_END_DATE,
    PRODUCTS, HORIZON_BARS, HORIZON_BARS_1H, HOURLY_SESSION_GAP, API_TOKEN,
)


# ── OHLC API client (self-contained — import this from your notebook) ─────────

def _date_to_unix(date_string: str) -> int:
    dt = datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def getData_OHLC_v2(
    instrs: str,
    start_date: str,
    end_date: str,
    timeinterval: str = "1D",
    token: str | None = None,
) -> pd.DataFrame | None:
    """
    Fetch OHLC bars from the internal QH API.

    Uses API_TOKEN from config.py unless a token is passed explicitly.
    SSL verification is disabled (internal API).

    Returns DataFrame with columns:
      Date, open, high, low, close, volume, buyvolume, sellvolume
    or None on error.
    """
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    _token = token or API_TOKEN
    if not _token or _token == "PASTE_YOUR_TOKEN_HERE":
        raise ValueError(
            "API token not set. Edit API_TOKEN in market_analysis/config.py"
        )

    start_unix = _date_to_unix(start_date)
    end_unix   = _date_to_unix(end_date)

    url = (
        f"https://qh-api.corp.hertshtengroup.com/apis/ohlc/"
        f"?instruments={instrs}"
        f"&interval={timeinterval}"
        f"&end={end_unix}"
        f"&start={start_unix}"
        f"&extraFields=buyvolume%2Csellvolume"
    )

    r = requests.get(url, headers={"Authorization": f"Bearer {_token}"}, verify=False)

    if r.status_code != 200:
        print(f"  [API ERROR] {r.status_code} for {instrs}: {r.text[:200]}")
        return None

    resp = r.json()
    d    = resp.get("data", resp) if isinstance(resp, dict) else resp

    if not d:
        print(f"  [WARN] Empty API response for {instrs}")
        return None

    df = pd.DataFrame(d)

    if "time" in df.columns:
        df["Date"] = (
            pd.to_datetime(df["time"], unit="ms", utc=True)
            .dt.strftime("%Y-%m-%d")
        )

    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})

    return df.sort_values("Date").reset_index(drop=True)


# ── internal helpers ──────────────────────────────────────────────────────────

def _cache_path(product_code: str, instrument: str) -> Path:
    """Path for the cached 1D CSV: DATA_DIR/sm/SMZ26-SMF27_1d.csv"""
    folder = DATA_DIR / product_code.lower()
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{instrument}_1d.csv"


def _clean_product_cache(product_code: str) -> None:
    folder = DATA_DIR / product_code.lower()
    if folder.exists():
        shutil.rmtree(folder)
        print(f"  [CLEAN] Removed cache: {folder}")


# ── rolling aggregation (pure pandas, no Bar objects needed) ─────────────────

def _aggregate_trailing(df1d: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Build trailing n-day OHLC bars from a sorted 1D DataFrame.

    Each output row is labelled by its LAST source row (row i covers i-n+1..i).
    open  = first bar's open
    high  = max of highs
    low   = min of lows
    close = last bar's close
    volume / buyvolume / sellvolume = sum
    """
    df = df1d.copy().sort_values("Date").reset_index(drop=True)
    rows = []
    for i in range(n - 1, len(df)):
        window = df.iloc[i - n + 1 : i + 1]
        row = {
            "Date"       : df.iloc[i]["Date"],
            "open"       : float(window["open"].iloc[0]),
            "high"       : float(window["high"].max()),
            "low"        : float(window["low"].min()),
            "close"      : float(window["close"].iloc[-1]),
            "volume"     : float(window["volume"].sum()),
            "buyvolume"  : float(window["buyvolume"].sum()),
            "sellvolume" : float(window["sellvolume"].sum()),
        }
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


# ── public API ────────────────────────────────────────────────────────────────

def fetch_ohlc_1d(
    instrument: str,
    product_code: str,
    getData_OHLC_v2,          # the function from the notebook
    start_date: str = OHLC_START_DATE,
    end_date:   str = OHLC_END_DATE,
) -> pd.DataFrame | None:
    """
    Fetch 1D OHLC bars for one instrument.
    Returns a DataFrame with lowercase columns:
      Date, open, high, low, close, volume, buyvolume, sellvolume

    Uses cache unless CLEAN_PREVIOUS_DATA is True.
    """
    if CLEAN_PREVIOUS_DATA:
        _clean_product_cache(product_code)

    cache = _cache_path(product_code, instrument)

    if cache.exists():
        df = pd.read_csv(cache)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        return df.sort_values("Date").reset_index(drop=True)

    # convert internal instrument "SMZ26-SMF27" to API form "SMZ26-F27"
    from ..utils.contracts import instrument_to_api_str
    api_str = instrument_to_api_str(instrument)

    print(f"  [FETCH] {api_str}  {start_date} → {end_date}")
    raw = getData_OHLC_v2(
        instrs     = api_str,
        start_date = start_date,
        end_date   = end_date,
    )

    if raw is None or len(raw) == 0:
        print(f"  [WARN] No OHLC data returned for {api_str}")
        return None

    # normalise columns to lowercase
    raw.columns = [c.lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    raw = raw.rename(columns={"date": "Date"})

    needed = ["Date", "open", "high", "low", "close", "volume", "buyvolume", "sellvolume"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        print(f"  [WARN] Missing columns {missing} for {api_str}")
        return None

    df = raw[needed].dropna(subset=["Date","close"]).sort_values("Date").reset_index(drop=True)

    # cache to disk
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def get_ohlc_for_horizon(
    instrument: str,
    product_code: str,
    horizon: str,
    getData_OHLC_v2,
    start_date: str = OHLC_START_DATE,
    end_date:   str = OHLC_END_DATE,
) -> pd.DataFrame | None:
    """
    Return OHLC bars for a given horizon ("1d"/"2d"/"3d"/"5d").
    1D bars are fetched/cached; multi-day bars are built by trailing aggregation.
    """
    df1d = fetch_ohlc_1d(instrument, product_code, getData_OHLC_v2,
                          start_date, end_date)
    if df1d is None:
        return None

    n = HORIZON_BARS[horizon]
    if n == 1:
        return df1d

    return _aggregate_trailing(df1d, n)


def get_all_horizon_ohlc(
    instrument: str,
    product_code: str,
    horizons: list[str],
    getData_OHLC_v2,
    start_date: str = OHLC_START_DATE,
    end_date:   str = OHLC_END_DATE,
) -> dict[str, pd.DataFrame]:
    """
    Fetch once, return a dict {horizon: ohlc_df} for all requested horizons.
    """
    df1d = fetch_ohlc_1d(instrument, product_code, getData_OHLC_v2,
                          start_date, end_date)
    if df1d is None:
        return {}

    result = {}
    for h in horizons:
        n = HORIZON_BARS[h]
        result[h] = df1d if n == 1 else _aggregate_trailing(df1d, n)

    return result


# ── 1H helpers (added alongside 1D — do NOT modify functions above) ───────────

import datetime as _dt   # noqa: E402  (local alias to avoid shadowing top-level import)


def _cache_path_1h(product_code: str, instrument: str) -> Path:
    """Path for the cached 1H CSV: DATA_DIR/sm/SMZ26-SMF27_1h.csv"""
    folder = DATA_DIR / product_code.lower()
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{instrument}_1h.csv"


def _aggregate_trailing_1h(
    df1h: pd.DataFrame,
    n: int,
    session_gap: str = HOURLY_SESSION_GAP,
) -> pd.DataFrame:
    """
    Build trailing n-bar OHLC from sorted 1H bars, respecting session breaks.

    A session break is detected when the time gap between consecutive bars
    exceeds `session_gap` (default 4 hours — catches Fri-close→Mon-open gaps).
    When a window of n bars would span a session break the row is omitted
    (set open/high/low/close to NaN) rather than blending overnight/weekend
    prices.  This avoids spurious signals at session opens.

    Columns expected: Date (datetime or str), open, high, low, close,
                       volume, buyvolume, sellvolume.
    """
    df = df1h.copy()
    # ensure datetime index for gap detection
    df["_dt"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.sort_values("_dt").reset_index(drop=True)

    gap_thresh = pd.Timedelta(session_gap)

    # pre-compute which bar indices have a session break BEFORE them
    breaks: set[int] = set()
    for i in range(1, len(df)):
        if df["_dt"].iloc[i] - df["_dt"].iloc[i - 1] > gap_thresh:
            breaks.add(i)

    rows = []
    for i in range(n - 1, len(df)):
        # check if any bar in window [i-n+1 .. i] starts a session
        window_indices = range(i - n + 1, i + 1)
        crosses_gap = any(j in breaks for j in window_indices)

        window = df.iloc[i - n + 1 : i + 1]
        row = {
            "Date"       : df["_dt"].iloc[i],  # datetime for downstream use
            "open"       : float(window["open"].iloc[0])     if not crosses_gap else np.nan,
            "high"       : float(window["high"].max())        if not crosses_gap else np.nan,
            "low"        : float(window["low"].min())         if not crosses_gap else np.nan,
            "close"      : float(window["close"].iloc[-1])   if not crosses_gap else np.nan,
            "volume"     : float(window["volume"].sum())     if not crosses_gap else np.nan,
            "buyvolume"  : float(window["buyvolume"].sum())  if not crosses_gap else np.nan,
            "sellvolume" : float(window["sellvolume"].sum()) if not crosses_gap else np.nan,
            "session_break": int(crosses_gap),
        }
        rows.append(row)

    result = pd.DataFrame(rows).reset_index(drop=True)
    # drop rows that cross a session boundary (NaN close = unusable bar)
    result = result.dropna(subset=["close"]).reset_index(drop=True)
    return result


def fetch_ohlc_1h(
    instrument: str,
    product_code: str,
    getData_OHLC_v2,
    start_date: str | None = None,
    end_date: str   | None = None,
) -> pd.DataFrame | None:
    """
    Fetch 1H OHLC bars for one instrument.

    Identical to fetch_ohlc_1d but requests interval="1H" from the API.
    Default fetch window: last 6 months (≈2 000 bars at 16 bars/day).
    Cache key: DATA_DIR/<product>/<instrument>_1h.csv.

    Returns a DataFrame with columns:
      Date (datetime UTC), open, high, low, close, volume, buyvolume, sellvolume
    or None on error.
    """
    if CLEAN_PREVIOUS_DATA:
        _clean_product_cache(product_code)

    # default: last 6 months
    if end_date is None:
        end_date = _dt.date.today().strftime("%Y-%m-%d")
    if start_date is None:
        six_months_ago = _dt.date.today() - _dt.timedelta(days=183)
        start_date = six_months_ago.strftime("%Y-%m-%d")

    cache = _cache_path_1h(product_code, instrument)

    if cache.exists():
        df = pd.read_csv(cache)
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        return df.sort_values("Date").reset_index(drop=True)

    from ..utils.contracts import instrument_to_api_str
    api_str = instrument_to_api_str(instrument)

    print(f"  [FETCH 1H] {api_str}  {start_date} → {end_date}")
    raw = getData_OHLC_v2(
        instrs      = api_str,
        start_date  = start_date,
        end_date    = end_date,
        timeinterval= "1H",       # <— only difference from 1D
    )

    if raw is None or len(raw) == 0:
        print(f"  [WARN] No 1H OHLC data returned for {api_str}")
        return None

    raw.columns = [c.lower() for c in raw.columns]

    # For 1H the API returns a unix-ms timestamp in "time"; convert to UTC datetime
    if "time" in raw.columns:
        raw["Date"] = pd.to_datetime(raw["time"], unit="ms", utc=True)
    elif "date" in raw.columns:
        raw["Date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    else:
        print(f"  [WARN] No time/date column in 1H response for {api_str}")
        return None

    needed = ["Date", "open", "high", "low", "close", "volume", "buyvolume", "sellvolume"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        print(f"  [WARN] Missing 1H columns {missing} for {api_str}")
        return None

    df = (raw[needed]
          .dropna(subset=["Date", "close"])
          .sort_values("Date")
          .reset_index(drop=True))

    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def get_ohlc_1h_for_horizon(
    instrument: str,
    product_code: str,
    horizon: str,
    getData_OHLC_v2,
    start_date: str | None = None,
    end_date:   str | None = None,
) -> pd.DataFrame | None:
    """
    Return 1H OHLC bars for a given 1H horizon ("1h"/"2h"/"4h"/"8h"/"16h").
    Fetches/caches 1H base bars; multi-bar horizons use _aggregate_trailing_1h.
    """
    df1h = fetch_ohlc_1h(instrument, product_code, getData_OHLC_v2,
                          start_date, end_date)
    if df1h is None:
        return None

    n = HORIZON_BARS_1H[horizon]
    if n == 1:
        return df1h

    return _aggregate_trailing_1h(df1h, n)


def get_all_horizon_ohlc_1h(
    instrument: str,
    product_code: str,
    horizons: list[str],
    getData_OHLC_v2,
    start_date: str | None = None,
    end_date:   str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch 1H base bars once, return {horizon: ohlc_df} for all requested 1H horizons.
    """
    df1h = fetch_ohlc_1h(instrument, product_code, getData_OHLC_v2,
                          start_date, end_date)
    if df1h is None:
        return {}

    result = {}
    for h in horizons:
        n = HORIZON_BARS_1H[h]
        result[h] = df1h if n == 1 else _aggregate_trailing_1h(df1h, n)

    return result