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
    PRODUCTS, HORIZON_BARS, HORIZON_BARS_1H, HOURLY_SESSION_GAP,
)


# ── OHLC API client (self-contained — import this from your notebook) ─────────

def _date_to_unix(date_string: str) -> int:
    dt = datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


class OHLCFetchError(RuntimeError):
    """Expose HTTP status for callers without exposing authentication headers."""
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def fetch_ohlc_from_db(
    instrs: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """
    Fetch OHLC daily volume data directly from PostgreSQL database (fundamentals_data).
    Used for softs spreads and outrights.
    """
    try:
        import psycopg2
        from .. import config as current_config
        db_cfg = getattr(current_config, "DB_CONFIG", None)
        if not db_cfg:
            return None

        conn = psycopg2.connect(**db_cfg)
        table = (
            "fundamentals_data.spreads_daily_volumes"
            if "-" in instrs
            else "fundamentals_data.outrights_daily_volumes"
        )
        query = f"""
            SELECT trade_date as "Date", open, high, low, close, volume, 
                   buy_volume as buyvolume, sell_volume as sellvolume
            FROM {table}
            WHERE qh_code = %s AND trade_date >= %s AND trade_date <= %s
            ORDER BY trade_date ASC;
        """
        df = pd.read_sql(query, conn, params=(instrs, start_date, end_date))
        conn.close()

        if df.empty:
            return None

        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        for col in ["open", "high", "low", "close", "volume", "buyvolume", "sellvolume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df.sort_values("Date").reset_index(drop=True)
    except Exception as exc:
        print(f"  [DB NOTICE] DB query for {instrs} skipped: {exc}")
        return None


def getData_OHLC_v2(
    instrs: str,
    start_date: str,
    end_date: str,
    timeinterval: str = "1D",
    token: str | None = None,
    raise_on_error: bool = False,
    timeout: float = 60,
) -> pd.DataFrame | None:
    """
    Fetch OHLC bars from the internal QH API.

    Uses API_TOKEN from config.py unless a token is passed explicitly.
    SSL verification is disabled (internal API).

    Returns DataFrame with columns:
      Date, open, high, low, close, volume, buyvolume, sellvolume
    or None on error.
    """
    # Check PostgreSQL DB first (primary for softs spreads and outrights)
    if timeinterval.upper() == "1D":
        db_df = fetch_ohlc_from_db(instrs, start_date, end_date)
        if db_df is not None and not db_df.empty:
            return db_df
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Config can be reloaded/edited in a running notebook. Never keep a stale
    # import-time token; callers may also provide an explicit token override.
    from .. import config as current_config
    _token = current_config.API_TOKEN if token is None else token
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

    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {_token}"}, verify=False, timeout=timeout)
    except Exception as e:
        if raise_on_error:
            raise OHLCFetchError(f"OHLC API request failed for {instrs}: {e}")
        print(f"  [API NETWORK ERROR] Failed to connect for {instrs}: {e}")
        return None

    if r.status_code != 200:
        if raise_on_error:
            raise OHLCFetchError(f"OHLC API HTTP {r.status_code} for {instrs}", r.status_code)
        print(f"  [API ERROR] {r.status_code} for {instrs}: {r.text[:200]}")
        return None

    try:
        resp = r.json()
    except Exception as e:
        if raise_on_error:
            raise OHLCFetchError(f"Non-JSON response from OHLC API: {e}", r.status_code)
        print(f"  [API JSON ERROR] Could not decode JSON for {instrs}: {e} (Status: {r.status_code})")
        return None

    d    = resp.get("data", resp) if isinstance(resp, dict) else resp

    if not d:
        if raise_on_error:
            return pd.DataFrame(columns=["Date", "open", "high", "low", "close", "volume", "buyvolume", "sellvolume"])
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


def _normalize_daily_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate one daily cache/API slice; retain the latest duplicate date."""
    columns = ["Date", "open", "high", "low", "close", "volume", "buyvolume", "sellvolume"]
    data = frame.copy()
    data.columns = ["Date" if str(c).lower() == "date" else str(c).lower() for c in data.columns]
    if not set(columns).issubset(data.columns):
        raise ValueError(f"OHLC missing columns: {set(columns) - set(data.columns)}")
    data = data[columns].copy()
    data["Date"] = pd.to_datetime(data["Date"], utc=True, errors="raise").dt.normalize()
    for name in columns[1:]:
        data[name] = pd.to_numeric(data[name], errors="raise")
    if data.Date.isna().any() or not np.isfinite(data[columns[1:]].to_numpy(dtype=float)).all():
        raise ValueError("OHLC contains missing/nonfinite values")
    invalid = (data.high < data[["open", "close", "low"]].max(axis=1)) | (data.low > data[["open", "close", "high"]].min(axis=1))
    if invalid.any() or (data[["volume", "buyvolume", "sellvolume"]] < 0).any().any():
        raise ValueError("OHLC contains invalid candle/volume values")
    return data.drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)


def fetch_ohlc_1d_incremental(
    instrument: str, product_code: str, start_date, end_date,
    fetcher=None, cache_dir: str | Path | None = None,
    allow_cached_on_fetch_error: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Use cached daily bars; fetch only missing leading/trailing ranges.

    start_date/end_date are inclusive completed-bar bounds. API requests use
    [start, end-exclusive), e.g. Oct6 -> Oct7 fetches the missing Oct6 bar.
    Weekends do not trigger tail fetches; no exchange-holiday/internal-gap
    completeness is claimed. Existing cache history is preserved, and atomic
    replacement occurs only after a successful nonempty missing-range fetch.
    On failure a usable cache may still support analysis, with explicit stale
    coverage metadata. Set allow_cached_on_fetch_error=False for strict runs.
    """
    from .. import config as current_config
    from ..utils.contracts import instrument_to_api_str
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    start, end = start.normalize(), end.normalize()
    target = pd.offsets.BDay().rollback(end)  # known weekends only
    if target < start:
        raise ValueError("No weekday in requested OHLC window")
    base = current_config.DATA_DIR if cache_dir is None else Path(cache_dir)
    cache = Path(base) / product_code.lower() / f"{instrument}_1d.csv"
    columns = ["Date", "open", "high", "low", "close", "volume", "buyvolume", "sellvolume"]
    original = _normalize_daily_data(pd.read_csv(cache)) if cache.exists() else pd.DataFrame(columns=columns)
    usable = original.loc[original.Date.between(start, target)] if not original.empty else original.copy()
    ranges = []
    if usable.empty:
        ranges.append((start, target + pd.Timedelta(days=1)))
    else:
        # Old history before start already covers the left boundary. The first
        # in-range row may be later because of a holiday/no trades, not a missing
        # historical cache prefix.
        first, last = original.Date.min(), usable.Date.max()
        if first > start and len(pd.bdate_range(start, first - pd.Timedelta(days=1))):
            ranges.append((start, first))
        if last < target:
            ranges.append((last + pd.Timedelta(days=1), target + pd.Timedelta(days=1)))
    pieces, statuses, warnings = [original], [], []
    new_dates = set()
    for left, right in ranges:
        print(f"  [FETCH MISSING] {instrument}: {left.date()} -> {right.date()} (end exclusive)", flush=True)
        try:
            if fetcher is None:
                fresh = getData_OHLC_v2(instrs=instrument_to_api_str(instrument),
                    start_date=left.strftime("%Y-%m-%d"), end_date=right.strftime("%Y-%m-%d"),
                    timeinterval="1D", token=current_config.API_TOKEN, raise_on_error=True)
            else:
                fresh = fetcher(instrs=instrument_to_api_str(instrument), start_date=left.strftime("%Y-%m-%d"),
                                end_date=right.strftime("%Y-%m-%d"), timeinterval="1D")
            if fresh is None:
                raise OHLCFetchError("OHLC API returned no usable response")
            if fresh.empty:
                statuses.append("empty_response")
                warnings.append(f"API returned no bars for {left.date()} -> {right.date()}")
                continue
            fresh = _normalize_daily_data(fresh)
            fresh = fresh.loc[(fresh.Date >= left) & (fresh.Date < right)]
            if fresh.empty:
                statuses.append("empty_response")
                warnings.append("API returned no completed bars inside requested missing range")
                continue
            new_dates.update(set(fresh.Date) - set(original.Date))
            pieces.append(fresh)
            statuses.append("fetched")
        except (ValueError, OSError, RuntimeError) as exc:
            if not allow_cached_on_fetch_error or usable.empty:
                raise
            statuses.append("fetch_failed_using_cache")
            warnings.append(str(exc))
            print(f"  [CACHE WARNING] {instrument}: {exc}; retaining saved bars", flush=True)
    merged = _normalize_daily_data(pd.concat(pieces, ignore_index=True)) if len(pieces) > 1 else original
    if len(pieces) > 1:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".csv.tmp")
        merged.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(cache)
    data = merged.loc[merged.Date.between(start, target)].reset_index(drop=True)
    if data.empty:
        raise ValueError(f"{instrument}: no cached/fetched bars inside requested range")
    reached = data.Date.max() >= target
    if not reached:
        warnings.append(f"Wanted completed data through {target.date()}; available through {data.Date.max().date()}")
    if not ranges:
        print(f"  [CACHE OK] {instrument}: latest completed bar {data.Date.max().date()}; no API call", flush=True)
    return data, {"cache_path": str(cache), "requested_start": start, "requested_end": end,
                  "target_weekday_end": target, "actual_start": data.Date.min(), "actual_end": data.Date.max(),
                  "target_date_reached": bool(reached), "api_requests": len(ranges), "new_dates": len(new_dates),
                  "fetch_status": "; ".join(statuses) if statuses else "cache_only",
                  "cache_fallback_used": "fetch_failed_using_cache" in statuses,
                  "coverage_warning": "; ".join(warnings),
                  "coverage_basis": "cached bars plus missing-range API updates; internal gaps/exchange holidays unverified"}


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

    print(f"  [FETCH] {api_str}  {start_date} -> {end_date}")
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
    exceeds `session_gap` (default 4 hours — catches Fri-close->Mon-open gaps).
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

    print(f"  [FETCH 1H] {api_str}  {start_date} -> {end_date}")
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
