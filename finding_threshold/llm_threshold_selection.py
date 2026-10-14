"""Compare Groq GPT-OSS and Qwen threshold choices on identical instrument tables.

Fill config.GROQ_API_KEYS, then run::

    run_llm_threshold_selection("SM", candle_lengths=[1, 2, 3, 5])
    run_llm_threshold_selection("BO", candle_lengths=[1, 2, 3, 5])

One table per instrument/group is sent independently to each model. Python
calculates every metric. Models return only an existing candidate ID, assessment,
and short explanation; Python retrieves the exact thresholds/numbers itself.
Two dated CSVs are saved incrementally. Normal runs reuse successful rows across
dates; call update_llm_threshold_selection explicitly to refresh all contracts.
These are candidate choices for the user's own backtest, not proven thresholds.
No API calls happen on import. Uses stdlib HTTP, so the groq SDK is not required.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .. import config
from .expected_profit_analysis import run_expected_profit_analysis

_STATUSES = ("promising", "weak", "not_working", "insufficient_data")


def _json_safe(value):
    """Convert pandas/numpy values to portable JSON; missing metrics become null."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    return value


def _json_text(value) -> str:
    """Stable serialization also makes experiment hashes reproducible."""
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def _selection_settings(overrides: Mapping | None = None) -> dict:
    """Merge nonsecret model/request settings without altering config defaults."""
    settings = {**config.GROQ_THRESHOLD_SELECTION, **(overrides or {})}
    if not settings["models"] or any(not str(m).strip() for m in settings["models"].values()):
        raise ValueError("At least one named model ID is required")
    for alias in settings["models"]:
        if not str(alias).replace("_", "").replace("-", "").isalnum():
            raise ValueError("Model aliases must be safe filename components")
    for key in ("max_completion_tokens", "request_timeout_seconds", "max_attempts"):
        if isinstance(settings[key], bool) or int(settings[key]) != settings[key] or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("min_seconds_between_calls", "retry_base_seconds", "target_signals_per_month_low"):
        if not np.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if settings["target_signals_per_month_high"] < settings["target_signals_per_month_low"]:
        raise ValueError("The upper monthly frequency preference must be >= the lower one")
    return settings


def build_instrument_candidate_tables(result: dict) -> dict[str, pd.DataFrame]:
    """Add monthly availability, maximum success size, and frequency to each grid.

    Exposure is calendar-month equivalents of actual scorable date coverage:
    June-September each count 1, a partial October counts only its day fraction.
    Count all selected signals for frequency, including unresolved/censored ones;
    expected size still uses only resolved successes/failures. Retention is relative
    to this instrument's maximum signal count WITHIN the tested grid, not a claimed
    universal baseline. Monthly counts show concentration rather than just averages.
    """
    tables = {}
    for group in result["threshold_table"]["analysis_group"].drop_duplicates():
        table = result["threshold_table"].loc[result["threshold_table"]["analysis_group"] == group].copy()
        table = table.sort_values(["delta_z_threshold", "price_z_threshold"]).reset_index(drop=True)
        features = result["features"]
        if group != "combined":
            features = features.loc[features["candle_length"] == int(group[:-1])]
        valid = features.loc[np.isfinite(features[["z_delta", "z_price", "atr"]]).all(axis=1)
                             & (features["atr"] > 0)]
        if valid.empty:
            start = end = None
            periods, fractions = [], {}
        else:
            start = valid["Date"].min().tz_localize(None).normalize()
            end = features["Date"].max().tz_localize(None).normalize()
            periods = list(pd.period_range(start, end, freq="M"))
            fractions = {}
            for month in periods:
                left = max(start, month.start_time.normalize())
                right = min(end, month.end_time.normalize())
                fractions[str(month)] = (right - left).days / month.days_in_month + 1 / month.days_in_month
        exposure = sum(fractions.values())
        max_count = int(table["n_signals"].max())
        extra = []
        for i, row in table.iterrows():
            key = (group, float(row["delta_z_threshold"]), float(row["price_z_threshold"]))
            events = result["event_ledgers"][key]
            monthly = events["Date"].dt.tz_localize(None).dt.to_period("M").value_counts()
            counts = {str(month): int(monthly.get(month, 0)) for month in periods}
            sizes = events.loc[events["outcome"] == "SUCCESS", "success_size_ticks"]
            extra.append({
                "candidate_id": f"C{i+1:04d}",
                "max_profit_ticks": float(sizes.max()) if not sizes.empty else np.nan,
                "signals_per_month": int(row["n_signals"]) / exposure if exposure else np.nan,
                "resolved_signals_per_month": int(row["n_resolved"]) / exposure if exposure else np.nan,
                "exposure_months": exposure, "coverage_start": str(start.date()) if start is not None else None,
                "coverage_end": str(end.date()) if end is not None else None,
                "scorable_signal_dates": int(valid["Date"].nunique()),
                "max_signals_in_tested_grid": max_count,
                "signal_retention_fraction": int(row["n_signals"]) / max_count if max_count else np.nan,
                "active_months": sum(n > 0 for n in counts.values()),
                "monthly_signal_counts": _json_text(counts),
                "monthly_exposure_fractions": _json_text(fractions),
            })
        tables[str(group)] = pd.concat([table, pd.DataFrame(extra)], axis=1)
    return tables


def build_selection_prompt(table: pd.DataFrame, context: dict, settings: dict) -> str:
    """Explain absorption, frequency preferences and row-selection constraints.

    No hidden numerical objective forces excessive rarity. The LLM considers the
    entire dynamic grid and returns a short decision summary, not chain-of-thought.
    """
    columns = ["candidate_id", "delta_z_threshold", "price_z_threshold", "max_profit_ticks",
               "expected_profit_ticks", "n_signals", "n_resolved", "n_success", "n_failure",
               "n_unresolved", "n_censored", "avg_success_ticks", "avg_failure_ticks",
               "success_rate", "signals_per_month", "resolved_fraction", "signal_retention_fraction",
               "active_months", "monthly_signal_counts", "n_stop_bar_order_unknown"]
    # Send column names once, rather than repeating twenty names for every row.
    # Six decimals is presentation precision only; CSV/Python retain exact values.
    compact = table[columns].copy()
    for column in compact.select_dtypes(include="number"):
        compact[column] = compact[column].round(6)
    candidates = {"columns": columns, "rows": compact.values.tolist()}
    return f"""Select one absorption-v2 threshold candidate for THIS instrument only.
Only the provided computed evidence may be used; do not invent market facts or numbers.

MEANING:
BUY absorption = positive aggressive delta with small price response; expected DOWN, short.
SELL absorption = negative aggressive delta with small price response; expected UP, long.
The signed delta distribution is NOT absolute. One positive X serves both tails:
BUY requires raw delta>0 and z_delta>=X; SELL requires raw delta<0 and z_delta<=-X.
Price displacement is absolute BEFORE scoring; its z-score is signed. Both sides require z_price<=Y.
Signals are detected on aggregated windows, outcomes on raw future 1D bars.
Required profit before first stop means SUCCESS; later stop cannot undo success.
First stop before required profit means FAILURE, sized at configured signal-time ATR distance.
Success size is maximum favorable movement BEFORE first stop or horizon end; no post-stop profit.
Expected ticks = average signed success size / failure loss across RESOLVED events.
UNRESOLVED has insufficient move without stop; CENSORED lacks complete size evidence.
Expected ticks and maximum profit are hindsight opportunity metrics, NOT proven realized P&L.
Combined windows use retrospective longest-first overlap suppression within this instrument.

SELECTION PREFERENCES:
Aim roughly {settings['target_signals_per_month_low']} to {settings['target_signals_per_month_high']}
signals per month or more when supported; this is a SOFT preference, not a hard band or upper cap.
Naturally sparse contracts may have fewer signals. Look at max_signals_in_tested_grid,
retention, coverage and monthly counts before blaming a threshold for low frequency.
For active contracts avoid unnecessarily suppressing signals just for higher per-event mean.
Profit scales differ between instruments. Do not impose one absolute profit floor on all contracts.
Front rank is context, NOT evidence that a particular contract must have many signals.
Compare nearby X/Y rows for consistency, sufficient resolved evidence and frequency/profit balance.
Higher |z_delta| or more negative z_price is NOT automatically better; profit can rise or fall.
Do not select based only on one maximum gain, a single rare event, or a high success rate.
No unseen-data backtest is provided; never claim validation, guaranteed profit or optimum.

OUTPUT:
Select exactly one candidate_id present in this table, even if evidence is weak.
assessment must be promising, weak, not_working or insufficient_data.
If every setting has poor payoffs/no coherent pattern, still report the least-bad supported
candidate and mark not_working. If evidence is too sparse/missing, mark insufficient_data.
Give a concise reason explaining the frequency/profit tradeoff and any weakness, at most 120 words.
Do not recompute thresholds; return JSON matching the supplied schema only.

CONTEXT: {_json_text(context)}
CANDIDATE TABLE (rows follow columns in the same order): {_json_text(candidates)}
"""


def selection_response_schema(candidate_ids: Sequence[str]) -> dict:
    """Strict JSON response: enum IDs prevent invented threshold combinations."""
    return {"type": "json_schema", "json_schema": {
        "name": "absorption_threshold_choice", "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "properties": {"candidate_id": {"type": "string", "enum": list(candidate_ids)},
                                  "assessment": {"type": "string", "enum": list(_STATUSES)},
                                  "reason": {"type": "string"}},
                   "required": ["candidate_id", "assessment", "reason"]}}}


def validate_selection(reply: dict, table: pd.DataFrame) -> pd.Series:
    """Validate semantic evidence as well as JSON shape; retrieve authoritative row."""
    if not isinstance(reply, dict) or set(reply) != {"candidate_id", "assessment", "reason"}:
        raise ValueError("Model output must contain exactly candidate_id, assessment and reason")
    if reply["assessment"] not in _STATUSES or not isinstance(reply["reason"], str) or not reply["reason"].strip():
        raise ValueError("Invalid assessment or empty reason")
    selected = table.loc[table["candidate_id"] == reply["candidate_id"]]
    if len(selected) != 1:
        raise ValueError("Model selected an unknown/nonunique candidate ID")
    row = selected.iloc[0]
    if reply["assessment"] == "promising" and (row["n_resolved"] == 0 or
        not np.isfinite(row["expected_profit_ticks"]) or row["expected_profit_ticks"] <= 0):
        raise ValueError("Promising assessment contradicts missing/nonpositive expected payoff")
    return row


class GroqThresholdClient:
    """Stdlib Groq client with shared pacing and bounded retries, no key logging.

    Use alternate keys only if credentials are rejected, not to evade shared quota.
    429 honors Retry-After; transient failures back off. Both model calls share
    this client's schedule. GET /models verifies actual availability before POST.
    """
    def __init__(self, api_keys: Sequence[str], settings: dict):
        self.keys = list(dict.fromkeys(k.strip() for k in api_keys if isinstance(k, str) and k.strip()))
        if not self.keys:
            raise ValueError("Fill config.GROQ_API_KEYS before running")
        self.settings, self.key_index, self.last_call = settings, 0, 0.0

    def _request(self, endpoint: str, payload: dict | None = None) -> dict:
        """Send one verified HTTPS request; mask secrets from any reported error."""
        last_error = "Request failed"
        for attempt in range(self.settings["max_attempts"]):
            wait = self.settings["min_seconds_between_calls"] - (time.monotonic() - self.last_call)
            if wait > 0:
                time.sleep(wait)
            body = _json_text(payload).encode("utf-8") if payload is not None else None
            request = Request("https://api.groq.com/openai/v1/" + endpoint, data=body,
                              headers={"Authorization": "Bearer " + self.keys[self.key_index],
                                       "Content-Type": "application/json", "User-Agent": "absorption-threshold-review"})
            self.last_call = time.monotonic()
            try:
                with urlopen(request, timeout=self.settings["request_timeout_seconds"]) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                last_error = f"Groq HTTP {exc.code} on {endpoint}"
                # Keep only the provider's diagnostic message, with keys redacted.
                # This exposes useful token-limit details without logging headers/body.
                try:
                    diagnostic = json.loads(exc.read()).get("error", {}).get("message", "")
                    if isinstance(diagnostic, str):
                        for key in self.keys:
                            diagnostic = diagnostic.replace(key, "[REDACTED]")
                        diagnostic = re.sub(r"gsk_[A-Za-z0-9_-]+", "[REDACTED]", diagnostic)
                        last_error += ": " + diagnostic[:1200].replace("\n", " ")
                except (ValueError, AttributeError, TypeError):
                    pass
                if exc.code in (401, 403) and self.key_index + 1 < len(self.keys):
                    self.key_index += 1
                    continue
                if exc.code != 429 and exc.code < 500:
                    raise RuntimeError(last_error) from None
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = float(retry_after) if retry_after else 0.0
                except ValueError:
                    try:
                        delay = max(0.0, parsedate_to_datetime(retry_after).timestamp() - time.time())
                    except (ValueError, TypeError):
                        delay = 0.0
            except (URLError, TimeoutError):
                last_error, delay = "Groq network request failed", 0.0
            if attempt + 1 < self.settings["max_attempts"]:
                time.sleep(max(delay, self.settings["retry_base_seconds"] * 2 ** attempt))
        raise RuntimeError(last_error + "; retries exhausted")

    def available_models(self) -> set[str]:
        """Return model IDs actually listed by Groq for the active credential."""
        return {item["id"] for item in self._request("models").get("data", [])}

    def _choose_single(self, model: str, prompt: str, candidate_ids: Sequence[str]) -> tuple[dict, dict]:
        """Request strict JSON and a short decision summary with reasoning enabled."""
        response = self._request("chat/completions", {
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "response_format": selection_response_schema(candidate_ids),
            "reasoning_effort": self.settings["reasoning_effort"], "include_reasoning": False,
            "max_completion_tokens": self.settings["max_completion_tokens"],
            "temperature": self.settings["temperature"],
        })
        choices = response.get("choices", [])
        if not choices or choices[0].get("finish_reason") != "stop":
            raise ValueError("Model response incomplete/refused; increase token budget if truncated")
        content = choices[0].get("message", {}).get("content")
        try:
            parsed = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            raise ValueError("Model did not return parseable JSON") from None
        return parsed, {"usage": response.get("usage"), "response_id": response.get("id")}

    def choose(self, model: str, prompt: str, candidate_ids: Sequence[str]) -> tuple[dict, dict]:
        """Review every candidate in bounded batches, then compare batch winners.

        This is a tournament selection, not a single full-grid decision. No rows
        are filtered by a Python profit objective. On HTTP 413, split the batch
        further; repeated smaller calls cannot fix a single-row/account limit.
        All stages share normal pacing/retries and are recorded in the audit.
        """
        marker = "CANDIDATE TABLE (rows follow columns in the same order): "
        if marker not in prompt:
            return self._choose_single(model, prompt, candidate_ids)
        header, encoded = prompt.rsplit(marker, 1)
        data = json.loads(encoded)
        columns = data["columns"]
        id_index = columns.index("candidate_id")
        rows = {row[id_index]: row for row in data["rows"]}
        limit = self.settings.get("max_candidates_per_call", 40)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 2:
            raise ValueError("max_candidates_per_call must be an integer >= 2")
        stages = []

        def review(ids):
            if len(ids) > limit:
                winners = [review(ids[i:i + limit])["candidate_id"] for i in range(0, len(ids), limit)]
                return review(winners)
            stage_prompt = header + (
                "\nFor large grids this is one tournament stage. Select from the supplied rows only; "
                "later stages compare batch winners. Do not claim every grid row is shown here.\n"
            ) + marker + _json_text({"columns": columns, "rows": [rows[cid] for cid in ids]})
            try:
                reply, audit = self._choose_single(model, stage_prompt, ids)
            except RuntimeError as exc:
                if "Groq HTTP 413" not in str(exc):
                    raise
                if len(ids) <= 1:
                    raise RuntimeError(str(exc) + "; even one candidate is too large: check account token limit and completion budget") from None
                middle = (len(ids) + 1) // 2
                left, right = review(ids[:middle]), review(ids[middle:])
                finalists = [left["candidate_id"], right["candidate_id"]]
                if len(ids) == 2:
                    raise RuntimeError(str(exc) + "; account cannot fit two candidates plus instructions: reduce completion budget or increase account token limit") from None
                return review(finalists)
            # Intermediate choices must also be valid before entering the next stage.
            if not isinstance(reply, dict) or set(reply) != {"candidate_id", "assessment", "reason"}:
                raise ValueError("Invalid JSON fields in batch choice")
            if reply["candidate_id"] not in ids or reply["assessment"] not in _STATUSES or not isinstance(reply["reason"], str) or not reply["reason"].strip():
                raise ValueError("Invalid candidate/assessment/reason in batch choice")
            stages.append({"candidate_ids": ids, "reply": reply, **audit})
            return reply

        reply = review(list(candidate_ids))
        return reply, {"selection_method": "batched_tournament" if len(stages) > 1 else "single_table",
                       "stages": stages}


def _save_csv(rows: list[dict], path: Path) -> None:
    """Incremental atomic replacement avoids a partially written final CSV."""
    temporary = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _saved_successes(folder: Path, product: str, alias: str, model: str, date: str) -> dict:
    """Find the latest successful choice per instrument/group, across dated CSVs.

    A completed weak/not_working assessment is still a completed analysis.
    Failed rows and files dated after the requested run date cannot satisfy resume.
    """
    saved = {}
    for path in sorted(folder.glob(f"{product}_????-??-??_{alias}_thresholds.csv")):
        file_date = path.name[len(product) + 1:len(product) + 11]
        if file_date > date:
            continue
        for row in pd.read_csv(path, keep_default_na=False).to_dict("records"):
            if row.get("model") != model or row.get("product") != product or row.get("selection_status") != "ok":
                continue
            try:
                valid = all(np.isfinite(float(row[name])) for name in ("z_volume_delta", "z_price_response"))
            except (KeyError, TypeError, ValueError):
                valid = False
            if valid:
                row["source_csv"] = str(path)
                saved[(row["instrument"], row["analysis_group"])] = row
    return saved


def run_llm_threshold_selection(
    product_code: str, candle_lengths: Sequence[int] | None = None, mode: str = "combined",
    params: Mapping | None = None, delta_z_grid: Sequence[float] | None = None,
    price_z_grid: Sequence[float] | None = None, front_n: int | None = None,
    llm_params: Mapping | None = None, as_of=None, output_dir: str | Path | None = None,
    resume: bool = True, analysis_result: dict | None = None, client=None,
) -> dict:
    """Compute once; obtain TWO independently selected, dated instrument tables.

    `analysis_result` can reuse an existing expected-profit run without refetching.
    `client` permits offline testing; normal usage reads config.GROQ_API_KEYS.
    Normal resume reuses saved successes across dates before data fetch/analysis.
    Parameter/data changes do not trigger refresh: use update_llm_threshold_selection.
    Errors get explicit rows with blank thresholds, never fake LLM selections.
    All successful instrument tables are retained as CSVs and each model's parsed
    output/provenance is appended to an audit JSONL. No automatic backtest.
    """
    settings = _selection_settings(llm_params)
    if product_code not in config.PRODUCTS:
        raise ValueError(f"Unknown product {product_code!r}")
    if analysis_result is not None and analysis_result.get("product_code") != product_code:
        raise ValueError("analysis_result must belong to the requested product")
    from .expected_profit_analysis import _settings, _lengths, analysis_date_range
    from ..utils.contracts import get_all_front_instruments
    cfg = _settings(params)
    lengths = _lengths(cfg["candle_lengths"] if candle_lengths is None else candle_lengths)
    if mode not in ("combined", "separate"):
        raise ValueError("mode must be 'combined' or 'separate'")
    count = config.FRONT_N if front_n is None else front_n
    if isinstance(count, bool) or int(count) != count or count < 1:
        raise ValueError("front_n must be a positive integer")
    timezone = cfg["timezone"]
    date = pd.Timestamp.now(tz=timezone) if as_of is None else pd.Timestamp(as_of)
    if date.tzinfo is not None:
        date = date.tz_convert(timezone)
    date_string = date.strftime("%Y-%m-%d")
    folder = Path(config.OUTPUT_DIR / "llm_thresholds" if output_dir is None else output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    evidence_dir = folder / f"{product_code}_{date_string}_evidence"
    evidence_dir.mkdir(exist_ok=True)
    groups = ["combined"] if mode == "combined" else [f"{k}d" for k in lengths]
    start, end = analysis_date_range(as_of, cfg)
    universe = (analysis_result["universe"] if analysis_result is not None else
                get_all_front_instruments(product_code, (end + pd.Timedelta(days=1)).date(), int(count)))
    history = {alias: _saved_successes(folder, product_code, alias, model, date_string) if resume else {}
               for alias, model in settings["models"].items()}
    pending = {inst for members in universe.values() for inst in members
               if any((inst, group) not in history[alias] for alias in history for group in groups)}
    if analysis_result is None:
        if pending:
            analysis_result = run_expected_profit_analysis(
                product_code, lengths, mode, cfg, delta_z_grid, price_z_grid,
                as_of=as_of, front_n=int(count), instrument_names=sorted(pending))
        else:
            # All completed: no market API, no computation, no Groq credentials needed.
            analysis_result = {"product_code": product_code, "mode": mode, "params": cfg,
                               "candle_lengths": lengths, "universe": universe,
                               "instruments": {}, "errors": {}, "analysis_start": start, "analysis_end": end}
    available = None  # Check Groq only when a model actually needs a new selection.
    all_tables, jobs = {}, []
    for structure, instruments in analysis_result["universe"].items():
        for rank, instrument in enumerate(instruments, start=1):
            result = analysis_result["instruments"].get(instrument)
            if result is not None and instrument in pending:
                all_tables[instrument] = build_instrument_candidate_tables(result)
            for group in groups:
                table = all_tables.get(instrument, {}).get(group)
                if table is not None:
                    table.to_csv(evidence_dir / f"{instrument}_{group}_candidates.csv", index=False, encoding="utf-8-sig")
                jobs.append((instrument, structure, rank, group, table))
    saved_tables, paths = {}, {}
    for alias, model in settings["models"].items():
        path = folder / f"{product_code}_{date_string}_{alias}_thresholds.csv"
        # Preserve existing rows during a refresh too, until each job is replaced.
        previous = pd.read_csv(path, keep_default_na=False).to_dict("records") if path.exists() else []
        rows = []
        active_jobs = {(inst, group or analysis_result["mode"]) for inst, _, _, group, _ in jobs}

        def checkpoint():
            """Keep unprocessed resume rows until their replacement is ready."""
            seen = {(row["instrument"], row["analysis_group"]) for row in rows}
            remaining = [row for row in previous if row.get("model") == model and
                         (row.get("instrument"), row.get("analysis_group")) in active_jobs - seen]
            _save_csv(rows + remaining, path)

        for instrument, structure, rank, group, table in jobs:
            old = history[alias].get((instrument, group))
            if old is not None:
                reused = {**old, "front_rank": rank, "reused_saved_threshold": True}
                rows.append(reused)
                checkpoint()
                print(f"{alias} | {instrument} | {group}: saved threshold reused ({old['run_date']})")
                continue
            if table is not None and available is None:
                if client is None:
                    client = GroqThresholdClient(config.GROQ_API_KEYS, settings)
                available = client.available_models()
            base = {"run_date": date_string, "product": product_code, "instrument": instrument,
                    "structure": structure, "front_rank": rank,
                    "analysis_group": group or analysis_result["mode"], "model": model,
                    "selection_status": "error", "assessment": None, "not_working": None,
                    "z_volume_delta": np.nan, "z_price_response": np.nan,
                    "max_profit_ticks": np.nan, "expected_profit_ticks": np.nan,
                    "reason": "", "candidate_id": None, "prompt_hash": None}
            base["reused_saved_threshold"] = False
            if table is None:
                base.update(selection_status="analysis_error", reason=analysis_result["errors"].get(instrument, "Analysis unavailable"))
            elif model not in available:
                base.update(selection_status="model_unavailable", reason="Model not present in Groq /models response")
            else:
                result = analysis_result["instruments"][instrument]
                context = {"instrument": instrument, "product": product_code, "structure": structure,
                           "front_rank_within_structure": rank, "analysis_group": group,
                           "candle_lengths": result["metadata"]["candle_lengths"],
                           "analysis_params": result["metadata"]["params"],
                           "coverage": result.get("coverage", {}),
                           "frequency_baseline": table.iloc[0][[
                               "exposure_months", "coverage_start", "coverage_end",
                               "max_signals_in_tested_grid", "scorable_signal_dates"]].to_dict()}
                prompt = build_selection_prompt(table, context, settings)
                digest = hashlib.sha256(_json_text({"prompt": prompt, "model": model,
                                                   "settings": settings}).encode("utf-8")).hexdigest()
                base["prompt_hash"] = digest
                try:
                    reply, audit = client.choose(model, prompt, table["candidate_id"].tolist())
                    selected = validate_selection(reply, table)
                    base.update(selected.to_dict())
                    base["selection_method"] = audit.get("selection_method", "single_table")
                    base.update(z_volume_delta=float(selected["delta_z_threshold"]),
                                z_price_response=float(selected["price_z_threshold"]),
                                selection_status="ok", assessment=reply["assessment"],
                                not_working=reply["assessment"] == "not_working", reason=reply["reason"].strip())
                    with (folder / f"{product_code}_{date_string}_{alias}_audit.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(_json_text({"instrument": instrument, "group": group, "model": model,
                                                 "prompt_hash": digest, "reply": reply, **audit}) + "\n")
                except (ValueError, RuntimeError) as exc:
                    base.update(selection_status="selection_error", reason=str(exc))
            rows.append(base)
            checkpoint()
            detail = "" if base["selection_status"] == "ok" else " | " + base["reason"]
            print(f"{alias} | {instrument} | {group}: {base['selection_status']}{detail}")
        saved_tables[alias], paths[alias] = pd.DataFrame(rows), str(path)
    return {"tables": saved_tables, "csv_paths": paths, "candidate_tables": all_tables,
            "analysis": analysis_result, "evidence_dir": str(evidence_dir)}


def update_llm_threshold_selection(
    product_code: str, candle_lengths: Sequence[int] | None = None, mode: str = "combined",
    params: Mapping | None = None, delta_z_grid: Sequence[float] | None = None,
    price_z_grid: Sequence[float] | None = None, front_n: int | None = None,
    llm_params: Mapping | None = None, as_of=None, output_dir: str | Path | None = None,
    client=None,
) -> dict:
    """Refresh ALL current front-N spreads/jumbos and both model selections.

    Deliberately bypass saved successes and fetch fresh data. Incremental CSV
    saving remains enabled, so completed instruments survive an interruption.
    """
    return run_llm_threshold_selection(
        product_code, candle_lengths, mode, params, delta_z_grid, price_z_grid,
        front_n=front_n, llm_params=llm_params, as_of=as_of, output_dir=output_dir,
        resume=False, client=client)
