"""Offline tests: synthetic evidence and mock Groq; no credentials/API use."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from market_analysis.finding_threshold.expected_profit_analysis import analyze_expected_profit
from market_analysis.finding_threshold.llm_threshold_selection import (
    GroqThresholdClient, _selection_settings, build_instrument_candidate_tables,
    build_selection_prompt, run_llm_threshold_selection, selection_response_schema,
    validate_selection, update_llm_threshold_selection,
)


def synthetic_analysis():
    """Two contracts share a deterministic grid but retain separate provenance."""
    rng = np.random.default_rng(41)
    size = 90
    close = 100 + rng.normal(0, 0.3, size).cumsum()
    open_ = np.r_[close[0], close[:-1]]
    delta = rng.uniform(-40, 40, size)
    raw = pd.DataFrame({"Date": pd.bdate_range("2026-06-01", periods=size),
                        "open": open_, "close": close,
                        "high": np.maximum(close, open_) + 1,
                        "low": np.minimum(close, open_) - 1,
                        "volume": 100, "buyvolume": (100 + delta)/2,
                        "sellvolume": (100 - delta)/2})
    params = {"baseline_window": 4, "baseline_min_periods": 4, "volume_window": 2,
              "atr_window": 2, "atr_period": 2, "forward_days": 3,
              "normalize_features": False}
    results = {}
    universe = {"spread": ["SMZ26-F27"], "jumbo": ["SMZ26-H27"]}
    for structure, instruments in universe.items():
        for instrument in instruments:
            result = analyze_expected_profit(raw, 0.1, [1, 3], params=params,
                delta_z_grid=[0.5, 1.0], price_z_grid=[0.0, -0.5])
            result["metadata"].update(product="SM", structure=structure, instrument=instrument)
            results[instrument] = result
    return {"product_code": "SM", "mode": "combined", "universe": universe,
            "instruments": results, "errors": {}, "params": result["metadata"]["params"]}


class FakeClient:
    """Return valid/invalid selections predictably, counting calls for resume checks."""
    def __init__(self, invalid=False):
        self.calls, self.invalid = [], invalid

    def available_models(self):
        return {"openai/gpt-oss-120b", "qwen/qwen3.8-27b"}

    def choose(self, model, prompt, candidate_ids):
        self.calls.append((model, prompt))
        return ({"candidate_id": "unknown" if self.invalid else candidate_ids[0],
                 "assessment": "weak", "reason": "Frequency is retained, but evidence needs backtesting."},
                {"usage": {"total_tokens": 123}, "response_id": "offline"})


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.run = synthetic_analysis()
        self.result = self.run["instruments"]["SMZ26-F27"]
        self.table = build_instrument_candidate_tables(self.result)["combined"]

    def test_candidate_counts_and_maximum_come_from_exact_events(self):
        for _, row in self.table.iterrows():
            events = self.result["event_ledgers"][("combined", row.delta_z_threshold, row.price_z_threshold)]
            successes = events.loc[events.outcome == "SUCCESS", "success_size_ticks"]
            self.assertEqual(row.n_signals, len(events))
            self.assertEqual(sum(json.loads(row.monthly_signal_counts).values()), len(events))
            self.assertAlmostEqual(row.signals_per_month, len(events) / row.exposure_months)
            if len(successes):
                self.assertEqual(row.max_profit_ticks, successes.max())

    def test_partial_month_and_missing_metrics_serialize_as_valid_json(self):
        prompt = build_selection_prompt(self.table, {"unknown": np.nan}, _selection_settings())
        self.assertIn('"unknown": null', prompt)
        self.assertNotIn(": NaN", prompt)
        self.assertIn("SOFT preference", prompt)
        self.assertIn("NOT automatically better", prompt)
        fractions = json.loads(self.table.iloc[0].monthly_exposure_fractions)
        self.assertTrue(any(0 < f < 1 for f in fractions.values()))

    def test_invalid_id_and_contradictory_promising_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_selection({"candidate_id": "bad", "assessment": "weak", "reason": "x"}, self.table)
        changed = self.table.copy()
        changed.loc[0, "expected_profit_ticks"] = -1.0
        with self.assertRaises(ValueError):
            validate_selection({"candidate_id": changed.iloc[0].candidate_id,
                                "assessment": "promising", "reason": "x"}, changed)

    def test_schema_enumerates_only_available_ids(self):
        schema = selection_response_schema(self.table.candidate_id.tolist())["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        self.assertEqual(schema["schema"]["properties"]["candidate_id"]["enum"], self.table.candidate_id.tolist())


class RunnerTests(unittest.TestCase):
    def test_two_csvs_resume_even_when_settings_change(self):
        run, client = synthetic_analysis(), FakeClient()
        with tempfile.TemporaryDirectory() as temporary:
            args = dict(analysis_result=run, client=client, as_of="2026-10-06", output_dir=temporary)
            result = run_llm_threshold_selection("SM", **args)
            self.assertEqual(set(result["tables"]), {"gpt_oss", "qwen"})
            self.assertEqual(len(client.calls), 4)
            self.assertEqual(client.calls[0][1], client.calls[2][1])
            for alias, path in result["csv_paths"].items():
                self.assertIn(f"SM_2026-10-06_{alias}", path)
                table = pd.read_csv(path)
                self.assertEqual(len(table), 2)
                self.assertEqual(set(table.selection_status), {"ok"})
                self.assertTrue(table.z_volume_delta.notna().all())
            run_llm_threshold_selection("SM", **args)
            self.assertEqual(len(client.calls), 4)  # resume does not charge again
            run_llm_threshold_selection("SM", llm_params={"temperature": 0.3}, **args)
            self.assertEqual(len(client.calls), 4)  # only explicit update refreshes successes

    def test_next_day_completed_contracts_need_no_data_or_groq(self):
        run, client = synthetic_analysis(), FakeClient()
        with tempfile.TemporaryDirectory() as temporary:
            run_llm_threshold_selection("SM", analysis_result=run, client=client,
                                        as_of="2026-10-06", output_dir=temporary)
            with patch("market_analysis.utils.contracts.get_all_front_instruments", return_value=run["universe"]), \
                 patch("market_analysis.finding_threshold.llm_threshold_selection.run_expected_profit_analysis") as analyze, \
                 patch("market_analysis.finding_threshold.llm_threshold_selection.GroqThresholdClient") as groq:
                result = run_llm_threshold_selection("SM", as_of="2026-10-07", output_dir=temporary)
            analyze.assert_not_called()
            groq.assert_not_called()
            self.assertEqual(result["analysis"]["instruments"], {})
            for table in result["tables"].values():
                self.assertTrue(table.reused_saved_threshold.all())
                self.assertEqual(set(table.run_date), {"2026-10-06"})
                self.assertEqual(len(table), 2)

    def test_only_missing_contract_and_model_are_recomputed(self):
        run, client = synthetic_analysis(), FakeClient()
        with tempfile.TemporaryDirectory() as temporary:
            result = run_llm_threshold_selection("SM", analysis_result=run, client=client,
                                                as_of="2026-10-06", output_dir=temporary)
            path = Path(result["csv_paths"]["qwen"])
            table = pd.read_csv(path)
            table.loc[table.instrument != "SMZ26-H27"].to_csv(path, index=False)
            with patch("market_analysis.utils.contracts.get_all_front_instruments", return_value=run["universe"]), \
                 patch("market_analysis.finding_threshold.llm_threshold_selection.run_expected_profit_analysis", return_value=run) as analyze:
                run_llm_threshold_selection("SM", client=client, as_of="2026-10-07", output_dir=temporary)
            self.assertEqual(analyze.call_args.kwargs["instrument_names"], ["SMZ26-H27"])
            self.assertEqual(len(client.calls), 5)
            self.assertEqual(client.calls[-1][0], "qwen/qwen3.8-27b")

    def test_explicit_update_refreshes_entire_front_universe(self):
        run, client = synthetic_analysis(), FakeClient()
        with tempfile.TemporaryDirectory() as temporary:
            run_llm_threshold_selection("SM", analysis_result=run, client=client,
                                        as_of="2026-10-06", output_dir=temporary)
            with patch("market_analysis.utils.contracts.get_all_front_instruments", return_value=run["universe"]), \
                 patch("market_analysis.finding_threshold.llm_threshold_selection.run_expected_profit_analysis", return_value=run) as analyze:
                result = update_llm_threshold_selection("SM", client=client, as_of="2026-10-06", output_dir=temporary)
            self.assertEqual(set(analyze.call_args.kwargs["instrument_names"]), set(run["instruments"]))
            self.assertEqual(len(client.calls), 8)
            for table in result["tables"].values():
                self.assertFalse(table.reused_saved_threshold.any())

    def test_model_errors_never_produce_fake_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_llm_threshold_selection("SM", analysis_result=synthetic_analysis(),
                client=FakeClient(invalid=True), as_of="2026-10-06", output_dir=temporary)
            for table in result["tables"].values():
                self.assertEqual(set(table.selection_status), {"selection_error"})
                self.assertTrue(table.z_volume_delta.isna().all())

    def test_analysis_failure_still_has_row_for_each_model(self):
        run = synthetic_analysis()
        del run["instruments"]["SMZ26-H27"]
        run["errors"]["SMZ26-H27"] = "Missing coverage"
        with tempfile.TemporaryDirectory() as temporary:
            result = run_llm_threshold_selection("SM", analysis_result=run, client=FakeClient(),
                as_of="2026-10-06", output_dir=temporary)
            for table in result["tables"].values():
                failed = table.loc[table.instrument == "SMZ26-H27"].iloc[0]
                self.assertEqual(failed.selection_status, "analysis_error")
                self.assertTrue(np.isnan(failed.z_volume_delta))

    def test_other_product_analysis_cannot_be_mislabelled(self):
        with self.assertRaises(ValueError):
            run_llm_threshold_selection("BO", analysis_result=synthetic_analysis(), client=FakeClient())


class ClientTests(unittest.TestCase):
    def test_batches_cover_every_row_and_compare_winners(self):
        run = synthetic_analysis()
        table = build_instrument_candidate_tables(run["instruments"]["SMZ26-F27"])["combined"]
        settings = _selection_settings({"max_candidates_per_call": 2})
        client = GroqThresholdClient(["fake-key"], settings)
        calls = []

        def choose(model, prompt, ids):
            calls.append(list(ids))
            return {"candidate_id": ids[0], "assessment": "weak", "reason": "test"}, {}

        prompt = build_selection_prompt(table, {}, settings)
        with patch.object(client, "_choose_single", side_effect=choose):
            reply, audit = client.choose("test", prompt, table.candidate_id.tolist())
        self.assertEqual(calls[:2], [table.candidate_id.tolist()[:2], table.candidate_id.tolist()[2:]])
        self.assertEqual(calls[-1], [calls[0][0], calls[1][0]])
        self.assertEqual(audit["selection_method"], "batched_tournament")
        self.assertEqual(reply["candidate_id"], calls[0][0])

    def test_413_splits_instead_of_repeating_oversized_request(self):
        run = synthetic_analysis()
        table = build_instrument_candidate_tables(run["instruments"]["SMZ26-F27"])["combined"]
        settings = _selection_settings({"max_candidates_per_call": 10})
        client = GroqThresholdClient(["fake-key"], settings)
        calls = []

        def choose(model, prompt, ids):
            calls.append(len(ids))
            if len(ids) > 2:
                raise RuntimeError("Groq HTTP 413 on chat/completions")
            return {"candidate_id": ids[0], "assessment": "weak", "reason": "test"}, {}

        with patch.object(client, "_choose_single", side_effect=choose):
            client.choose("test", build_selection_prompt(table, {}, settings), table.candidate_id.tolist())
        self.assertEqual(calls, [4, 2, 2, 2])

    def test_single_candidate_413_is_actionable_and_finite(self):
        run = synthetic_analysis()
        table = build_instrument_candidate_tables(run["instruments"]["SMZ26-F27"])["combined"].iloc[:1]
        settings = _selection_settings()
        client = GroqThresholdClient(["fake-key"], settings)
        with patch.object(client, "_choose_single", side_effect=RuntimeError("Groq HTTP 413")) as request:
            with self.assertRaisesRegex(RuntimeError, "even one candidate"):
                client.choose("test", build_selection_prompt(table, {}, settings), table.candidate_id.tolist())
        self.assertEqual(request.call_count, 1)

    def test_payload_is_strict_json_without_raw_reasoning(self):
        client = GroqThresholdClient(["fake-key"], _selection_settings())
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "candidate_id": "C1", "assessment": "weak", "reason": "test"})}}]}
        with patch.object(client, "_request", return_value=response) as request:
            reply, _ = client.choose("qwen/qwen3.8-27b", "prompt", ["C1"])
            body = request.call_args.args[1]
            self.assertFalse(body["include_reasoning"])
            self.assertTrue(body["response_format"]["json_schema"]["strict"])
            self.assertEqual(reply["candidate_id"], "C1")

    def test_truncated_output_rejected(self):
        client = GroqThresholdClient(["fake-key"], _selection_settings())
        with patch.object(client, "_request", return_value={"choices": [{"finish_reason": "length"}]}):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                client.choose("openai/gpt-oss-120b", "prompt", ["C1"])

    def test_empty_keys_rejected_before_analysis(self):
        with self.assertRaises(ValueError):
            GroqThresholdClient(["", "  "], _selection_settings())


if __name__ == "__main__":
    unittest.main()
