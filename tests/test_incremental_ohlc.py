"""Offline verification of cache reuse, request boundaries and failure preservation."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from market_analysis import config
from market_analysis.data.ohlc import fetch_ohlc_1d_incremental, getData_OHLC_v2, OHLCFetchError


def bars(dates, close=2.):
    return pd.DataFrame({"Date": dates, "open": close, "high": close + 1,
                         "low": close - 1, "close": close, "volume": 100,
                         "buyvolume": 60, "sellvolume": 40})


class IncrementalTests(unittest.TestCase):
    def cache(self, directory, dates):
        path = Path(directory) / "bo/BOF27-H27_1d.csv"
        path.parent.mkdir(parents=True)
        bars(dates).to_csv(path, index=False)
        return path

    def test_complete_cache_does_not_call_api_or_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.cache(directory, ["2026-10-02", "2026-10-05", "2026-10-06"])
            original = path.read_bytes()
            fetcher = Mock(side_effect=AssertionError("must not fetch"))
            data, coverage = fetch_ohlc_1d_incremental("BOF27-H27", "BO", "2026-10-02", "2026-10-06", fetcher, directory)
            fetcher.assert_not_called()
            self.assertTrue(coverage["target_date_reached"])
            self.assertEqual(coverage["api_requests"], 0)
            self.assertEqual(path.read_bytes(), original)

    def test_only_missing_oct6_is_requested_and_today_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.cache(directory, ["2026-09-30", "2026-10-02", "2026-10-05"])
            fetcher = Mock(return_value=bars(["2026-10-05", "2026-10-06", "2026-10-07"], close=3.))
            data, coverage = fetch_ohlc_1d_incremental("BOF27-H27", "BO", "2026-10-02", "2026-10-06", fetcher, directory)
            self.assertEqual(fetcher.call_args.kwargs["start_date"], "2026-10-06")
            self.assertEqual(fetcher.call_args.kwargs["end_date"], "2026-10-07")
            self.assertEqual(len(data), 3)
            self.assertEqual(data.Date.max(), pd.Timestamp("2026-10-06", tz="UTC"))
            saved = pd.read_csv(path)
            self.assertEqual(len(saved), 4)  # preserve Sep30 history
            self.assertEqual(float(data.loc[data.Date == pd.Timestamp("2026-10-05", tz="UTC"), "close"].iloc[0]), 2.)
            self.assertEqual(coverage["new_dates"], 1)

    def test_missing_history_and_tail_are_separate_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            self.cache(directory, ["2026-10-05"])
            fetcher = Mock(side_effect=[bars(["2026-10-02"]), bars(["2026-10-06"])])
            data, _ = fetch_ohlc_1d_incremental("BOF27-H27", "BO", "2026-10-02", "2026-10-06", fetcher, directory)
            bounds = [(c.kwargs["start_date"], c.kwargs["end_date"]) for c in fetcher.call_args_list]
            self.assertEqual(bounds, [("2026-10-02", "2026-10-05"), ("2026-10-06", "2026-10-07")])
            self.assertEqual(len(data), 3)

    def test_401_retains_cache_and_warns_target_not_reached(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.cache(directory, ["2026-10-02", "2026-10-05"])
            original = path.read_bytes()
            fetcher = Mock(side_effect=OHLCFetchError("OHLC API HTTP 401", 401))
            data, coverage = fetch_ohlc_1d_incremental("BOF27-H27", "BO", "2026-10-02", "2026-10-06", fetcher, directory)
            self.assertEqual(len(data), 2)
            self.assertTrue(coverage["cache_fallback_used"])
            self.assertFalse(coverage["target_date_reached"])
            self.assertIn("available through 2026-10-05", coverage["coverage_warning"])
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(OHLCFetchError):
                fetch_ohlc_1d_incremental("BOF27-H27", "BO", "2026-10-02", "2026-10-06", fetcher, directory,
                                          allow_cached_on_fetch_error=False)

    def test_weekend_does_not_trigger_an_api_request(self):
        with tempfile.TemporaryDirectory() as directory:
            self.cache(directory, ["2026-10-02"])
            fetcher = Mock(side_effect=AssertionError("weekend must not fetch"))
            _, coverage = fetch_ohlc_1d_incremental("BOF27-H27", "BO", "2026-10-02", "2026-10-04", fetcher, directory)
            fetcher.assert_not_called()
            self.assertTrue(coverage["target_date_reached"])

    def test_no_cache_fetches_full_requested_window(self):
        with tempfile.TemporaryDirectory() as directory:
            fetcher = Mock(return_value=bars(["2026-10-02", "2026-10-05", "2026-10-06"]))
            data, coverage = fetch_ohlc_1d_incremental("BOF27-H27", "BO", "2026-10-02", "2026-10-06", fetcher, directory)
            self.assertEqual(len(data), 3)
            self.assertEqual(fetcher.call_args.kwargs["start_date"], "2026-10-02")
            self.assertTrue(Path(coverage["cache_path"]).exists())

    def test_current_config_token_used_after_module_import(self):
        response = Mock(status_code=200)
        response.json.return_value = {"data": bars(["2026-10-06"]).to_dict("records")}
        transport = Mock()
        transport.get.return_value = response
        with patch.object(config, "API_TOKEN", "fresh-test-token"), patch.dict(sys.modules, {"requests": transport, "urllib3": Mock()}):
            getData_OHLC_v2("BOF27-H27", "2026-10-06", "2026-10-07")
        self.assertEqual(transport.get.call_args.kwargs["headers"]["Authorization"], "Bearer fresh-test-token")
        self.assertEqual(transport.get.call_args.kwargs["timeout"], 60)


if __name__ == "__main__":
    unittest.main()
