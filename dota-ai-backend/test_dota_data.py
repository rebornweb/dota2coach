import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import dota_data


class HeroCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "cache.json"
        for name, value in [("CACHE", self.path), ("_memory_cache", None), ("_retry_after", 0)]:
            patcher = patch.object(dota_data, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.heroes = [{"localized_name": "Dazzle"}]

    def cache(self, age=0):
        self.path.write_text(json.dumps({
            "heroes": self.heroes,
            "fetched_at": (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat(),
        }), encoding="utf-8")

    def test_fresh_cache_skips_network(self):
        self.cache()
        with patch.object(dota_data.httpx, "get") as get:
            self.assertEqual(dota_data.get_heroes(), self.heroes)
            get.assert_not_called()

    def test_expired_cache_refreshes_and_persists(self):
        self.cache(age=7200)
        updated = [{"localized_name": "Anti-Mage"}]
        with patch.object(dota_data.httpx, "get", return_value=Mock(json=lambda: updated)) as get:
            self.assertEqual(dota_data.get_heroes(), updated)
            self.assertEqual(dota_data.get_heroes(), updated)
            get.assert_called_once()
        self.assertEqual(json.loads(self.path.read_text())["heroes"], updated)

    def test_timeout_uses_stale_cache_and_delays_retry(self):
        self.cache(age=7200)
        with patch.object(dota_data.httpx, "get", side_effect=httpx.ReadTimeout("timeout")) as get:
            self.assertEqual(dota_data.get_heroes(), self.heroes)
            self.assertEqual(dota_data.get_heroes(), self.heroes)
            get.assert_called_once()

    def test_corrupt_cache_and_outage_give_clear_error(self):
        self.path.write_text("broken JSON")
        with patch.object(dota_data.httpx, "get", side_effect=httpx.ConnectError("offline")) as get:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "no valid local cache"):
                    dota_data.get_heroes()
            get.assert_called_once()

    def test_invalid_api_data_does_not_replace_stale_cache(self):
        self.cache(age=7200)
        with patch.object(dota_data.httpx, "get", return_value=Mock(json=lambda: [None])):
            self.assertEqual(dota_data.get_heroes(), self.heroes)

    def test_disk_failure_keeps_successful_response_in_memory(self):
        with patch.object(dota_data.httpx, "get", return_value=Mock(json=lambda: self.heroes)) as get, \
                patch.object(dota_data, "_write_cache", side_effect=OSError("disk full")):
            self.assertEqual(dota_data.get_heroes(), self.heroes)
            self.assertEqual(dota_data.get_heroes(), self.heroes)
            get.assert_called_once()


if __name__ == "__main__":
    unittest.main()
