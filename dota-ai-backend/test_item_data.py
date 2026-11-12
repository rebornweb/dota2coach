import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import item_data
import main
from fastapi.testclient import TestClient
from ollama import ChatResponse
from ml.train_item_popularity import train, training_rows


def counts():
    return {stage: {"1": 3, "2": 1} for stage in item_data.STAGES}


class ItemTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        patcher = patch.object(item_data, "ROOT", Path(self.folder.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        item_data._retry.clear()

    def test_cache_refresh_stale_fallback_and_retry_suppression(self):
        response = Mock()
        response.json.return_value = counts()
        with patch.object(item_data.httpx, "get", return_value=response) as get:
            first = item_data.cached_fetch("1", "https://example.test", item_data.validate_popularity)
            self.assertFalse(first[2])
            item_data.cached_fetch("1", "https://example.test", item_data.validate_popularity)
            self.assertEqual(get.call_count, 1)
        path = item_data.ROOT / "1.json"
        payload = json.loads(path.read_text())
        payload["fetched_at"] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        path.write_text(json.dumps(payload))
        with patch.object(item_data.httpx, "get", side_effect=httpx.ConnectError("offline")) as get:
            self.assertTrue(item_data.cached_fetch("1", "url", item_data.validate_popularity)[2])
            self.assertTrue(item_data.cached_fetch("1", "url", item_data.validate_popularity)[2])
            self.assertEqual(get.call_count, 1)

    def test_invalid_payload_and_no_cache_fail_clearly(self):
        for invalid in ({}, {**counts(), "mid_game_items": {"1": -1}},
                        {**counts(), "early_game_items": {"1": True}}):
            with self.assertRaises(ValueError):
                item_data.validate_popularity(invalid)
        with patch.object(item_data.httpx, "get", side_effect=httpx.ConnectError("offline")):
            with self.assertRaises(RuntimeError):
                item_data.get_item_popularity(1)

    def test_names_counts_and_shares(self):
        item_data.validate_items({"recipe": {"id": 907}})
        with patch.object(item_data, "cached_fetch", side_effect=[
            (counts(), "now", False), ({"blink": {"id": 1, "dname": "Blink Dagger"}}, "then", False)
        ]):
            data = item_data.get_item_popularity(1)
        rows = data["stages"]["mid_game_items"]
        self.assertEqual(rows[0]["name"], "Blink Dagger")
        self.assertEqual(rows[0]["purchase_share_pct"], 75)
        self.assertEqual(rows[1]["name"], "Item #2")

    def test_selected_hero_controls_endpoint_and_chat_after_history(self):
        api = TestClient(main.app)
        result = {"hero_id": 1, "stages": {stage: [] for stage in item_data.STAGES}}
        with patch.object(main, "get_heroes", return_value=[{"id": 1, "localized_name": "Anti-Mage"}]), \
             patch.object(main, "get_item_popularity", return_value=result) as fetch, \
             patch.object(main.client, "chat", return_value=ChatResponse(message={"role": "assistant", "content": "Popular items"})) as llm:
            self.assertEqual(api.get("/item-popularity", params={"hero": "Anti-Mage"}).status_code, 200)
            response = api.post("/chat", json={"hero": "Anti-Mage", "prompt": "What items should I buy?",
                "history": [{"role": "assistant", "content": "You are Dazzle"}]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(fetch.call_args.args, (1,))
            self.assertIn('"hero_id": 1', llm.call_args.kwargs["messages"][-2]["content"])
            self.assertEqual(api.get("/item-popularity", params={"hero": "Unknown"}).status_code, 404)

    def test_training_smoke_and_grouped_evaluation(self):
        snapshots = [{"hero_id": hero, "stages": {stage: [
            {"item_id": 1, "count": hero}, {"item_id": 2, "count": 2}
        ] for stage in item_data.STAGES}} for hero in range(1, 7)]
        frame = training_rows(snapshots)
        self.assertAlmostEqual(frame.iloc[0]["share"], 100 / 3)
        output = Path(self.folder.name) / "model.joblib"
        train(snapshots, output)
        self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
