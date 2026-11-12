import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from ollama import ChatResponse
import main
import scenario_data as data
from ml.train_item_timings import prepare, train


def row(hero=50, item="spirit_vessel", time=900, games="100", wins="60"):
    return dict(hero_id=hero, item=item, time=time, games=games, wins=wins)


class ScenarioTests(unittest.TestCase):
    def test_counts_units_and_zero_games(self):
        with patch.object(data, "cached_fetch", return_value=([row(), row(games="0", wins="0")], "stamp", True)):
            result = data.get_scenario("itemTimings", 50)
        self.assertTrue(result["stale"])
        self.assertEqual(result["rows"][0]["win_rate_pct"], 60)
        self.assertEqual(result["rows"][0]["time_minutes"], 15)
        self.assertIsNone(result["rows"][1]["win_rate_pct"])

    def test_invalid_data_rejected(self):
        for rows in ([row(wins="101")], [row(games=True)], [row(hero=1)], [row(time=-1)]):
            with self.assertRaises(ValueError):
                data.validate_rows(rows, "itemTimings", 50)

    def test_partial_outage(self):
        with patch.object(data, "get_scenario", side_effect=[RuntimeError("offline"), {"rows": []}, {"rows": []}]):
            result = data.get_scenarios(50)
        self.assertIn("error", result["itemTimings"])
        self.assertEqual(result["laneRoles"]["rows"], [])

    def test_selection_to_endpoint_and_ollama_overrides_history(self):
        api = TestClient(main.app)
        snapshot = {"itemTimings": {"rows": [row(hero=1)], "stale": False}}
        with patch.object(main, "get_heroes", return_value=[{"id": 1, "localized_name": "Anti-Mage"}]), \
             patch.object(main, "get_scenarios", return_value=snapshot) as fetch, \
             patch.object(main, "get_item_popularity", side_effect=RuntimeError("offline")), \
             patch.object(main.client, "chat", return_value=ChatResponse(message={"role": "assistant", "content": "Advice"})) as llm:
            self.assertEqual(api.get("/scenarios", params={"hero": "Anti-Mage"}).json()["hero_id"], 1)
            self.assertEqual(api.get("/scenarios", params={"hero": "unknown"}).status_code, 404)
            response = api.post("/chat", json={"hero": "Anti-Mage", "prompt": "Item timing advice?",
                "history": [{"role": "assistant", "content": "You selected Dazzle"}]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(fetch.call_args.args, (1, ["itemTimings"]))
            self.assertIn('"hero_id": 1', llm.call_args.kwargs["messages"][-2]["content"])

    def test_training_and_duplicate_guard(self):
        rows = [row(hero=hero, time=seconds, wins=str(40 + hero))
                for hero in range(1, 9) for seconds in (600, 900, 1200)]
        with self.assertRaises(ValueError):
            prepare([row(), row()], 100)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "model.joblib"
            metrics = train(rows, output)
            self.assertTrue(output.exists())
            self.assertEqual(metrics["training_pairs"] + metrics["test_pairs"], 8)


if __name__ == "__main__":
    unittest.main()
