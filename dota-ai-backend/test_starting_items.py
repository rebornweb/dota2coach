import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import main


class StartingItemTests(unittest.TestCase):
    def test_start_games_is_grounded_in_current_hero_not_bad_history(self):
        popularity = {"stages": {"start_game_items": [
            {"name": "Tango", "count": 24, "purchase_share_pct": 60}]},
            "stale": True, "items_stale": True, "fetched_at": "2026-11-12T00:00:00+00:00"}
        with patch.object(main, "get_heroes", return_value=[{"localized_name": "Beastmaster", "id": 38}]), \
             patch.object(main, "get_item_popularity", return_value=popularity) as fetch, \
             patch.object(main.client, "chat") as llm, \
             patch.object(main.hero_model, "predict") as model:
            response = TestClient(main.app).post("/chat", json={"hero": "Beastmaster",
                "prompt": "What are my start games in Turbo?", "history": [
                    {"role": "assistant", "content": "Pudge's Meat Hook is Beastmaster's starting item"}]})
        self.assertEqual(response.status_code, 200)
        answer = response.json()["answer"]
        self.assertIn("Beastmaster", answer)
        self.assertIn("Tango: 24 purchases", answer)
        self.assertIn("stale", answer)
        self.assertIn("not Turbo-specific", answer)
        self.assertNotIn("Meat Hook", answer)
        self.assertEqual(fetch.call_args.args, (38,))
        self.assertIsNone(response.json()["hero_prediction"])
        llm.assert_not_called()
        model.assert_not_called()

    def test_outage_does_not_fall_back_to_model_guesses(self):
        with patch.object(main, "get_heroes", return_value=[{"localized_name": "Beastmaster", "id": 38}]), \
             patch.object(main, "get_item_popularity", side_effect=RuntimeError("offline")), \
             patch.object(main.client, "chat") as llm:
            response = TestClient(main.app).post("/chat", json={"hero": "Beastmaster", "prompt": "Starting items?"})
        self.assertIn("unavailable", response.json()["answer"])
        llm.assert_not_called()

    def test_empty_and_unknown_hero_do_not_invent_items(self):
        self.assertIn("no starting purchases", main.starting_item_answer("Beastmaster", "Turbo", {"stages": {}}))
        self.assertIn("unavailable", main.starting_item_answer("Unknown", "Turbo", None))

    def test_intent_variants(self):
        for prompt in ("start games", "starting items?", "early game build", "start_game_items", "opening build"):
            self.assertTrue(main.asks_starting_items(prompt), prompt)
        self.assertFalse(main.asks_starting_items("Hello, how are you?"))


if __name__ == "__main__":
    unittest.main()
