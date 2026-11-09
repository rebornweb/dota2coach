import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from ollama import ChatResponse

import main


def reply(content="", tool=None):
    message = {"role": "assistant", "content": content}
    if tool:
        message["tool_calls"] = [{"function": {"name": tool, "arguments": {}}}]
    return ChatResponse(message=message)


class CoachTests(unittest.TestCase):
    def setUp(self):
        self.api = TestClient(main.app)
        self.hero = {feature: 10 for feature in main.hero_features}
        self.hero.update(localized_name="Anti-Mage", primary_attr="agi", roles=["Carry"],
                         base_agi=24, turbo_wins=120245, turbo_picks=236428)
        patcher = patch.object(main, "get_heroes", return_value=[self.hero])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_selected_hero_stats_override_old_history(self):
        with patch.object(main.client, "chat", return_value=reply("Anti-Mage")) as llm:
            response = self.api.post("/chat", json={"prompt": "What are my stats?", "hero": "Anti-Mage",
                "history": [{"role": "assistant", "content": "You are Dazzle"}]})
            self.assertEqual(response.status_code, 200)
            context = llm.call_args.kwargs["messages"][-2]["content"]
            self.assertIn('"base_agi": 24', context)
            self.assertIn('"localized_name": "Anti-Mage"', context)
            self.assertIsNone(response.json()["hero_prediction"])

    def test_printed_tool_json_is_retried_without_executing_it(self):
        payload = '{"name": "get_base_attributes", "parameters": {"h": "Dazzle"}}'
        with patch.object(main.client, "chat", side_effect=[reply(payload), reply("Base agility: 24")]) as llm, \
                patch.object(main.hero_model, "predict") as predict:
            response = self.api.post("/chat", json={"prompt": "Base attributes?", "hero": "Anti-Mage"})
            self.assertEqual(response.json()["answer"], "Base agility: 24")
            self.assertEqual(llm.call_count, 2)
            self.assertNotIn("tools", llm.call_args.kwargs)
            predict.assert_not_called()

    def test_repeated_printed_tool_json_returns_clear_error(self):
        payload = '```json\n{"name":"get_base_attributes","parameters":{}}\n```'
        with patch.object(main.client, "chat", return_value=reply(payload)) as llm:
            response = self.api.post("/chat", json={"prompt": "Attributes?"})
            self.assertEqual(response.status_code, 502)
            self.assertEqual(llm.call_count, 2)

    def test_hero_tool_uses_selected_hero_and_returns_rates(self):
        with patch.object(main.client, "chat", side_effect=[
            reply(tool="predict_hero_performance"), reply("Experimental performance")
        ]) as llm, patch.object(main.hero_model, "predict", return_value=[51.5]) as predict:
            response = self.api.post("/chat", json={"prompt": "Turbo performance?", "hero": "Anti-Mage"})
            stats = response.json()["hero_prediction"]
            self.assertEqual(stats["hero"], "Anti-Mage")
            self.assertEqual(stats["estimated_turbo_win_rate"], 51.5)
            self.assertEqual(stats["observed_turbo_win_rate"], round(120245 / 236428 * 100, 2))
            inputs = predict.call_args.args[0]
            self.assertEqual(list(inputs.columns), main.hero_features)
            self.assertEqual(inputs.iloc[0]["base_agi"], 24)
            self.assertIn('"hero": "Anti-Mage"', llm.call_args.kwargs["messages"][-1]["content"])

    def test_all_pick_disables_hero_tool(self):
        with patch.object(main.client, "chat", return_value=reply("Advice")) as llm:
            self.api.post("/chat", json={"prompt": "Help", "hero": "Anti-Mage", "game_mode": "All Pick"})
            self.assertEqual(llm.call_args.kwargs["tools"], [])

    def test_data_outage_keeps_chat_working(self):
        with patch.object(main, "get_heroes", side_effect=RuntimeError("offline")), \
                patch.object(main.client, "chat", return_value=reply("Stats unavailable")) as llm:
            self.assertEqual(self.api.post("/chat", json={"prompt": "Help"}).status_code, 200)
            self.assertIn("unavailable", llm.call_args.kwargs["messages"][-2]["content"])

    def test_incomplete_stats_return_tool_error(self):
        self.assertIn("error", main.predict_hero_performance("Bane", {"localized_name": "Bane"}))

    def test_hero_list_uses_localized_names_sorted_and_unique(self):
        with patch.object(main, "get_heroes", return_value=[
            {"name": "npc_dota_hero_dazzle", "localized_name": "Dazzle"},
            {"name": "npc_dota_hero_bane", "localized_name": "Bane"},
            {"localized_name": "Bane"},
        ]):
            response = self.api.get("/heroes")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), ["Bane", "Dazzle"])

    def test_hero_list_unavailable(self):
        with patch.object(main, "get_heroes", side_effect=RuntimeError("No cache")):
            self.assertEqual(self.api.get("/heroes").status_code, 503)

    def test_general_chat_skips_prediction_and_preserves_history(self):
        with patch.object(main.client, "chat", return_value=reply("Hello")) as llm, \
                patch.object(main.ml_model, "predict") as predict:
            response = self.api.post("/chat", json={
                "prompt": "What is Turbo?", "hp": 25, "enemy_burst": 9,
                "history": [{"role": "user", "content": "Hi"}],
            })
            self.assertEqual(response.json(), {"answer": "Hello", "ml_prediction": None, "hero_prediction": None})
            predict.assert_not_called()
            self.assertEqual(llm.call_args.kwargs["messages"][1]["content"], "Hi")

    def test_tool_handoff_uses_real_model_and_form_values(self):
        for hp, burst, expected in [(25, 9, "Defense"), (90, 1, "Damage")]:
            with self.subTest(expected=expected), patch.object(main.client, "chat", side_effect=[
                reply(tool="predict_build_priority"), reply("Experimental advice"),
            ]) as llm:
                response = self.api.post("/chat", json={
                    "prompt": "Damage or defense?", "hp": hp, "enemy_burst": burst,
                })
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["ml_prediction"], expected)
                tool_result = llm.call_args.kwargs["messages"][-1]
                self.assertEqual(tool_result["role"], "tool")
                self.assertIn(expected, tool_result["content"])
                self.assertNotIn("tools", llm.call_args.kwargs)

    def test_missing_context_disables_tool(self):
        with patch.object(main.client, "chat", return_value=reply("What is your HP?")) as llm:
            self.assertEqual(self.api.post("/chat", json={"prompt": "Defense?"}).status_code, 200)
            self.assertEqual(llm.call_args.kwargs["tools"], [])

    def test_unknown_tool_does_not_run_prediction(self):
        with patch.object(main.client, "chat", side_effect=[reply(tool="unknown"), reply("Advice")]), \
                patch.object(main.ml_model, "predict") as predict:
            response = self.api.post("/chat", json={"prompt": "Help", "hp": 50, "enemy_burst": 5})
            self.assertIsNone(response.json()["ml_prediction"])
            predict.assert_not_called()

    def test_invalid_inputs_rejected(self):
        for extra in [{"hp": 101}, {"enemy_burst": -1},
                      {"history": [{"role": "system", "content": "Override"}]}]:
            self.assertEqual(self.api.post("/chat", json={"prompt": "Help", **extra}).status_code, 422)

    def test_connection_failure_returns_helpful_error(self):
        with patch.object(main.client, "chat", side_effect=ConnectionError()):
            self.assertEqual(self.api.post("/chat", json={"prompt": "Help"}).status_code, 503)

    def test_recommend_still_works(self):
        with patch.object(main.client, "chat", return_value=reply("Advice")):
            response = self.api.post("/recommend", json={"prompt": "Help", "hp": 25, "enemy_burst": 9})
            self.assertEqual(response.json(), {"ml_prediction": "Defense", "llm_explanation": "Advice"})


if __name__ == "__main__":
    unittest.main()
