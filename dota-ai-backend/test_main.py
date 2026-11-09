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
            self.assertEqual(response.json(), {"answer": "Hello", "ml_prediction": None})
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
