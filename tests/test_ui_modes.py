import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch
from assistant import ui
from assistant.adapters import MockAdapter, proposal_json


class UIModeTests(unittest.TestCase):
    def invoke(self, commands, *flags):
        output = io.StringIO()
        with patch("sys.argv", ["assistant.ui", "example_objective.json", *flags]), \
             patch("sys.stdin", io.StringIO(commands)), redirect_stdout(output), redirect_stderr(output):
            ui.main()
        return output.getvalue()

    def test_default_is_offline_even_with_api_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "dummy-offline-value"}), \
             patch("assistant.openai_model._client", side_effect=AssertionError("API forbidden")):
            output = self.invoke("CONFIRM\nanswer Nine to five\nABORT\n")
        self.assertIn("SCRIPTED MODE", output)
        self.assertIn("APPROVED MODEL SPEECH", output)
        self.assertNotIn("dummy-offline-value", output)

    def test_api_mode_requires_both_flags(self):
        for flags in (("--adapter", "openai"), ("--allow-api",)):
            with self.assertRaises(SystemExit) as error:
                self.invoke("", *flags)
            self.assertEqual(error.exception.code, 2)

    def test_api_mode_cancellation_never_requests(self):
        adapter = MockAdapter([proposal_json()])
        with patch("assistant.openai_model.OpenAIRealtimeAdapter", return_value=adapter):
            self.invoke("CANCEL\n", "--adapter", "openai", "--allow-api")
        self.assertEqual(adapter.contexts, [])

    def test_ollama_backend_env_selection_never_falls_back_to_openai(self):
        adapter = MockAdapter([ConnectionError("local Ollama unavailable")])
        with patch.dict("os.environ", {"PHONEBOT_MODEL_BACKEND": "ollama"}), \
             patch("assistant.ollama_model.OllamaModelAdapter", return_value=adapter), \
             patch("assistant.openai_model.OpenAIRealtimeAdapter",
                   side_effect=AssertionError("OpenAI fallback forbidden")):
            output = self.invoke("CONFIRM\nanswer Nine to five\nABORT\n")
        self.assertIn("OLLAMA MODE", output)
        self.assertIn("ASK_USER", output)
        self.assertEqual(len(adapter.contexts), 1)

    def test_explicit_scripted_mode_overrides_ollama_environment(self):
        with patch.dict("os.environ", {"PHONEBOT_MODEL_BACKEND": "ollama"}), \
             patch("assistant.ollama_model.OllamaModelAdapter",
                   side_effect=AssertionError("Scripted mode must not load Ollama")):
            output = self.invoke("CONFIRM\nanswer Nine to five\nABORT\n", "--adapter", "scripted")
        self.assertIn("SCRIPTED MODE", output)

    def test_mocked_api_ui_and_separate_local_audit_export(self):
        adapter = MockAdapter([proposal_json(text="I agree to the contract.")])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audit.json"
            with patch("assistant.openai_model.OpenAIRealtimeAdapter", return_value=adapter):
                output = self.invoke("CONFIRM\nanswer Nine to five\nABORT\n",
                                     "--adapter", "openai", "--allow-api", "--save", str(path))
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("NOT speech", output)
        self.assertIn("ASK_USER", output)
        self.assertEqual(data["model_proposals"][0]["decision"], "rejected")
        self.assertNotIn("I agree to the contract.", [row["text"] for row in data["transcript"]])


if __name__ == "__main__":
    unittest.main()
