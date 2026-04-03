"""Unit tests for TopicExplorer module."""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from topic_explorer import TopicExplorer


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _ImmediateExecutor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        return _ImmediateFuture(fn(*args, **kwargs))


class _ExecutorFactory:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return _ImmediateExecutor()


def _fake_oci_module():
    class _Model:
        API_FORMAT_GENERIC = "generic"

        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

    models = types.SimpleNamespace(
        TextContent=_Model,
        Message=_Model,
        GenericChatRequest=_Model,
        BaseChatRequest=_Model,
        ChatDetails=_Model,
        OnDemandServingMode=_Model,
    )
    return types.SimpleNamespace(
        generative_ai_inference=types.SimpleNamespace(models=models)
    )


class TestTopicExplorer(unittest.TestCase):
    def setUp(self):
        self.mock_config = {
            "compartment_id": "test_compartment",
            "config_profile": "DEFAULT",
            "model_id": "test_model",
        }
        self.mock_client = MagicMock()

    def test_generate_questions_dedupes_and_uses_cache(self):
        explorer = TopicExplorer(config_data=self.mock_config, genai_client=self.mock_client)

        with patch.object(
            explorer,
            "_generate_question_batch",
            side_effect=[
                ["Question 1?", "Question 2?"],
                ["Question 1?", "Question 3?"],
                ["Question 4?"],
            ],
        ) as batch_mock:
            with patch("topic_explorer.ThreadPoolExecutor", return_value=_ImmediateExecutor()):
                with patch("topic_explorer.as_completed", side_effect=lambda futures: list(futures)):
                    first = explorer.generate_questions("Test Topic")
                    second = explorer.generate_questions("Test Topic")

        self.assertEqual(first, ["Question 1?", "Question 2?", "Question 3?", "Question 4?"])
        self.assertEqual(second, first)
        self.assertEqual(batch_mock.call_count, 3)

    def test_generate_topic_bundle_reuses_one_executor(self):
        explorer = TopicExplorer(config_data=self.mock_config, genai_client=self.mock_client)
        executor_factory = _ExecutorFactory()

        def fake_explore(question, results):
            results[question] = f"Answer for {question}"
            explorer.execution_times["responses"][question] = 0.01

        with patch.object(
            explorer,
            "_generate_question_batch",
            side_effect=[
                ["Question 1?", "Question 2?"],
                ["Question 3?"],
                ["Question 4?"],
            ],
        ):
            with patch.object(explorer, "_explore_question_thread", side_effect=fake_explore):
                with patch("topic_explorer.ThreadPoolExecutor", side_effect=executor_factory):
                    with patch("topic_explorer.as_completed", side_effect=lambda futures: list(futures)):
                        bundle = explorer.generate_topic_bundle("Test Topic", save=False)

        self.assertEqual(executor_factory.calls, 1)
        self.assertEqual(bundle["questions"], ["Question 1?", "Question 2?", "Question 3?", "Question 4?"])
        self.assertIn("Answer for Question 1?", bundle["content"])

    def test_generate_topic_bundle_uses_cached_questions(self):
        explorer = TopicExplorer(config_data=self.mock_config, genai_client=self.mock_client)

        def fake_explore(question, results):
            results[question] = f"Answer for {question}"
            explorer.execution_times["responses"][question] = 0.01

        with patch.object(
            explorer,
            "_generate_question_batch",
            side_effect=[
                ["Question 1?", "Question 2?"],
                ["Question 3?"],
                ["Question 4?"],
            ],
        ) as batch_mock:
            with patch.object(explorer, "_explore_question_thread", side_effect=fake_explore):
                with patch("topic_explorer.as_completed", side_effect=lambda futures: list(futures)):
                    first = explorer.generate_topic_bundle("Test Topic", save=False)
                    second = explorer.generate_topic_bundle("Test Topic", save=False)

        self.assertEqual(batch_mock.call_count, 3)
        self.assertEqual(second["questions"], first["questions"])
        self.assertEqual(second["content"], first["content"])

    def test_explore_question_returns_llm_text(self):
        explorer = TopicExplorer(config_data=self.mock_config, genai_client=self.mock_client)

        with patch.object(explorer, "_call_llm", return_value="Detailed answer") as call_mock:
            answer = explorer.explore_question("Test question?")

        self.assertEqual(answer, "Detailed answer")
        call_mock.assert_called_once()

    def test_make_llm_call_raises_after_retry_exhaustion(self):
        self.mock_client.chat.side_effect = Exception("API Error")
        explorer = TopicExplorer(config_data=self.mock_config, genai_client=self.mock_client)

        with patch.dict(sys.modules, {"oci": _fake_oci_module()}):
            with patch("topic_explorer.time.sleep"):
                with self.assertRaises(RuntimeError):
                    explorer._make_llm_call("Test prompt")


if __name__ == "__main__":
    unittest.main()
