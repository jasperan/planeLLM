"""Unit tests for TopicExplorer module."""

import unittest
from unittest.mock import MagicMock, patch

from topic_explorer import TopicExplorer


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
            first = explorer.generate_questions("Test Topic")
            second = explorer.generate_questions("Test Topic")

        self.assertEqual(first, ["Question 1?", "Question 2?", "Question 3?", "Question 4?"])
        self.assertEqual(second, first)
        self.assertEqual(batch_mock.call_count, 3)

    def test_explore_question_raises_after_retry_exhaustion(self):
        self.mock_client.chat.side_effect = Exception("API Error")
        explorer = TopicExplorer(config_data=self.mock_config, genai_client=self.mock_client)

        with patch("topic_explorer.time.sleep"):
            with self.assertRaises(RuntimeError):
                explorer.explore_question("Test question?")


if __name__ == "__main__":
    unittest.main()
