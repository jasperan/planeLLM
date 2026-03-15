"""Unit tests for LessonWriter module."""

import unittest
from unittest.mock import MagicMock, patch

from lesson_writer import PodcastWriter


class TestPodcastWriter(unittest.TestCase):
    def setUp(self):
        self.mock_config = {
            "compartment_id": "test_compartment",
            "config_profile": "DEFAULT",
            "model_id": "test_model",
        }
        self.mock_client = MagicMock()

    def test_transcript_length_configures_prompt_targets(self):
        writer = PodcastWriter(
            config_data=self.mock_config,
            genai_client=self.mock_client,
            transcript_length="long",
        )

        self.assertEqual(writer.transcript_length, "long")
        self.assertIn("18-24", writer.system_prompt)

    def test_detailed_transcript_uses_section_local_context(self):
        writer = PodcastWriter(
            config_data=self.mock_config,
            genai_client=self.mock_client,
        )

        prompts = []

        def fake_call(prompt):
            prompts.append(prompt)
            if "introduction section" in prompt:
                return "Speaker 1: Intro"
            if "conclusion section" in prompt:
                return "Speaker 1: Outro"
            return "Speaker 1: Segment"

        content = """# Demo Topic

## Question One?

Answer one only.

## Question Two?

Answer two only.
"""

        with patch.object(writer, "_call_llm", side_effect=fake_call):
            transcript = writer.create_detailed_podcast_transcript(content)

        self.assertIn("Speaker 1: Intro", transcript)
        self.assertIn("Speaker 1: Outro", transcript)
        self.assertIn("Answer one only.", prompts[1])
        self.assertNotIn("Answer two only.", prompts[1])
        self.assertIn("Answer two only.", prompts[2])
        self.assertNotIn("Answer one only.", prompts[2])


if __name__ == "__main__":
    unittest.main()
