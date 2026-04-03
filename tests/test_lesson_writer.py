"""Unit tests for LessonWriter module."""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from lesson_writer import PodcastWriter


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


class _FakeServiceError(Exception):
    def __str__(self):
        return "{'status': 404, 'message': 'Entity with key ocid1.generativeaimodel.oc1.test not found'}"


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

    def test_call_llm_surfaces_stale_model_guidance(self):
        self.mock_client.chat.side_effect = _FakeServiceError()
        writer = PodcastWriter(
            config_data=self.mock_config,
            genai_client=self.mock_client,
        )

        with patch.dict(sys.modules, {"oci": _fake_oci_module()}):
            with self.assertRaisesRegex(RuntimeError, "Configured model_id 'test_model' was not found"):
                writer._call_llm("Test prompt")


if __name__ == "__main__":
    unittest.main()
