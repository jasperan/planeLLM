"""Tests for Gradio quick-start integration."""

import unittest
from unittest.mock import patch

from gradio_app import PlaneLLMInterface, create_interface


class TestPlaneLLMInterface(unittest.TestCase):
    @patch(
        "gradio_app.build_runtime_preflight",
        return_value={
            "config_file_present": False,
            "config_profile": "DEFAULT",
            "config_profile_source": "~/.oci/config",
            "oci_profiles": ["DEFAULT", "plane"],
            "live_ready": False,
            "demo_ready": True,
            "oci_auth": True,
            "ffmpeg": True,
            "fish_sdk": True,
            "fish_api_key": False,
            "resources_count": 7,
            "recommended_mode": "demo",
            "next_step": "Create a demo bundle.",
            "issues": ["Config file not found: config.yaml"],
        },
    )
    def test_preflight_report_includes_mode_and_profiles(self, preflight_mock):
        interface = PlaneLLMInterface()

        report = interface.get_preflight_report()

        self.assertIn("Recommended mode: demo", report)
        self.assertIn("Detected OCI profiles: DEFAULT, plane", report)
        preflight_mock.assert_called()

    @patch(
        "gradio_app.create_demo_bundle",
        return_value={
            "questions_text": "1. Question?",
            "content": "Demo content",
            "message": "Created a demo bundle.",
            "audio_message": "Demo audio generated successfully.",
        },
    )
    def test_create_demo_topic_content_returns_bundle_outputs(self, bundle_mock):
        interface = PlaneLLMInterface()

        questions, content, report, status = interface.create_demo_topic_content("Ancient Rome")

        self.assertEqual(questions, "1. Question?")
        self.assertEqual(content, "Demo content")
        self.assertIn("Created a demo bundle.", status)
        self.assertIsInstance(report, str)
        bundle_mock.assert_called_once()

    def test_create_interface_builds_blocks(self):
        app = create_interface()
        self.assertEqual(type(app).__name__, "Blocks")


if __name__ == "__main__":
    unittest.main()
