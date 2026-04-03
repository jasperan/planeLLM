import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import api_server
import api_workflow
from fastapi import HTTPException


class TestAPIWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.resources = Path(self.temp_dir.name) / "resources"
        self.resources.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_transcript_rejects_path_traversal(self):
        outside_file = Path(self.temp_dir.name) / "outside.txt"
        outside_file.write_text("secret", encoding="utf-8")

        fake_writer = MagicMock()
        fake_writer.create_podcast_transcript.return_value = "Speaker 1: safe"
        request = SimpleNamespace(content_file="../outside.txt", detailed=False)

        with self.assertRaises(ValueError):
            api_workflow.create_transcript_response(
                resources=self.resources,
                request=request,
                get_writer=lambda: fake_writer,
            )

    def test_generate_audio_does_not_mutate_process_environment(self):
        transcript_path = self.resources / "podcast_transcript_demo.txt"
        transcript_path.write_text("Speaker 1: Hello", encoding="utf-8")

        fake_tts = MagicMock()
        fake_tts.generate_podcast.return_value = str(self.resources / "podcast_demo.mp3")
        fake_tts_class = MagicMock(return_value=fake_tts)
        request = SimpleNamespace(
            transcript_file=transcript_path.name,
            tts_model="fish",
            fish_reference="ref-123",
            fish_emotion="(neutral)",
        )

        with api_workflow.patched_environ({}):
            response = api_workflow.generate_audio_response(
                resources=self.resources,
                request=request,
                get_tts_generator_class=lambda: fake_tts_class,
            )

        self.assertTrue(response["success"])
        self.assertNotIn("FISH_REFERENCE_ID", os.environ)

    def test_create_transcript_reports_missing_runtime_config_without_fake_404(self):
        request = api_server.TranscriptRequest(content_file="content_demo.txt", detailed=True)

        with patch.object(api_server, "RESOURCES", self.resources):
            (self.resources / request.content_file).write_text("lesson content", encoding="utf-8")
            with patch.object(
                api_server,
                "_get_writer",
                side_effect=FileNotFoundError("Config file not found: config.yaml"),
            ):
                response = api_server.create_transcript(request)

        self.assertFalse(response["success"])
        self.assertIn("config.yaml", response["message"])

    def test_create_transcript_still_404s_for_missing_content_file(self):
        request = api_server.TranscriptRequest(content_file="missing.txt", detailed=True)

        with patch.object(api_server, "RESOURCES", self.resources):
            with self.assertRaises(HTTPException) as exc:
                api_server.create_transcript(request)

        self.assertEqual(exc.exception.status_code, 404)
        self.assertEqual(exc.exception.detail, "File not found: missing.txt")

    @patch(
        "api_server.build_runtime_preflight",
        return_value={
            "oci_config": False,
            "ffmpeg": True,
            "fish_sdk": True,
            "resources_count": 10,
            "recommended_mode": "demo",
            "next_step": "Create a demo bundle.",
        },
    )
    def test_status_requires_real_oci_runtime_config(self, preflight_mock):
        status = api_server.get_status()

        self.assertFalse(status["oci_config"])
        self.assertTrue(status["ffmpeg"])
        self.assertEqual(status["resources_count"], 10)
        self.assertEqual(status["recommended_mode"], "demo")
        preflight_mock.assert_called_once()

    @patch(
        "api_server.build_runtime_preflight",
        return_value={
            "oci_config": False,
            "ffmpeg": True,
            "fish_sdk": True,
            "resources_count": 4,
            "recommended_mode": "demo",
            "next_step": "Create a demo bundle.",
        },
    )
    def test_preflight_endpoint_returns_runtime_details(self, preflight_mock):
        status = api_server.get_preflight()

        self.assertEqual(status["recommended_mode"], "demo")
        self.assertEqual(status["resources_count"], 4)
        preflight_mock.assert_called_once()

    @patch(
        "api_server.create_demo_bundle",
        return_value={
            "success": True,
            "message": "Created a demo bundle for 'Ancient Rome'.",
            "topic": "Ancient Rome",
            "questions_file": "questions_demo.txt",
            "content_file": "content_demo.txt",
            "transcript_file": "podcast_transcript_demo.txt",
            "audio_file": "podcast_demo.mp3",
            "audio_message": "Demo audio generated successfully.",
            "questions": ["Question 1?"],
        },
    )
    def test_bootstrap_demo_returns_bundle_metadata(self, bundle_mock):
        request = api_server.DemoRequest(topic="Ancient Rome")

        response = api_server.bootstrap_demo(request)

        self.assertTrue(response["success"])
        self.assertEqual(response["audio_file"], "podcast_demo.mp3")
        self.assertEqual(response["questions"], ["Question 1?"])
        bundle_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
