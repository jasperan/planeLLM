import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import api_server
import api_workflow


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

    @patch("api_server._count_resources", side_effect=[1, 2, 3, 4])
    @patch("api_server.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("api_server.has_oci_runtime_config", return_value=False)
    def test_status_requires_real_oci_runtime_config(self, config_mock, which_mock, count_mock):
        status = api_server.get_status()

        self.assertFalse(status["oci_config"])
        self.assertTrue(status["ffmpeg"])
        self.assertEqual(status["resources_count"], 10)
        config_mock.assert_called_once_with()
        which_mock.assert_called_once_with("ffmpeg")
        self.assertEqual(count_mock.call_count, 4)


if __name__ == "__main__":
    unittest.main()
