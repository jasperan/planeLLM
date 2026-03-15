import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import api_server


class TestAPIServer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.resources = Path(self.temp_dir.name) / "resources"
        self.resources.mkdir()
        self.client = TestClient(api_server.app)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_transcript_rejects_path_traversal(self):
        outside_file = Path(self.temp_dir.name) / "outside.txt"
        outside_file.write_text("secret", encoding="utf-8")

        fake_writer = MagicMock()
        fake_writer.create_podcast_transcript.return_value = "Speaker 1: safe"

        with patch.object(api_server, "RESOURCES", self.resources):
            with patch.object(api_server, "_get_writer", return_value=fake_writer):
                response = self.client.post(
                    "/api/transcript/create",
                    json={"content_file": "../outside.txt", "detailed": False},
                )

        self.assertEqual(response.status_code, 400)

    def test_generate_audio_does_not_mutate_process_environment(self):
        transcript_path = self.resources / "podcast_transcript_demo.txt"
        transcript_path.write_text("Speaker 1: Hello", encoding="utf-8")

        fake_tts = MagicMock()
        fake_tts.generate_podcast.return_value = str(self.resources / "podcast_demo.mp3")

        with patch.object(api_server, "RESOURCES", self.resources):
            with patch.dict(os.environ, {}, clear=True):
                with patch("tts_generator.TTSGenerator", return_value=fake_tts):
                    response = self.client.post(
                        "/api/audio/generate",
                        json={
                            "transcript_file": transcript_path.name,
                            "tts_model": "fish",
                            "fish_reference": "ref-123",
                            "fish_emotion": "(neutral)",
                        },
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertNotIn("FISH_REFERENCE_ID", os.environ)


if __name__ == "__main__":
    unittest.main()
