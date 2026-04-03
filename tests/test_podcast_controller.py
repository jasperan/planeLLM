#!/usr/bin/env python
"""Tests for the podcast controller module."""

import os
import tempfile
import unittest
from io import StringIO
from unittest.mock import MagicMock, mock_open, patch

from podcast_controller import main


class TestPodcastController(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.mock_args = [
            "podcast_controller.py",
            "--topic", "Test Topic",
            "--tts-model", "coqui",
            "--output", os.path.join(self.test_dir, "test_output.mp3"),
            "--transcript-length", "long",
        ]

    def tearDown(self):
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    @patch("podcast_controller.TopicExplorer")
    @patch("podcast_controller.PodcastWriter")
    @patch("podcast_controller.TTSGenerator")
    @patch("podcast_controller.ensure_runtime_config_ready")
    def test_main_function_uses_all_questions_and_supported_models(self, config_mock, mock_tts, mock_writer, mock_explorer):
        mock_explorer_instance = mock_explorer.return_value
        mock_explorer_instance.generate_questions.return_value = ["Question 1", "Question 2", "Question 3"]
        mock_explorer_instance.explore_question.return_value = "Answer to the question"

        mock_writer_instance = mock_writer.return_value
        mock_writer_instance.create_podcast_transcript.return_value = "Speaker 1: Hello\nSpeaker 2: Hi"

        mock_tts_instance = mock_tts.return_value
        mock_tts_instance.generate_podcast.return_value = os.path.join(self.test_dir, "test_output.mp3")

        with patch("sys.argv", self.mock_args):
            with patch("builtins.open", mock_open()):
                with patch("os.makedirs"):
                    with patch("os.listdir", return_value=[]):
                        main()

        config_mock.assert_called_once()
        mock_explorer.assert_called_once_with(config_file="config.yaml")
        self.assertEqual(mock_explorer_instance.explore_question.call_count, 3)
        mock_writer.assert_called_once_with(config_file="config.yaml", transcript_length="long")
        mock_tts.assert_called_once_with(model_type="coqui", config_file="config.yaml")
        mock_tts_instance.generate_podcast.assert_called_once()

    def test_main_exits_with_guidance_when_config_is_missing(self):
        with patch("sys.argv", ["podcast_controller.py", "--topic", "Test Topic"]):
            with patch("sys.stderr", new_callable=StringIO) as stderr:
                with self.assertRaises(SystemExit) as exc:
                    main()

        self.assertEqual(exc.exception.code, 1)
        self.assertIn("Copy config_example.yaml to config.yaml", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
