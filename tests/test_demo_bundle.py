"""Tests for the demo bundle helper."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from demo_bundle import create_demo_bundle


class TestDemoBundle(unittest.TestCase):
    def test_create_demo_bundle_writes_pipeline_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            resources = Path(temp_dir) / "resources"
            with patch("demo_bundle._build_demo_audio") as audio_mock:
                audio_mock.return_value.export.side_effect = lambda output_path, format="mp3": Path(output_path).write_bytes(b"demo")
                bundle = create_demo_bundle("Ancient Rome", resources_dir=resources)

            self.assertTrue((resources / bundle["questions_file"]).exists())
            self.assertTrue((resources / bundle["content_file"]).exists())
            self.assertTrue((resources / bundle["transcript_file"]).exists())
            self.assertTrue((resources / bundle["audio_file"]).exists())
            self.assertEqual(len(bundle["questions"]), 4)

    def test_create_demo_bundle_reports_missing_audio_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            resources = Path(temp_dir) / "resources"
            with patch("demo_bundle.shutil.which", return_value=None):
                bundle = create_demo_bundle("Ancient Rome", resources_dir=resources)

            self.assertEqual(bundle["audio_file"], "")
            self.assertIn("FFmpeg", bundle["audio_message"])
