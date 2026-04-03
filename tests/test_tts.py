"""Tests for the TTS generator module."""

import os
import sys
import types
import unittest
from unittest.mock import patch

from tts_generator import TTSGenerator


class TestTTSGenerator(unittest.TestCase):
    def test_init_without_config_or_model_boot(self):
        tts = TTSGenerator(model_type="bark", initialize_model=False)
        self.assertEqual(tts.model_type, "bark")
        self.assertEqual(tts.config, {})

    def test_fish_reference_is_instance_local(self):
        tts = TTSGenerator(
            model_type="fish",
            initialize_model=False,
            fish_reference_id="ref-123",
        )
        self.assertEqual(tts.fish_reference_id, "ref-123")
        self.assertNotEqual(os.environ.get("FISH_REFERENCE_ID"), "ref-123")

    def test_fish_requires_fish_specific_runtime_config(self):
        fake_fishaudio = types.ModuleType("fishaudio")

        class _FakeFishAudio:
            def __init__(self, api_key: str = "", **kwargs):
                if not api_key:
                    raise ValueError("API key must be provided either as argument or via FISH_API_KEY environment variable")

        fake_fishaudio.FishAudio = _FakeFishAudio

        with patch.dict(sys.modules, {"fishaudio": fake_fishaudio}):
            with patch.dict(os.environ, {"FISH_API_KEY": "", "FISH_REFERENCE_ID": ""}, clear=False):
                with patch.object(TTSGenerator, "_init_bark") as bark_mock:
                    with self.assertRaisesRegex(RuntimeError, "FISH_API_KEY"):
                        TTSGenerator(model_type="fish")
                bark_mock.assert_not_called()

    def test_model_type_validation(self):
        with self.assertRaises(ValueError):
            TTSGenerator(model_type="invalid_model", initialize_model=False)


if __name__ == "__main__":
    unittest.main()
