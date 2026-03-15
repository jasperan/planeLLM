"""Tests for the TTS generator module."""

import os
import unittest

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

    def test_model_type_validation(self):
        with self.assertRaises(ValueError):
            TTSGenerator(model_type="invalid_model", initialize_model=False)


if __name__ == "__main__":
    unittest.main()
