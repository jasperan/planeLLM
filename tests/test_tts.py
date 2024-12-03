"""Unit tests for TTS module."""

import unittest
from unittest.mock import Mock, patch
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts import TTSGenerator

class TestTTSGenerator(unittest.TestCase):
    """Test cases for TTSGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        with patch('transformers.BarkModel.from_pretrained') as self.mock_model:
            with patch('transformers.AutoProcessor.from_pretrained') as self.mock_processor:
                self.generator = TTSGenerator(model_type="bark")

    def test_generate_speaker1_audio(self):
        """Test audio generation for speaker 1."""
        mock_audio = np.zeros(1000)
        self.generator.model.generate = Mock(return_value=mock_audio)
        
        audio_array, rate = self.generator.generate_speaker1_audio("Test text")
        self.assertEqual(rate, 24000)
        self.assertEqual(len(audio_array), 1000)

    def test_generate_speaker2_audio(self):
        """Test audio generation for speaker 2."""
        mock_audio = np.zeros(1000)
        self.generator.model.generate = Mock(return_value=mock_audio)
        
        audio_array, rate = self.generator.generate_speaker2_audio("Test text")
        self.assertEqual(rate, 24000)
        self.assertEqual(len(audio_array), 1000)

    def test_error_handling(self):
        """Test error handling in audio generation."""
        self.generator.model.generate = Mock(side_effect=Exception("Model Error"))
        
        with self.assertRaises(Exception):
            self.generator.generate_speaker1_audio("Test text")

    def test_invalid_model_type(self):
        """Test handling of invalid model type."""
        with self.assertRaises(ValueError):
            TTSGenerator(model_type="invalid_model")

if __name__ == '__main__':
    unittest.main() 