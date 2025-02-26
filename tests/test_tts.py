#!/usr/bin/env python
"""
Tests for the TTS generator module.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile

# Import the TTSGenerator class from the adapter module
from tts_generator import TTSGenerator

class TestTTSGenerator(unittest.TestCase):
    """Test cases for the TTSGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a sample transcript file
        self.transcript_path = os.path.join(self.test_dir, "test_transcript.txt")
        with open(self.transcript_path, "w", encoding="utf-8") as f:
            f.write("Expert: Hello, today we'll discuss quantum physics.\n")
            f.write("Student: That sounds interesting!\n")
            f.write("Expert: Let's start with the basics.\n")
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary files
        if os.path.exists(self.transcript_path):
            os.remove(self.transcript_path)
        
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    @patch("tts_generator.TTSGenerator")
    def test_init(self, mock_tts):
        """Test initialization of TTSGenerator."""
        # Create a TTSGenerator instance
        tts = TTSGenerator(model_type="bark")
        
        # Check that the model_type is set correctly
        self.assertEqual(tts.model_type, "bark")
    
    @patch("tts_generator.TTSGenerator.generate_podcast")
    def test_generate_podcast_with_output_path(self, mock_generate):
        """Test generate_podcast with a specified output path."""
        # Create a TTSGenerator instance
        tts = TTSGenerator(model_type="bark")
        
        # Call generate_podcast with an output path
        output_path = os.path.join(self.test_dir, "test_output.mp3")
        tts.generate_podcast(self.transcript_path, output_path=output_path)
        
        # Check that generate_podcast was called with the correct arguments
        mock_generate.assert_called_once_with(self.transcript_path, output_path=output_path)
    
    @patch("tts_generator.TTSGenerator.generate_podcast")
    def test_generate_podcast_without_output_path(self, mock_generate):
        """Test generate_podcast without a specified output path."""
        # Create a TTSGenerator instance
        tts = TTSGenerator(model_type="bark")
        
        # Call generate_podcast without an output path
        tts.generate_podcast(self.transcript_path)
        
        # Check that generate_podcast was called with the correct arguments
        mock_generate.assert_called_once_with(self.transcript_path, output_path=None)
    
    @patch("tts_generator.TTSGenerator")
    def test_model_type_validation(self, mock_tts):
        """Test validation of model_type."""
        # Test with valid model types
        tts1 = TTSGenerator(model_type="bark")
        self.assertEqual(tts1.model_type, "bark")
        
        tts2 = TTSGenerator(model_type="parler")
        self.assertEqual(tts2.model_type, "parler")
        
        # Test with invalid model type
        with self.assertRaises(ValueError):
            TTSGenerator(model_type="invalid_model")

if __name__ == "__main__":
    unittest.main() 