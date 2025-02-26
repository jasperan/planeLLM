#!/usr/bin/env python
"""
Tests for the podcast controller module.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import sys

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main function from the podcast controller
from podcast_controller import main

class TestPodcastController(unittest.TestCase):
    """Test cases for the podcast controller."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Mock command line arguments
        self.mock_args = [
            "podcast_controller.py",
            "--topic", "Test Topic",
            "--tts-model", "bark",
            "--output", os.path.join(self.test_dir, "test_output.mp3")
        ]
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    @patch("podcast_controller.TopicExplorer")
    @patch("podcast_controller.PodcastWriter")
    @patch("podcast_controller.TTSGenerator")
    @patch("sys.argv")
    def test_main_function(self, mock_argv, mock_tts, mock_writer, mock_explorer):
        """Test the main function of the podcast controller."""
        # Set up mock argv
        mock_argv.__getitem__.side_effect = self.mock_args.__getitem__
        mock_argv.__len__.return_value = len(self.mock_args)
        
        # Set up mock TopicExplorer
        mock_explorer_instance = mock_explorer.return_value
        mock_explorer_instance.generate_questions.return_value = ["Question 1", "Question 2"]
        mock_explorer_instance.explore_question.return_value = "Answer to the question"
        
        # Set up mock PodcastWriter
        mock_writer_instance = mock_writer.return_value
        mock_writer_instance.create_podcast_transcript.return_value = "Expert: Hello\nStudent: Hi"
        
        # Set up mock TTSGenerator
        mock_tts_instance = mock_tts.return_value
        mock_tts_instance.generate_podcast.return_value = None
        
        # Call the main function
        with patch("builtins.open", unittest.mock.mock_open()):
            with patch("os.makedirs"):
                main()
        
        # Check that the TopicExplorer was initialized
        mock_explorer.assert_called_once()
        
        # Check that generate_questions was called with the correct topic
        mock_explorer_instance.generate_questions.assert_called_once_with("Test Topic")
        
        # Check that explore_question was called for each question
        self.assertEqual(mock_explorer_instance.explore_question.call_count, 2)
        
        # Check that the PodcastWriter was initialized
        mock_writer.assert_called_once()
        
        # Check that create_podcast_transcript was called
        mock_writer_instance.create_podcast_transcript.assert_called_once()
        
        # Check that the TTSGenerator was initialized with the correct model
        mock_tts.assert_called_once_with(model_type="bark")
        
        # Check that generate_podcast was called with the correct output path
        mock_tts_instance.generate_podcast.assert_called_once()
        self.assertTrue(mock_tts_instance.generate_podcast.call_args[1]["output_path"].endswith("test_output.mp3"))

if __name__ == "__main__":
    unittest.main() 