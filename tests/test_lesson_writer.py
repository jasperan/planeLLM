"""Unit tests for LessonWriter module."""

import unittest
from unittest.mock import Mock, patch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lesson_writer import PodcastWriter

class TestPodcastWriter(unittest.TestCase):
    """Test cases for PodcastWriter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'compartment_id': 'test_compartment',
            'config_profile': 'DEFAULT',
            'model_id': 'test_model'
        }
        with patch('builtins.open', create=True) as self.mock_open:
            self.mock_open.return_value.__enter__.return_value.read.return_value = str(self.mock_config)
            with patch('yaml.safe_load') as self.mock_yaml:
                self.mock_yaml.return_value = self.mock_config
                with patch('oci.config.from_file') as self.mock_oci_config:
                    self.writer = PodcastWriter()

    def test_create_podcast_transcript(self):
        """Test podcast transcript generation."""
        mock_response = Mock()
        mock_response.data = {'chat_response': {'choices': [{'message': {'content': [{'text': 'Speaker 1: Hello\nSpeaker 2: Hi'}]}}]}}
        
        with patch.object(self.writer.genai_client, 'chat', return_value=mock_response):
            transcript = self.writer.create_podcast_transcript("Test content")
            self.assertIsInstance(transcript, str)
            self.assertTrue('Speaker 1' in transcript)
            self.assertTrue('Speaker 2' in transcript)

    def test_timing_statistics(self):
        """Test execution time tracking."""
        mock_response = Mock()
        mock_response.data = {'chat_response': {'choices': [{'message': {'content': [{'text': 'Test response'}]}}]}}
        
        with patch.object(self.writer.genai_client, 'chat', return_value=mock_response):
            self.writer._call_llm("Test prompt")
            stats = self.writer._generate_timing_summary()
            self.assertTrue('Execution Time Summary' in stats)
            self.assertTrue('LLM Statistics' in stats)

    def test_error_handling(self):
        """Test error handling in transcript generation."""
        with patch.object(self.writer.genai_client, 'chat', side_effect=Exception("API Error")):
            with self.assertRaises(Exception):
                self.writer.create_podcast_transcript("Test content")

if __name__ == '__main__':
    unittest.main() 