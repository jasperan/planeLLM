"""Unit tests for TopicExplorer module."""

import unittest
from unittest.mock import Mock, patch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topic_explorer import TopicExplorer

class TestTopicExplorer(unittest.TestCase):
    """Test cases for TopicExplorer class."""

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
                    self.explorer = TopicExplorer()

    def test_generate_questions(self):
        """Test question generation."""
        mock_response = Mock()
        mock_response.data = {'chat_response': {'choices': [{'message': {'content': [{'text': 'Question 1?\nQuestion 2?'}]}}]}}
        
        with patch.object(self.explorer.genai_client, 'chat', return_value=mock_response):
            questions = self.explorer.generate_questions("Test Topic")
            self.assertEqual(len(questions), 2)
            self.assertTrue(all('?' in q for q in questions))

    def test_explore_question(self):
        """Test detailed answer generation."""
        mock_response = Mock()
        mock_response.data = {'chat_response': {'choices': [{'message': {'content': [{'text': 'Detailed answer'}]}}]}}
        
        with patch.object(self.explorer.genai_client, 'chat', return_value=mock_response):
            answer = self.explorer.explore_question("Test question?")
            self.assertIsInstance(answer, str)
            self.assertTrue(len(answer) > 0)

    def test_error_handling(self):
        """Test error handling in LLM calls."""
        with patch.object(self.explorer.genai_client, 'chat', side_effect=Exception("API Error")):
            with self.assertRaises(Exception):
                self.explorer.generate_questions("Test Topic")

if __name__ == '__main__':
    unittest.main() 