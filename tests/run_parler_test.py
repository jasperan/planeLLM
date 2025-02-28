#!/usr/bin/env python
"""
Simple script to run the Parler audio generation test.

This script provides a convenient way to test Parler TTS functionality
without running the entire test suite.

Usage:
    python run_parler_test.py
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the test case
from test_parler_audio import TestParlerAudioGeneration

if __name__ == "__main__":
    # Create a test suite with just the Parler tests
    suite = unittest.TestSuite()
    suite.addTest(TestParlerAudioGeneration('test_parler_initialization'))
    suite.addTest(TestParlerAudioGeneration('test_generate_audio_parler'))
    suite.addTest(TestParlerAudioGeneration('test_generate_podcast_with_parler'))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate status code
    sys.exit(not result.wasSuccessful()) 