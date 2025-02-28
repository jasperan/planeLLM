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
import argparse

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the test case
from test_parler_audio import TestParlerAudioGeneration

def main():
    """Run the Parler TTS tests."""
    parser = argparse.ArgumentParser(description='Test Parler TTS audio generation')
    parser.add_argument('--voice-test', action='store_true', help='Run voice selection test only')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--init', action='store_true', help='Run initialization test only')
    parser.add_argument('--generate', action='store_true', help='Run audio generation test only')
    parser.add_argument('--podcast', action='store_true', help='Run podcast generation test only')
    args = parser.parse_args()
    
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add tests based on arguments
    if args.all or not any([args.voice_test, args.init, args.generate, args.podcast]):
        # If --all is specified or no specific test is requested, run all tests
        suite.addTest(TestParlerAudioGeneration('test_parler_initialization'))
        suite.addTest(TestParlerAudioGeneration('test_generate_audio_parler'))
        suite.addTest(TestParlerAudioGeneration('test_generate_podcast_with_parler'))
        suite.addTest(TestParlerAudioGeneration('test_parler_voice_selection'))
    else:
        # Add specific tests based on arguments
        if args.init:
            suite.addTest(TestParlerAudioGeneration('test_parler_initialization'))
        if args.generate:
            suite.addTest(TestParlerAudioGeneration('test_generate_audio_parler'))
        if args.podcast:
            suite.addTest(TestParlerAudioGeneration('test_generate_podcast_with_parler'))
        if args.voice_test:
            suite.addTest(TestParlerAudioGeneration('test_parler_voice_selection'))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate status code
    sys.exit(not result.wasSuccessful())

if __name__ == "__main__":
    main() 