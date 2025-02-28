#!/usr/bin/env python
"""
Simple script to run the Parler audio generation test.

This script provides a convenient way to test Parler TTS functionality
without running the entire test suite.

These tests are ONLY for Parler TTS and will fail if Parler TTS is not available.

Usage:
    python run_parler_test.py
"""

import unittest
import sys
import os
import argparse

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_parler_availability():
    """Check if Parler TTS is available."""
    try:
        # Try to import Parler TTS
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
        except ImportError:
            from parler.tts import ParlerTTSForConditionalGeneration
            
        print("Parler TTS is available. Tests will proceed.")
        return True
    except ImportError:
        print("ERROR: Parler TTS is not available. Tests cannot proceed.")
        print("Please install Parler TTS with: pip install git+https://github.com/huggingface/parler-tts.git")
        return False

def main():
    """Run the Parler TTS tests."""
    # First check if Parler TTS is available
    if not check_parler_availability():
        print("Exiting tests as Parler TTS is not available.")
        sys.exit(1)
    
    # Import the test case only if Parler is available
    from test_parler_audio import TestParlerAudioGeneration
    
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