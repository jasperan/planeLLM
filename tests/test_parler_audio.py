#!/usr/bin/env python
"""
Test for Parler TTS audio generation with a sample sentence.

This test file specifically tests the Parler TTS model's ability to generate
audio from a simple sample sentence. It can be used to verify that the Parler
TTS integration is working correctly.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pydub import AudioSegment

# Import the TTSGenerator class
from tts_generator import TTSGenerator

class TestParlerAudioGeneration(unittest.TestCase):
    """Test cases for Parler audio generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.test_dir, "parler_test_output.mp3")
        
        # Sample sentence for testing
        self.sample_text = "Speaker 1: This is a test sentence for Parler TTS generation."
        
        # Create a sample transcript file
        self.transcript_path = os.path.join(self.test_dir, "parler_test_transcript.txt")
        with open(self.transcript_path, "w", encoding="utf-8") as f:
            f.write(self.sample_text)
        
        # Sample voice descriptions for testing
        self.voice_descriptions = {
            "male_clear": "Jon's voice is clear and professional with a moderate pace, with very clear audio that has no background noise.",
            "female_clear": "Laura's voice is clear and expressive with a moderate pace, with very clear audio that has no background noise.",
            "male_expressive": "Gary's voice is expressive and animated with varied intonation, with very clear audio that has no background noise.",
            "female_expressive": "Jenna's voice is expressive and animated with varied intonation, with very clear audio that has no background noise."
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary files
        if os.path.exists(self.transcript_path):
            os.remove(self.transcript_path)
        
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_parler_initialization(self):
        """Test initialization of TTSGenerator with Parler model."""
        try:
            # Create a TTSGenerator instance with Parler model
            tts = TTSGenerator(model_type="parler")
            
            # Check that the model_type is set correctly
            self.assertEqual(tts.model_type, "parler")
            
            # Check if Parler is available (if not, it should fall back to Bark)
            if hasattr(tts, 'parler_available') and not tts.parler_available:
                self.assertEqual(tts.model_type, "bark")
                print("Note: Parler TTS not available, test skipped")
                return
                
            # Verify speaker presets are defined
            self.assertIn("Speaker 1", tts.speakers)
            self.assertIn("Speaker 2", tts.speakers)
            
            print("Parler TTS initialized successfully")
        except Exception as e:
            self.fail(f"Parler initialization failed with error: {str(e)}")
    
    def test_generate_audio_parler(self):
        """Test generating audio with Parler TTS."""
        try:
            # Create a TTSGenerator instance with Parler model
            tts = TTSGenerator(model_type="parler")
            
            # Check if Parler is available (if not, it should fall back to Bark)
            if hasattr(tts, 'parler_available') and not tts.parler_available:
                self.assertEqual(tts.model_type, "bark")
                print("Note: Parler TTS not available, test skipped")
                return
            
            # Generate audio from the sample text
            if hasattr(tts, '_generate_audio_parler'):
                # Direct test of the Parler generation method
                audio_segment = tts._generate_audio_parler("This is a test sentence.", "Speaker 1")
                
                # Verify that audio was generated
                self.assertIsInstance(audio_segment, AudioSegment)
                self.assertGreater(len(audio_segment), 0)
                
                # Export the audio segment to verify it works
                audio_segment.export(self.output_path, format="mp3")
                self.assertTrue(os.path.exists(self.output_path))
                self.assertGreater(os.path.getsize(self.output_path), 0)
                
                print(f"Parler audio generated successfully: {self.output_path}")
            else:
                print("Note: _generate_audio_parler method not available, test skipped")
        except Exception as e:
            self.fail(f"Parler audio generation failed with error: {str(e)}")
    
    def test_generate_podcast_with_parler(self):
        """Test generating a podcast with Parler TTS."""
        try:
            # Create a TTSGenerator instance with Parler model
            tts = TTSGenerator(model_type="parler")
            
            # Check if Parler is available (if not, it should fall back to Bark)
            if hasattr(tts, 'parler_available') and not tts.parler_available:
                self.assertEqual(tts.model_type, "bark")
                print("Note: Parler TTS not available, test skipped")
                return
            
            # Generate podcast from the transcript file
            output_file = tts.generate_podcast(self.transcript_path, output_path=self.output_path)
            
            # Verify that the output file exists and has content
            self.assertTrue(os.path.exists(output_file))
            self.assertGreater(os.path.getsize(output_file), 0)
            
            print(f"Podcast generated with Parler TTS: {output_file}")
        except Exception as e:
            self.fail(f"Podcast generation with Parler failed with error: {str(e)}")
    
    def test_parler_voice_selection(self):
        """Test Parler TTS with different voice descriptions."""
        try:
            # Create a TTSGenerator instance with Parler model
            tts = TTSGenerator(model_type="parler")
            
            # Check if Parler is available (if not, it should fall back to Bark)
            if hasattr(tts, 'parler_available') and not tts.parler_available:
                self.assertEqual(tts.model_type, "bark")
                print("Note: Parler TTS not available, test skipped")
                return
            
            # Test if the voice_descriptions attribute exists
            if not hasattr(tts, 'voice_descriptions'):
                print("Note: voice_descriptions attribute not available, test skipped")
                return
            
            # Test generating audio with different voice descriptions
            for voice_name, description in self.voice_descriptions.items():
                print(f"Testing voice: {voice_name}")
                
                # Set the voice description for Speaker 1
                tts.set_voice_description("Speaker 1", description)
                
                # Generate audio
                audio_segment = tts._generate_audio_parler("This is a test with different voices.", "Speaker 1")
                
                # Verify that audio was generated
                self.assertIsInstance(audio_segment, AudioSegment)
                self.assertGreater(len(audio_segment), 0)
                
                # Export to a voice-specific output file
                voice_output_path = os.path.join(self.test_dir, f"parler_test_{voice_name}.mp3")
                audio_segment.export(voice_output_path, format="mp3")
                
                # Verify the file exists and has content
                self.assertTrue(os.path.exists(voice_output_path))
                self.assertGreater(os.path.getsize(voice_output_path), 0)
                
                # Clean up the voice-specific output file
                if os.path.exists(voice_output_path):
                    os.remove(voice_output_path)
                
                print(f"Voice {voice_name} tested successfully")
            
            print("All voice descriptions tested successfully")
        except Exception as e:
            self.fail(f"Parler voice selection test failed with error: {str(e)}")

if __name__ == "__main__":
    unittest.main() 