"""
Text-to-Speech Module for planeLLM.

This module converts podcast transcripts into audio using various TTS models.
It currently supports two TTS engines:
1. Bark (default): High-quality but slower
2. Parler: Faster but lower quality

The module handles speaker separation, audio segment generation, and final audio
compilation with comprehensive error handling and progress tracking.

Example:
    generator = TTSGenerator(model_type="bark")
    generator.generate_podcast("podcast_transcript.txt")

Classes:
    TTSGenerator: Main class for audio generation
"""

import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress specific PyTorch/transformers warnings
import torch
torch.set_warn_always(False)

from typing import Tuple, List, Optional, Dict
import argparse
from transformers import BarkModel, AutoProcessor, logging
logging.set_verbosity_error()  # Only show errors, not warnings
from parler_tts import ParlerTTSForConditionalGeneration
import numpy as np
from pydub import AudioSegment
import io
import os
from tqdm import tqdm
import ast

class TTSGenerator:
    """Class for generating audio from podcast transcripts using various TTS models."""
    
    def __init__(self, model_type: str = "bark") -> None:
        """Initialize TTSGenerator with specified model.
        
        Args:
            model_type: Type of TTS model to use ('bark' or 'parler')
            
        Raises:
            ValueError: If model_type is not supported
            RuntimeError: If model fails to load
        """
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            if model_type == "bark":
                self.model = BarkModel.from_pretrained("suno/bark")
                self.processor = AutoProcessor.from_pretrained("suno/bark")
                self.model.to(self.device)
            elif model_type == "parler":
                self.model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1")
                self.model.to(self.device)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_type} model: {str(e)}")
            
    def _numpy_to_audio_segment(self, audio_array: np.ndarray, sample_rate: int) -> AudioSegment:
        """Convert numpy array to AudioSegment.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            AudioSegment object
        """
        # Normalize audio
        audio_array = np.clip(audio_array, -1, 1)
        audio_array = (audio_array * 32767).astype(np.int16)
        
        # Convert to bytes
        byte_io = io.BytesIO()
        import wave
        with wave.open(byte_io, 'wb') as wave_file:
            wave_file.setnchannels(1)
            wave_file.setsampwidth(2)
            wave_file.setframerate(sample_rate)
            wave_file.writeframes(audio_array.tobytes())
        
        byte_io.seek(0)
        return AudioSegment.from_wav(byte_io)

    def generate_speaker1_audio(self, text: str) -> tuple:
        """Generate audio for speaker 1 (expert)."""
        if self.model_type == "bark":
            inputs = self.processor(
                text,
                voice_preset="v2/en_speaker_6",
                return_tensors="pt"
            ).to(self.device)
            audio_array = self.model.generate(**inputs)
            audio_array = audio_array.cpu().numpy().squeeze()
            return audio_array, 24000
        else:  # parler
            description = "A male expert speaking clearly and confidently"
            inputs = {"text": text, "description": description}
            audio = self.model.generate(**inputs)
            return audio.cpu().numpy().squeeze(), 24000

    def generate_speaker2_audio(self, text: str) -> tuple:
        """Generate audio for speaker 2 (student)."""
        if self.model_type == "bark":
            inputs = self.processor(
                text,
                voice_preset="v2/en_speaker_9",
                return_tensors="pt"
            ).to(self.device)
            audio_array = self.model.generate(**inputs)
            audio_array = audio_array.cpu().numpy().squeeze()
            return audio_array, 24000
        else:  # parler
            description = "A curious young student speaking with enthusiasm"
            inputs = {"text": text, "description": description}
            audio = self.model.generate(**inputs)
            return audio.cpu().numpy().squeeze(), 24000

    def generate_podcast(self, transcript_path: str) -> None:
        """Generate full podcast audio from transcript."""
        print("\nStarting podcast generation...")
        
        # Read the transcript
        print("Reading transcript file...")
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        except Exception as e:
            raise Exception(f"Error reading transcript file: {str(e)}")

        if not transcript:
            raise ValueError("Transcript file is empty")

        # Parse the transcript into speaker segments
        print("Parsing transcript into speaker segments...")
        segments = []
        current_speaker = None
        current_text = []
        
        # Split by lines and clean up
        lines = [line.strip() for line in transcript.split('\n') if line.strip()]
        
        for line in lines:
            # Check for speaker markers with more flexible pattern matching
            if 'Speaker 1:' in line or 'Speaker 2:' in line:
                # Save previous segment if exists
                if current_speaker and current_text:
                    segments.append((current_speaker, ' '.join(current_text)))
                
                # Start new segment
                current_speaker = 'Speaker 1' if 'Speaker 1:' in line else 'Speaker 2'
                current_text = [line.split(':', 1)[1].strip()]
            elif current_speaker:  # Continuation of current speaker's text
                current_text.append(line)
        
        # Add the last segment
        if current_speaker and current_text:
            segments.append((current_speaker, ' '.join(current_text)))
        
        if not segments:
            raise ValueError(
                "No valid segments found in transcript. "
                "Make sure the transcript contains 'Speaker 1:' or 'Speaker 2:' markers."
            )

        print(f"Found {len(segments)} segments to process")

        # Generate audio for each segment
        final_audio = None
        
        for i, (speaker, text) in enumerate(tqdm(segments, desc="Generating podcast segments", unit="segment")):
            print(f"\nProcessing segment {i+1}/{len(segments)} ({speaker})")
            print(f"Text length: {len(text)} characters")
            
            try:
                if speaker == "Speaker 1":
                    audio_arr, rate = self.generate_speaker1_audio(text)
                else:  # Speaker 2
                    audio_arr, rate = self.generate_speaker2_audio(text)
                
                # Convert to AudioSegment
                audio_segment = self._numpy_to_audio_segment(audio_arr, rate)
                
                # Add to final audio
                if final_audio is None:
                    final_audio = audio_segment
                else:
                    final_audio = final_audio + audio_segment
                    
                print(f"Successfully processed segment {i+1}")
            except Exception as e:
                print(f"Error processing segment {i+1}: {str(e)}")
                raise

        if final_audio is None:
            raise RuntimeError("Failed to generate any audio segments")

        # Ensure resources directory exists
        os.makedirs('./resources', exist_ok=True)
        
        print("\nExporting final podcast...")
        # Export the final podcast
        final_audio.export(
            "./resources/podcast.mp3",
            format="mp3",
            bitrate="192k",
            parameters=["-q:a", "0"]
        )
        print("Podcast exported successfully!")

def main():
    parser = argparse.ArgumentParser(description='Generate podcast audio from transcript')
    parser.add_argument('--model', type=str, choices=['bark', 'parler'], 
                      default='bark', help='TTS model to use (default: bark)')
    parser.add_argument('--transcript', type=str, 
                      default='./resources/podcast_transcript.txt',
                      help='Path to transcript file (default: ./resources/podcast_transcript.txt)')
    
    args = parser.parse_args()
    
    print(f"Initializing TTS Generator with {args.model} model...")
    generator = TTSGenerator(model_type=args.model)
    
    print(f"Generating podcast from transcript: {args.transcript}")
    generator.generate_podcast(args.transcript)
    
    print("Podcast generated and saved to ./resources/podcast.mp3")

if __name__ == "__main__":
    main() 