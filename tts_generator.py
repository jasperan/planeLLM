#!/usr/bin/env python
"""
TTS Generator Module for planeLLM.

This module handles the conversion of podcast transcripts into audio using
various TTS models. It supports multiple TTS engines and handles speaker separation.

Example:
    generator = TTSGenerator(model_type="bark")
    generator.generate_podcast("podcast_transcript.txt")
    
    # Or directly from transcript text:
    generator.generate_podcast(transcript_text, output_path="podcast.mp3")
"""

import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

import os
import torch
# Suppress Flash Attention 2 warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

import time
import yaml
import re
from typing import Dict, List, Optional, Union, Tuple
from pydub import AudioSegment
import tempfile
import tqdm

class TTSGenerator:
    """Class for generating podcast audio from transcripts."""
    
    def __init__(self, model_type: str = "bark", config_file: str = 'config.yaml') -> None:
        """Initialize the TTS generator.
        
        Args:
            model_type: Type of TTS model to use ('bark' or 'parler')
            config_file: Path to configuration file
            
        Raises:
            ValueError: If model_type is not supported
        """
        self.model_type = model_type.lower()
        
        if self.model_type not in ["bark", "parler"]:
            raise ValueError("Unsupported TTS model type. Choose 'bark' or 'parler'")
        
        # Load configuration
        with open(config_file, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize model-specific components
        if self.model_type == "bark":
            self._init_bark()
        else:  # parler
            self._init_parler()
        
        # Initialize execution time tracking
        self.execution_times = {
            'start_time': 0,
            'total_time': 0,
            'segments': []
        }
    
    def _init_bark(self) -> None:
        """Initialize the Bark TTS model."""
        print("Initializing Bark TTS model...")
        from transformers import AutoProcessor, BarkModel
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        self.model = BarkModel.from_pretrained("suno/bark")
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("Bark model loaded on GPU")
        else:
            print("Bark model loaded on CPU")
        
        # Define speaker presets
        self.speakers = {
            "Speaker 1": "v2/en_speaker_6",  # Male expert
            "Speaker 2": "v2/en_speaker_9",  # Female student
            "Speaker 3": "v2/en_speaker_3"   # Second expert
        }
    
    def _init_parler(self) -> None:
        """Initialize the Parler TTS model."""
        print("Initializing Parler TTS model...")
        from parler.tts import ParlerTTS
        
        # Initialize Parler TTS
        self.model = ParlerTTS()
        
        # Define speaker presets (speaker IDs for Parler)
        self.speakers = {
            "Speaker 1": 0,  # Male expert
            "Speaker 2": 1,  # Female student
            "Speaker 3": 2   # Second expert
        }
    
    def _generate_audio_bark(self, text: str, speaker: str) -> AudioSegment:
        """Generate audio using Bark TTS.
        
        Args:
            text: Text to convert to speech
            speaker: Speaker identifier
            
        Returns:
            AudioSegment containing the generated speech
        """
        # Prepare inputs
        inputs = self.processor(
            text=text,
            voice_preset=self.speakers[speaker],
            return_tensors="pt"
        )
        
        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate audio
        speech_output = self.model.generate(**inputs)
        
        # Convert to audio segment
        audio_array = speech_output.cpu().numpy().squeeze()
        
        # Save to temporary file and load as AudioSegment
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Save as WAV
        import scipy.io.wavfile as wavfile
        wavfile.write(temp_path, rate=24000, data=audio_array)
        
        # Load as AudioSegment
        audio_segment = AudioSegment.from_wav(temp_path)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return audio_segment
    
    def _generate_audio_parler(self, text: str, speaker: str) -> AudioSegment:
        """Generate audio using Parler TTS.
        
        Args:
            text: Text to convert to speech
            speaker: Speaker identifier
            
        Returns:
            AudioSegment containing the generated speech
        """
        try:
            # Generate audio
            audio_array = self.model.synthesize(
                text=text,
                speaker_id=self.speakers[speaker],
                temperature=0.7
            )
            
            # Save to temporary file and load as AudioSegment
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save as WAV
            import scipy.io.wavfile as wavfile
            wavfile.write(temp_path, rate=24000, data=audio_array)
            
            # Load as AudioSegment
            audio_segment = AudioSegment.from_wav(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return audio_segment
        except Exception as e:
            print(f"Error generating audio with Parler: {str(e)}")
            # Return a silent segment as fallback
            return AudioSegment.silent(duration=1000)
    
    def _parse_transcript(self, transcript: str) -> List[Tuple[str, str]]:
        """Parse the transcript into speaker segments.
        
        Args:
            transcript: The podcast transcript
            
        Returns:
            List of (speaker, text) tuples
        """
        # Split transcript into lines
        lines = transcript.strip().split('\n')
        
        segments = []
        current_speaker = None
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for speaker label
            speaker_match = re.match(r'^(Speaker \d+):\s*(.*)', line)
            if speaker_match:
                # If we have accumulated text for a previous speaker, add it
                if current_speaker and current_text:
                    segments.append((current_speaker, ' '.join(current_text)))
                
                # Start new speaker segment
                current_speaker = speaker_match.group(1)
                current_text = [speaker_match.group(2)] if speaker_match.group(2) else []
            else:
                # Continue with current speaker
                if current_speaker:
                    current_text.append(line)
        
        # Add the last segment
        if current_speaker and current_text:
            segments.append((current_speaker, ' '.join(current_text)))
        
        return segments
    
    def generate_podcast(self, transcript: Union[str, os.PathLike], output_path: Optional[str] = None) -> str:
        """Generate podcast audio from transcript.
        
        Args:
            transcript: Either a path to a transcript file or the transcript text
            output_path: Path to save the generated audio (default: auto-generated)
            
        Returns:
            Path to the generated audio file
        """
        print(f"\nGenerating podcast audio using {self.model_type.upper()} model...")
        
        self.execution_times['start_time'] = time.time()
        
        # Determine if transcript is a file path or text
        if isinstance(transcript, str) and os.path.exists(transcript) and os.path.isfile(transcript):
            # It's a file path
            print(f"Reading transcript from file: {transcript}")
            with open(transcript, 'r', encoding='utf-8') as file:
                transcript_text = file.read()
        else:
            # It's the transcript text
            print("Using provided transcript text")
            transcript_text = transcript
        
        # Parse transcript into speaker segments
        segments = self._parse_transcript(transcript_text)
        print(f"Parsed {len(segments)} speaker segments")
        
        # Generate audio for each segment
        full_audio = AudioSegment.empty()
        
        for i, (speaker, text) in enumerate(tqdm.tqdm(segments, desc="Generating audio segments")):
            start_time = time.time()
            
            print(f"\nProcessing segment {i+1}/{len(segments)}: {speaker} ({len(text)} chars)")
            
            # Split long text into smaller chunks (max 200 chars)
            chunks = []
            max_chunk_size = 150  # Reduced from 200 to ensure better handling
            
            # Improved chunking by sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                # Skip empty sentences
                if not sentence.strip():
                    continue
                    
                # If this sentence alone is longer than max_chunk_size, split it further
                if len(sentence) > max_chunk_size:
                    # Split by commas or other natural pauses
                    sub_parts = re.split(r'(?<=[,;:])\s+', sentence)
                    for part in sub_parts:
                        if len(part) > max_chunk_size:
                            # If still too long, just add it as is - TTS will have to handle it
                            chunks.append(part)
                        elif current_length + len(part) > max_chunk_size and current_chunk:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = [part]
                            current_length = len(part)
                        else:
                            current_chunk.append(part)
                            current_length += len(part)
                elif current_length + len(sentence) > max_chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence)
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Generate audio for each chunk
            segment_audio = AudioSegment.empty()
            for j, chunk in enumerate(chunks):
                print(f"  Chunk {j+1}/{len(chunks)}: {len(chunk)} chars")
                
                # Generate audio based on model type
                if self.model_type == "bark":
                    chunk_audio = self._generate_audio_bark(chunk, speaker)
                else:  # parler
                    chunk_audio = self._generate_audio_parler(chunk, speaker)
                
                segment_audio += chunk_audio
            
            # Add a short pause between speakers (500ms)
            pause = AudioSegment.silent(duration=500)
            full_audio += segment_audio + pause
            
            # Track execution time
            duration = time.time() - start_time
            self.execution_times['segments'].append({
                'speaker': speaker,
                'text_length': len(text),
                'duration': duration
            })
        
        # Calculate total execution time
        self.execution_times['total_time'] = time.time() - self.execution_times['start_time']
        
        # Generate output path if not provided
        if not output_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"./resources/podcast_{timestamp}.mp3"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Export as MP3
        print(f"\nExporting podcast to {output_path}...")
        full_audio.export(output_path, format="mp3")
        
        print(f"Podcast generated in {self.execution_times['total_time']:.2f} seconds")
        print(f"Total audio duration: {len(full_audio)/1000:.2f} seconds")
        
        return output_path
    
    def _generate_timing_summary(self) -> str:
        """Generate a summary of execution times."""
        summary = ["=== Execution Time Summary ==="]
        
        # Get segment stats
        total_segments = len(self.execution_times['segments'])
        total_text_length = sum(segment['text_length'] for segment in self.execution_times['segments'])
        total_segment_time = sum(segment['duration'] for segment in self.execution_times['segments'])
        
        summary.append("\nSegment Statistics:")
        summary.append(f"  - Total Segments: {total_segments}")
        summary.append(f"  - Total Text Length: {total_text_length} characters")
        summary.append(f"  - Average Segment Length: {total_text_length/total_segments:.2f} characters")
        summary.append(f"  - Total Segment Processing Time: {total_segment_time:.2f} seconds")
        summary.append(f"  - Average Time per Segment: {total_segment_time/total_segments:.2f} seconds")
        summary.append(f"Total Execution Time: {self.execution_times['total_time']:.2f} seconds")
        
        return "\n".join(summary) 