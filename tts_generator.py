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
# Suppress HF text generation warnings
os.environ["HF_SUPPRESS_GENERATION_WARNINGS"] = "true"

import time
import yaml
import re
import shutil
from typing import Dict, List, Optional, Union, Tuple
from pydub import AudioSegment
import tempfile
import tqdm

class TTSGenerator:
    """Class for generating podcast audio from transcripts."""
    
    def __init__(self, model_type: str = "bark", config_file: str = 'config.yaml') -> None:
        """Initialize the TTS generator.
        
        Args:
            model_type: Type of TTS model to use ('bark', 'parler', or 'coqui')
            config_file: Path to configuration file
            
        Raises:
            ValueError: If model_type is not supported
        """
        self.model_type = model_type.lower()
        
        if self.model_type not in ["bark", "parler", "coqui"]:
            raise ValueError("Unsupported TTS model type. Choose 'bark', 'parler', or 'coqui'")
        
        # Check for FFmpeg dependencies
        self.ffmpeg_available = self._check_ffmpeg()
        if not self.ffmpeg_available:
            print("WARNING: FFmpeg/ffprobe not found. Audio export may fail.")
            print("Please install FFmpeg: https://ffmpeg.org/download.html")
        
        # Load configuration
        with open(config_file, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize model-specific components
        if self.model_type == "bark":
            self._init_bark()
        elif self.model_type == "parler":
            self._init_parler()
        else:  # coqui
            self._init_coqui()
        
        # Initialize execution time tracking
        self.execution_times = {
            'start_time': 0,
            'total_time': 0,
            'segments': []
        }
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg and ffprobe are available."""
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        return ffmpeg is not None and ffprobe is not None
    
    def _init_bark(self) -> None:
        """Initialize the Bark TTS model."""
        print("Initializing Bark TTS model...")
        from transformers import AutoProcessor, BarkModel
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        self.model = BarkModel.from_pretrained("suno/bark")
        
        # Set pad token ID to avoid warnings
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
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
        try:
            # Try both import paths for compatibility
            try:
                from parler_tts import ParlerTTS
            except ImportError:
                from parler.tts import ParlerTTS
            
            # Initialize Parler TTS
            self.model = ParlerTTS()
            
            # Define speaker presets (speaker IDs for Parler)
            self.speakers = {
                "Speaker 1": 0,  # Male expert
                "Speaker 2": 1,  # Female student
                "Speaker 3": 2   # Second expert
            }
            self.parler_available = True
        except ImportError:
            print("WARNING: Parler TTS module not found. Using fallback TTS instead.")
            print("To install Parler TTS, run: pip install git+https://github.com/huggingface/parler-tts.git")
            # Fall back to Bark if Parler is not available
            self.model_type = "bark"
            self._init_bark()
            self.parler_available = False
    
    def _init_coqui(self) -> None:
        """Initialize the Coqui TTS model."""
        print("Initializing Coqui TTS model...")
        try:
            from TTS.api import TTS
            
            # Initialize Coqui TTS with a multi-speaker model
            # Using VITS model which supports multi-speaker synthesis
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            
            # Define speaker presets (speaker names for Coqui XTTS)
            self.speakers = {
                "Speaker 1": "p326",  # Male expert
                "Speaker 2": "p225",  # Female student
                "Speaker 3": "p330"   # Second expert
            }
            self.coqui_available = True
            
            # Store sample rate for later use
            self.sample_rate = 24000  # Default for XTTS
            
        except ImportError:
            print("WARNING: Coqui TTS module not found. Using fallback TTS instead.")
            print("To install Coqui TTS, run: pip install TTS")
            # Fall back to Bark if Coqui is not available
            self.model_type = "bark"
            self._init_bark()
            self.coqui_available = False
        except Exception as e:
            print(f"WARNING: Error initializing Coqui TTS: {str(e)}. Using fallback TTS instead.")
            # Fall back to Bark if there's an error with Coqui
            self.model_type = "bark"
            self._init_bark()
            self.coqui_available = False
    
    def _generate_audio_bark(self, text: str, speaker: str) -> AudioSegment:
        """Generate audio using Bark TTS.
        
        Args:
            text: Text to convert to speech
            speaker: Speaker identifier
            
        Returns:
            AudioSegment containing the generated speech
        """
        try:
            # Prepare inputs
            inputs = self.processor(
                text=text,
                voice_preset=self.speakers[speaker],
                return_tensors="pt"
            )
            
            # Create attention mask if not present
            if "attention_mask" not in inputs:
                # Create attention mask (all 1s, same shape as input_ids)
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate audio with specific generation parameters
            generation_kwargs = {
                "pad_token_id": self.model.config.pad_token_id,
                "do_sample": True,
                "temperature": 0.7,
                "max_new_tokens": 250
            }
            
            # Make a clean copy of inputs without any generation parameters
            # to avoid conflicts with generation_kwargs
            model_inputs = {}
            for k, v in inputs.items():
                if k not in ["max_new_tokens", "do_sample", "temperature", "pad_token_id"]:
                    model_inputs[k] = v
            
            # Generate the audio
            speech_output = self.model.generate(**model_inputs, **generation_kwargs)
            
            # Convert to audio segment
            audio_array = speech_output.cpu().numpy().squeeze()
            
            # Save to temporary file and load as AudioSegment
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save as WAV
            import scipy.io.wavfile as wavfile
            wavfile.write(temp_path, rate=24000, data=audio_array)
            
            # Load as AudioSegment
            if not self.ffmpeg_available:
                print("WARNING: FFmpeg not available. Using silent audio as fallback.")
                audio_segment = AudioSegment.silent(duration=len(audio_array) * 1000 // 24000)
            else:
                try:
                    audio_segment = AudioSegment.from_wav(temp_path)
                except Exception as e:
                    print(f"Error loading audio segment: {str(e)}")
                    # Fallback to silent audio
                    audio_segment = AudioSegment.silent(duration=len(audio_array) * 1000 // 24000)
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_path}: {str(e)}")
            
            return audio_segment
        except Exception as e:
            print(f"Error in _generate_audio_bark: {str(e)}")
            # Return a silent segment as fallback
            return AudioSegment.silent(duration=1000)
    
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
    
    def _generate_audio_coqui(self, text: str, speaker: str) -> AudioSegment:
        """Generate audio using Coqui TTS.
        
        Args:
            text: Text to convert to speech
            speaker: Speaker identifier
            
        Returns:
            AudioSegment containing the generated speech
        """
        try:
            # Create a temporary file to save the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Generate audio with Coqui TTS
            # For XTTS, we need to provide a reference audio file for the speaker
            # Since we don't have that, we'll use the built-in speaker IDs
            self.model.tts_to_file(
                text=text,
                file_path=temp_path,
                speaker=self.speakers[speaker],
                language="en"
            )
            
            # Load as AudioSegment
            if not self.ffmpeg_available:
                print("WARNING: FFmpeg not available. Using silent audio as fallback.")
                # Estimate duration based on text length (rough approximation)
                estimated_duration = len(text) * 60  # ~60ms per character
                audio_segment = AudioSegment.silent(duration=estimated_duration)
            else:
                try:
                    audio_segment = AudioSegment.from_wav(temp_path)
                except Exception as e:
                    print(f"Error loading audio segment: {str(e)}")
                    # Fallback to silent audio
                    estimated_duration = len(text) * 60  # ~60ms per character
                    audio_segment = AudioSegment.silent(duration=estimated_duration)
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_path}: {str(e)}")
            
            return audio_segment
        except Exception as e:
            print(f"Error generating audio with Coqui: {str(e)}")
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
        
        try:
            # Check for FFmpeg
            if not self.ffmpeg_available:
                print("WARNING: FFmpeg/ffprobe not found. Audio export will likely fail.")
                print("Please install FFmpeg: https://ffmpeg.org/download.html")
                
                # Create a timestamp for the error file
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                error_path = f"./resources/error_missing_ffmpeg_{timestamp}.txt"
                
                # Create an error file with instructions
                with open(error_path, 'w', encoding='utf-8') as f:
                    f.write("Error: FFmpeg/ffprobe not found. Audio generation failed.\n\n")
                    f.write("Please install FFmpeg to generate audio:\n")
                    f.write("- Ubuntu/Debian: sudo apt-get install ffmpeg\n")
                    f.write("- CentOS/RHEL: sudo yum install ffmpeg\n")
                    f.write("- macOS: brew install ffmpeg\n")
                    f.write("- Windows: Download from https://ffmpeg.org/download.html\n")
                
                # Return the error file path
                return error_path
            
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
                    elif self.model_type == "parler":
                        chunk_audio = self._generate_audio_parler(chunk, speaker)
                    else:  # coqui
                        chunk_audio = self._generate_audio_coqui(chunk, speaker)
                    
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
            try:
                if not self.ffmpeg_available:
                    raise RuntimeError("FFmpeg/ffprobe not found. Cannot export audio.")
                
                full_audio.export(output_path, format="mp3")
            except Exception as e:
                print(f"Error exporting audio: {str(e)}")
                
                # Create a timestamp for error files
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                
                # Handle different error types
                if "Permission denied" in str(e):
                    # Try with a different filename if permission error
                    alt_output_path = f"./resources/podcast_alt_{timestamp}.mp3"
                    print(f"Trying alternative path: {alt_output_path}")
                    try:
                        full_audio.export(alt_output_path, format="mp3")
                        output_path = alt_output_path
                    except Exception as e2:
                        print(f"Error with alternative path: {str(e2)}")
                        # Create an error file instead
                        error_path = f"./resources/error_permission_{timestamp}.txt"
                        with open(error_path, 'w', encoding='utf-8') as f:
                            f.write(f"Error: Permission denied when exporting audio.\n\n")
                            f.write(f"Original error: {str(e)}\n")
                            f.write(f"Please check if the output directory is writable.")
                        output_path = error_path
                elif "ffprobe" in str(e) or "ffmpeg" in str(e):
                    # FFmpeg/ffprobe not found
                    error_path = f"./resources/error_missing_ffmpeg_{timestamp}.txt"
                    with open(error_path, 'w', encoding='utf-8') as f:
                        f.write("Error: FFmpeg/ffprobe not found. Audio generation failed.\n\n")
                        f.write("Please install FFmpeg to generate audio:\n")
                        f.write("- Ubuntu/Debian: sudo apt-get install ffmpeg\n")
                        f.write("- CentOS/RHEL: sudo yum install ffmpeg\n")
                        f.write("- macOS: brew install ffmpeg\n")
                        f.write("- Windows: Download from https://ffmpeg.org/download.html\n")
                    output_path = error_path
                else:
                    # Other errors
                    error_path = f"./resources/error_general_{timestamp}.txt"
                    with open(error_path, 'w', encoding='utf-8') as f:
                        f.write(f"Error exporting audio: {str(e)}\n\n")
                        f.write("Please check the logs for more details.")
                    output_path = error_path
            
            print(f"Podcast generated in {self.execution_times['total_time']:.2f} seconds")
            if isinstance(output_path, str) and output_path.endswith('.mp3'):
                print(f"Total audio duration: {len(full_audio)/1000:.2f} seconds")
            
            return output_path
            
        except Exception as e:
            print(f"Error in generate_podcast: {str(e)}")
            
            # Create a timestamp for error files
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            if "No module named 'parler'" in str(e):
                print("Parler TTS is not installed. Please install it with:")
                print("pip install git+https://github.com/huggingface/parler-tts.git")
                # Create an error file with instructions
                error_path = f"./resources/error_missing_parler_{timestamp}.txt"
                with open(error_path, 'w', encoding='utf-8') as f:
                    f.write("Error: Parler TTS module is not installed.\n\n")
                    f.write("Please install it with:\n")
                    f.write("pip install git+https://github.com/huggingface/parler-tts.git\n")
                return error_path
            elif "ffprobe" in str(e) or "ffmpeg" in str(e):
                # FFmpeg/ffprobe not found
                error_path = f"./resources/error_missing_ffmpeg_{timestamp}.txt"
                with open(error_path, 'w', encoding='utf-8') as f:
                    f.write("Error: FFmpeg/ffprobe not found. Audio generation failed.\n\n")
                    f.write("Please install FFmpeg to generate audio:\n")
                    f.write("- Ubuntu/Debian: sudo apt-get install ffmpeg\n")
                    f.write("- CentOS/RHEL: sudo yum install ffmpeg\n")
                    f.write("- macOS: brew install ffmpeg\n")
                    f.write("- Windows: Download from https://ffmpeg.org/download.html\n")
                return error_path
            else:
                # General error
                error_path = f"./resources/error_general_{timestamp}.txt"
                with open(error_path, 'w', encoding='utf-8') as f:
                    f.write(f"Error generating podcast: {str(e)}\n\n")
                    f.write("Please check the logs for more details.")
                return error_path
    
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