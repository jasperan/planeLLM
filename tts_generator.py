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

# Specifically suppress the attention mask warnings
warnings.filterwarnings('ignore', message='.*The attention mask.*')
warnings.filterwarnings('ignore', message='.*The pad token id is not set.*')
warnings.filterwarnings('ignore', message='.*You have modified the pretrained model configuration.*')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import torch
# Suppress Flash Attention 2 warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
# Suppress HF text generation warnings
os.environ["HF_SUPPRESS_GENERATION_WARNINGS"] = "true"
# Additional environment variables to suppress warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import yaml
import re
import shutil
from typing import Dict, List, Optional, Union, Tuple
from pydub import AudioSegment
import tempfile
import tqdm
import numpy as np

# Disable logging from transformers
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)

class TTSGenerator:
    """Class for generating podcast audio from transcripts."""
    
    def __init__(self, model_type: str = "parler", config_file: str = 'config.yaml') -> None:
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
        
        # Define speaker presets - using consistent voice presets for better voice consistency
        self.speakers = {
            "Speaker 1": "v2/en_speaker_6",  # Male expert
            "Speaker 2": "v2/en_speaker_9",  # Female student
            "Speaker 3": "v2/en_speaker_3"   # Second expert
        }
        
        # Store the voice consistency settings
        self.generation_params = {
            "temperature": 0.01,  # Very low temperature for consistent voices
            "do_sample": False    # Deterministic generation for maximum consistency
        }
    
    def _init_parler(self) -> None:
        """Initialize the Parler TTS model."""
        print("Initializing Parler TTS model...")
        try:
            # Try both import paths for compatibility
            try:
                from parler_tts import ParlerTTSForConditionalGeneration
                from transformers import AutoTokenizer
            except ImportError:
                from parler.tts import ParlerTTSForConditionalGeneration, AutoTokenizer
            
            # Initialize Parler TTS
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.parler_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
            self.parler_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
            
            # Define available Parler voices
            self.available_voices = [
                "Laura", "Gary", "Jon", "Lea", "Karen", "Rick", "Brenda", "David", 
                "Eileen", "Jordan", "Mike", "Yann", "Joy", "James", "Eric", "Lauren", 
                "Rose", "Will", "Jason", "Aaron", "Naomie", "Alisa", "Patrick", "Jerry", 
                "Tina", "Jenna", "Bill", "Tom", "Carol", "Barbara", "Rebecca", "Anna", 
                "Bruce", "Emily"
            ]
            
            # Define voice descriptions for different speaker types
            self.voice_descriptions = {
                # Male voices with different styles
                "male_clear": "Jon's voice is clear and professional with a moderate pace, with very clear audio that has no background noise.",
                "male_expressive": "Gary's voice is expressive and animated with varied intonation, with very clear audio that has no background noise.",
                "male_deep": "Bruce's voice is deep and authoritative with a measured pace, with very clear audio that has no background noise.",
                "male_casual": "Rick's voice is casual and conversational with a natural flow, with very clear audio that has no background noise.",
                
                # Female voices with different styles
                "female_clear": "Laura's voice is clear and professional with a moderate pace, with very clear audio that has no background noise.",
                "female_expressive": "Jenna's voice is expressive and animated with varied intonation, with very clear audio that has no background noise.",
                "female_warm": "Rose's voice is warm and engaging with a pleasant tone, with very clear audio that has no background noise.",
                "female_casual": "Lea's voice is casual and conversational with a natural flow, with very clear audio that has no background noise."
            }
            
            # Map speakers to voice descriptions
            self.speaker_voice_map = {
                "Speaker 1": "male_clear",  # Default for Speaker 1 (expert)
                "Speaker 2": "female_expressive",  # Default for Speaker 2 (student)
                "Speaker 3": "male_expressive"   # Default for Speaker 3 (second expert)
            }
            
            # Store the sample rate for later use
            self.sample_rate = 24000  # Default for Parler TTS
            
            self.parler_available = True
            print(f"Parler TTS initialized successfully on {device}")
            
        except ImportError:
            print("WARNING: Parler TTS module not found. Using fallback TTS instead.")
            print("To install Parler TTS, run: pip install git+https://github.com/huggingface/parler-tts.git")
            # Fall back to Bark if Parler is not available
            self.model_type = "bark"
            self._init_bark()
            self.parler_available = False
        except Exception as e:
            print(f"WARNING: Error initializing Parler TTS: {str(e)}. Using fallback TTS instead.")
            # Fall back to Bark if there's an error with Parler
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
                "do_sample": False,  # Deterministic generation (no sampling) for maximum voice consistency
                "temperature": 0.01   # Very low temperature (near 0) for consistent voice characteristics
                # Lower temperature = more consistent voice
                # Higher temperature = more varied, creative voice
            }
            
            # IMPORTANT: Check if max_new_tokens is already in the inputs
            # If it is, we need to remove it to avoid the conflict
            if "max_new_tokens" in inputs:
                del inputs["max_new_tokens"]
            else:
                # Add max_new_tokens only if not already in inputs
                generation_kwargs["max_new_tokens"] = 250
            
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
            # Get the voice description for this speaker
            voice_type = self.speaker_voice_map.get(speaker, "male_clear")
            description = self.voice_descriptions.get(voice_type, self.voice_descriptions["male_clear"])
            
            # Process text to remove any problematic characters or patterns that might cause silent audio
            # Remove any excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove any control characters
            text = re.sub(r'[\x00-\x1F\x7F]', '', text)
            
            # Replace problematic apostrophes with standard ones
            text = text.replace("'", "'").replace("'", "'")
            
            # Check if text is empty after processing
            if not text:
                print(f"Warning: Empty text after processing for {speaker}")
                return AudioSegment.silent(duration=500)  # Return a short silence
                
            print(f"Generating audio with Parler for '{text[:50]}...' with voice: {voice_type}")
            
            # Try a much simpler voice description that's more likely to work
            simple_description = f"A {voice_type.split('_')[0]} voice with clear audio"
            
            # Prepare inputs for Parler TTS
            input_ids = self.parler_tokenizer(simple_description, return_tensors="pt").input_ids
            prompt_input_ids = self.parler_tokenizer(text, return_tensors="pt").input_ids
            
            # Move to the same device as the model
            device = next(self.parler_model.parameters()).device
            input_ids = input_ids.to(device)
            prompt_input_ids = prompt_input_ids.to(device)
            
            # Fix for tensor size mismatch error
            # The error "The size of tensor a (20) must match the size of tensor b (21) at non-singleton dimension 1"
            # occurs when the model expects inputs of the same sequence length
            
            try:
                # Method 1: Use the tokenizer's padding feature to ensure same sequence length
                # Re-tokenize with padding to ensure same sequence length
                tokenizer_output = self.parler_tokenizer(
                    [simple_description, text],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=128  # Set a reasonable max length
                )
                
                # Extract the padded input_ids for both sequences
                padded_input_ids = tokenizer_output.input_ids
                description_input_ids = padded_input_ids[0:1]  # First sequence
                text_input_ids = padded_input_ids[1:2]  # Second sequence
                
                # Move to device
                description_input_ids = description_input_ids.to(device)
                text_input_ids = text_input_ids.to(device)
                
                # Generate audio
                generation = self.parler_model.generate(
                    input_ids=description_input_ids,
                    prompt_input_ids=text_input_ids,
                    do_sample=True,  # Use sampling for more natural speech
                    temperature=0.3  # Lower temperature for more consistent voice
                )
                
                # Convert to numpy array
                audio_array = generation.cpu().numpy().squeeze()
            except Exception as inner_e:
                # Method 2: Fallback to using the model directly with the text
                print(f"Warning: First generation method failed: {str(inner_e)}")
                print("Trying alternative generation method...")
                
                # Try a different approach - use a very simple description
                try:
                    # Last resort: Use an even simpler voice description
                    very_simple_description = "A clear voice"
                    simple_input_ids = self.parler_tokenizer(very_simple_description, return_tensors="pt").input_ids.to(device)
                    
                    # Try with simplified description
                    generation = self.parler_model.generate(
                        input_ids=simple_input_ids,
                        prompt_input_ids=prompt_input_ids,
                        do_sample=True,  # Use sampling as a fallback
                        temperature=0.5  # Moderate temperature
                    )
                    audio_array = generation.cpu().numpy().squeeze()
                except Exception as final_e:
                    print(f"Error: All generation methods failed. Last error: {str(final_e)}")
                    # Create a beep sound to indicate the error
                    sample_rate = self.sample_rate
                    duration_ms = 500
                    t = np.linspace(0, duration_ms/1000, int(sample_rate * duration_ms/1000), False)
                    audio_array = np.sin(2 * np.pi * 440 * t) * 0.3  # 440 Hz sine wave
            
            # Check if audio array contains only zeros or very low values (silent audio)
            if np.max(np.abs(audio_array)) < 0.01:
                print(f"Warning: Generated audio for '{text[:30]}...' appears to be silent. Retrying...")
                
                # Retry with a completely different approach
                try:
                    # Try with a different voice type entirely
                    if "male" in voice_type:
                        alt_voice_type = "female_clear"
                    else:
                        alt_voice_type = "male_clear"
                        
                    alt_description = f"A {alt_voice_type.split('_')[0]} voice with very clear audio"
                    alt_input_ids = self.parler_tokenizer(alt_description, return_tensors="pt").input_ids.to(device)
                    
                    # Re-tokenize the text to ensure it's clean
                    clean_text = re.sub(r'[^\w\s.,?!;:\-\'"]', '', text)  # Remove any special characters
                    clean_prompt_input_ids = self.parler_tokenizer(clean_text, return_tensors="pt").input_ids.to(device)
                    
                    generation = self.parler_model.generate(
                        input_ids=alt_input_ids,
                        prompt_input_ids=clean_prompt_input_ids,
                        do_sample=True,  # Use sampling for retry
                        temperature=0.7  # Higher temperature for more variation
                    )
                    audio_array = generation.cpu().numpy().squeeze()
                except Exception as retry_e:
                    print(f"Warning: Retry failed: {str(retry_e)}")
                
                # If still silent, create a speech-like sound instead of just a beep
                if np.max(np.abs(audio_array)) < 0.01:
                    print(f"Warning: Still generated silent audio for '{text[:30]}...'")
                    # Create a more complex sound that resembles speech
                    sample_rate = self.sample_rate
                    duration_sec = len(text) * 0.05  # Roughly 50ms per character
                    if duration_sec < 1.0:
                        duration_sec = 1.0  # Minimum 1 second
                    
                    # Create a time array
                    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
                    
                    # Create a complex waveform that sounds more like speech
                    # Mix several frequencies with amplitude modulation
                    audio_array = (
                        0.3 * np.sin(2 * np.pi * 150 * t) +  # Base frequency
                        0.2 * np.sin(2 * np.pi * 300 * t) +  # First harmonic
                        0.1 * np.sin(2 * np.pi * 450 * t) +  # Second harmonic
                        0.05 * np.sin(2 * np.pi * 600 * t)   # Third harmonic
                    )
                    
                    # Add amplitude modulation to simulate speech patterns
                    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
                    audio_array = audio_array * modulation
                    
                    # Normalize
                    audio_array = audio_array / np.max(np.abs(audio_array)) * 0.7
            
            # Save to temporary file and load as AudioSegment
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save as WAV
            import scipy.io.wavfile as wavfile
            wavfile.write(temp_path, rate=self.sample_rate, data=audio_array)
            
            # Load as AudioSegment
            audio_segment = AudioSegment.from_wav(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Trim silence at the beginning and end
            audio_segment = audio_segment.strip_silence(silence_thresh=-50, padding=100)
            
            return audio_segment
        except Exception as e:
            print(f"Error generating audio with Parler: {str(e)}")
            # Create a speech-like sound instead of just a beep
            sample_rate = self.sample_rate if hasattr(self, 'sample_rate') else 24000
            duration_sec = len(text) * 0.05 if 'text' in locals() else 2.0  # Roughly 50ms per character or 2 seconds
            if duration_sec < 1.0:
                duration_sec = 1.0  # Minimum 1 second
            
            # Create a time array
            t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
            
            # Create a complex waveform that sounds more like speech
            audio_array = (
                0.3 * np.sin(2 * np.pi * 150 * t) +  # Base frequency
                0.2 * np.sin(2 * np.pi * 300 * t) +  # First harmonic
                0.1 * np.sin(2 * np.pi * 450 * t)    # Second harmonic
            )
            
            # Add amplitude modulation
            modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
            audio_array = audio_array * modulation
            
            # Normalize
            audio_array = audio_array / np.max(np.abs(audio_array)) * 0.7
            
            # Save to temporary file and load as AudioSegment
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save as WAV
            import scipy.io.wavfile as wavfile
            wavfile.write(temp_path, rate=sample_rate, data=audio_array)
            
            # Load as AudioSegment
            audio_segment = AudioSegment.from_wav(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return audio_segment
    
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
        
        # Process natural conversation elements
        processed_segments = []
        for speaker, text in segments:
            # Replace conversation elements with appropriate pauses or silence
            # [laughs], [sighs], [pauses] etc. will be replaced with short pauses
            processed_text = re.sub(r'\[(laughs|chuckles)\]', ' ', text)
            processed_text = re.sub(r'\[(sighs|pauses|hmm)\]', ' ', text)
            processed_text = re.sub(r'\[.*?\]', ' ', processed_text)  # Remove any other bracketed elements
            
            # Clean up extra spaces
            processed_text = re.sub(r'\s+', ' ', processed_text).strip()
            
            if processed_text:  # Only add if there's text left after processing
                processed_segments.append((speaker, processed_text))
        
        return processed_segments
    
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
            
            # Store speaker voice assignments to ensure consistency
            speaker_voices = {}
            
            # Generate audio for each segment
            full_audio = AudioSegment.empty()
            
            for i, (speaker, text) in enumerate(tqdm.tqdm(segments, desc="Generating audio segments")):
                start_time = time.time()
                
                print(f"\nProcessing segment {i+1}/{len(segments)}: {speaker} ({len(text)} chars)")
                
                # Ensure consistent voice for each speaker
                if speaker not in speaker_voices:
                    # First time seeing this speaker, assign a voice
                    if self.model_type == "bark":
                        voice = self.speakers.get(speaker, 'default')
                    elif self.model_type == "parler":
                        voice_type = self.speaker_voice_map.get(speaker, "male_clear")
                        voice = voice_type
                    else:  # coqui
                        voice = self.speakers.get(speaker, 'default')
                    
                    speaker_voices[speaker] = voice
                    print(f"  Assigned voice for {speaker}: {voice}")
                else:
                    # Use the previously assigned voice
                    voice = speaker_voices[speaker]
                    print(f"  Using consistent voice for {speaker}: {voice}")
                
                # For Parler, we don't need to chunk the text as it can handle longer inputs
                if self.model_type == "parler":
                    print(f"  Generating audio for entire segment ({len(text)} chars)")
                    segment_audio = self._generate_audio_parler(text, speaker)
                else:
                    # Split long text into smaller chunks (max 150 chars) for other models
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
                    
                    # Generate audio for each chunk - all chunks use the same speaker voice
                    segment_audio = AudioSegment.empty()
                    
                    # Process natural conversation elements like [laughs] with appropriate pauses
                    for j, chunk in enumerate(chunks):
                        print(f"  Chunk {j+1}/{len(chunks)}: {len(chunk)} chars")
                        
                        # Check for natural conversation elements and add appropriate pauses
                        has_laugh = '[laughs]' in chunk or '[chuckles]' in chunk
                        has_pause = '[pauses]' in chunk or '[sighs]' in chunk
                        
                        # Generate audio based on model type
                        if self.model_type == "bark":
                            chunk_audio = self._generate_audio_bark(chunk, speaker)
                        else:  # coqui
                            chunk_audio = self._generate_audio_coqui(chunk, speaker)
                        
                        # Add appropriate pauses for natural elements
                        if has_laugh:
                            # Add a short laugh pause (300ms)
                            laugh_pause = AudioSegment.silent(duration=300)
                            chunk_audio = chunk_audio + laugh_pause
                        
                        if has_pause:
                            # Add a thoughtful pause (500ms)
                            thoughtful_pause = AudioSegment.silent(duration=500)
                            chunk_audio = chunk_audio + thoughtful_pause
                        
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
            
            # Ensure output_path is a string and not a directory
            if isinstance(output_path, str):
                if os.path.isdir(output_path):
                    # If output_path is a directory, create a file inside it
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(output_path, f"podcast_{timestamp}.mp3")
            else:
                # If output_path is not a string, create a default path
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
                elif "Is a directory" in str(e):
                    # Handle directory error
                    error_path = f"./resources/error_directory_{timestamp}.txt"
                    with open(error_path, 'w', encoding='utf-8') as f:
                        f.write(f"Error: Output path is a directory, not a file.\n\n")
                        f.write(f"Original error: {str(e)}\n")
                        f.write(f"Please provide a valid file path for the output.")
                    output_path = error_path
                elif "Size should be 1, 2, 3 or 4" in str(e):
                    # Handle the specific size error
                    error_path = f"./resources/error_size_{timestamp}.txt"
                    with open(error_path, 'w', encoding='utf-8') as f:
                        f.write(f"Error: Size should be 1, 2, 3 or 4.\n\n")
                        f.write(f"This is likely due to an issue with the audio generation.\n")
                        f.write(f"Please try again with a different voice or model.")
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
            elif "Is a directory" in str(e):
                # Handle directory error
                error_path = f"./resources/error_directory_{timestamp}.txt"
                with open(error_path, 'w', encoding='utf-8') as f:
                    f.write(f"Error: Output path is a directory, not a file.\n\n")
                    f.write(f"Original error: {str(e)}\n")
                    f.write(f"Please provide a valid file path for the output.")
                return error_path
            elif "Size should be 1, 2, 3 or 4" in str(e):
                # Handle the specific size error
                error_path = f"./resources/error_size_{timestamp}.txt"
                with open(error_path, 'w', encoding='utf-8') as f:
                    f.write(f"Error: Size should be 1, 2, 3 or 4.\n\n")
                    f.write(f"This is likely due to an issue with the audio generation.\n")
                    f.write(f"Please try again with a different voice or model.")
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
    
    def set_voice_description(self, speaker: str, description: str) -> None:
        """Set a custom voice description for a speaker.
        
        Args:
            speaker: The speaker identifier (e.g., "Speaker 1")
            description: The voice description to use for Parler TTS
        """
        if self.model_type != "parler" or not self.parler_available:
            print("Warning: Voice descriptions are only supported with Parler TTS")
            return
            
        # Store the custom description directly
        self.voice_descriptions[f"custom_{speaker}"] = description
        self.speaker_voice_map[speaker] = f"custom_{speaker}"
        print(f"Set custom voice description for {speaker}")
    
    def set_voice_type(self, speaker: str, voice_type: str) -> None:
        """Set the voice type for a speaker.
        
        Args:
            speaker: The speaker identifier (e.g., "Speaker 1")
            voice_type: The voice type to use (e.g., "male_clear", "female_expressive")
        """
        if self.model_type != "parler" or not self.parler_available:
            print("Warning: Voice types are only supported with Parler TTS")
            return
            
        if voice_type in self.voice_descriptions:
            self.speaker_voice_map[speaker] = voice_type
            print(f"Set voice type for {speaker} to {voice_type}")
        else:
            print(f"Warning: Voice type '{voice_type}' not found. Using default.") 