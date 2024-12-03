import argparse
from transformers import BarkModel, AutoProcessor
from parler_tts import ParlerTTSForConditionalGeneration
import torch
import numpy as np
from pydub import AudioSegment
import io
import os
from tqdm import tqdm
import ast

class TTSGenerator:
    def __init__(self, model_type: str = "bark"):
        """Initialize TTS generator with specified model type."""
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_type == "bark":
            self.model = BarkModel.from_pretrained("suno/bark")
            self.processor = AutoProcessor.from_pretrained("suno/bark")
            self.model.to(self.device)
        elif model_type == "parler":
            self.model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1")
            self.model.to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _numpy_to_audio_segment(self, audio_array: np.ndarray, sample_rate: int) -> AudioSegment:
        """Convert numpy array to AudioSegment."""
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
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()

        # Parse the transcript into speaker segments
        print("Parsing transcript into speaker segments...")
        segments = []
        current_speaker = None
        current_text = []
        
        for line in transcript.split('\n'):
            if line.startswith('Speaker 1:') or line.startswith('Speaker 2:'):
                if current_speaker and current_text:  # Only add if there's content
                    segments.append((current_speaker, ' '.join(current_text)))
                current_speaker = line[:9]
                current_text = [line[10:].strip()]
            elif line.strip():
                current_text.append(line.strip())
        
        # Add the last segment if it exists
        if current_speaker and current_text:  # Only add if there's content
            segments.append((current_speaker, ' '.join(current_text)))
        
        if not segments:
            raise ValueError("No valid segments found in transcript. Check transcript format.")

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