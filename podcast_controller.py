import os
from typing import Optional
from generate_lesson_content import generate_lesson_content
from lesson_writer import PodcastWriter
from tts import TTSGenerator

class PodcastController:
    def __init__(self, config_file: str = 'config.yaml'):
        """Initialize the podcast controller with configuration."""
        self.config_file = config_file
        self._ensure_resources_dir()
        
    def _ensure_resources_dir(self):
        """Ensure the resources directory exists."""
        os.makedirs('./resources', exist_ok=True)
        
    def generate_podcast(self, topic: str, tts_model: str = 'bark') -> None:
        """
        Generate a complete podcast from a topic, handling all three steps:
        1. Generate lesson content
        2. Create podcast transcript
        3. Convert transcript to speech
        
        Args:
            topic (str): The topic to generate content about
            tts_model (str, optional): The TTS model to use ('bark' or 'parler'). Defaults to 'bark'.
        """
        print("Step 1: Generating lesson content...")
        lesson_content = generate_lesson_content(topic)
        print("✓ Lesson content generated\n")
        
        print("Step 2: Creating podcast transcript...")
        writer = PodcastWriter(config_file=self.config_file)
        transcript = writer.create_podcast_transcript(lesson_content)
        print("✓ Podcast transcript created\n")
        
        print("Step 3: Generating audio...")
        tts = TTSGenerator(model=tts_model)
        tts.generate_podcast('./resources/podcast_transcript.txt')
        print("✓ Audio generated\n")
        
        print("Podcast generation complete! Files created:")
        print("- ./resources/raw_lesson_content.txt")
        print("- ./resources/podcast_transcript.txt")
        print("- ./resources/podcast.mp3")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate a podcast from a topic')
    parser.add_argument('--topic', type=str, required=True, help='Topic to generate content about')
    parser.add_argument('--model', type=str, default='bark', choices=['bark', 'parler'], 
                      help='TTS model to use (default: bark)')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to config file (default: config.yaml)')
    
    args = parser.parse_args()
    
    controller = PodcastController(config_file=args.config)
    controller.generate_podcast(args.topic, args.model) 