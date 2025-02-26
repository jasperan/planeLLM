#!/usr/bin/env python
"""
Controller script for the planeLLM podcast generation pipeline.

This script provides a unified interface to run the entire pipeline
from topic exploration to audio generation in a single command.

Usage:
    python podcast_controller.py --topic "Ancient Rome"
    python podcast_controller.py --topic "Quantum Physics" --tts-model parler
    python podcast_controller.py --topic "Machine Learning" --config my_config.yaml
"""

import os
import time
import argparse
from typing import Optional

# Import planeLLM components
from topic_explorer import TopicExplorer
from lesson_writer import PodcastWriter
from tts_generator import TTSGenerator

def main():
    """Run the planeLLM podcast generation pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate educational podcasts with planeLLM")
    parser.add_argument("--topic", type=str, required=True, help="Topic to explore")
    parser.add_argument("--tts-model", type=str, default="bark", choices=["bark", "parler"], 
                        help="TTS model to use (default: bark)")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--output", type=str, default=None, help="Path to output audio file")
    args = parser.parse_args()
    
    # Create resources directory if it doesn't exist
    os.makedirs('./resources', exist_ok=True)
    
    # Generate timestamp for file naming
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Generate educational content
    print(f"🔍 Exploring topic: {args.topic}")
    topic_explorer = TopicExplorer()
    
    # Generate questions
    print("Generating questions...")
    questions = topic_explorer.generate_questions(args.topic)
    
    # Save questions to file
    questions_file = f"./resources/questions_{timestamp}.txt"
    with open(questions_file, 'w', encoding='utf-8') as f:
        questions_text = f"# Questions for {args.topic}\n\n"
        for i, q in enumerate(questions, 1):
            questions_text += f"{i}. {q}\n"
        f.write(questions_text)
    print(f"✅ Questions saved to {questions_file}")
    
    # Generate content for each question
    print("Exploring questions...")
    results = {}
    for i, question in enumerate(questions):
        print(f"  Question {i+1}/{len(questions)}: {question}")
        response = topic_explorer.explore_question(question)
        results[question] = response
    
    # Combine content
    full_content = f"# {args.topic}\n\n"
    for question, response in results.items():
        full_content += f"{response}\n\n"
    
    # Save content to file
    content_file = f"./resources/raw_lesson_content_{timestamp}.txt"
    with open(content_file, 'w', encoding='utf-8') as f:
        f.write(full_content)
    print(f"✅ Content saved to {content_file}")
    
    # Step 2: Create podcast transcript
    print("\n📝 Creating podcast transcript...")
    podcast_writer = PodcastWriter()
    transcript = podcast_writer.create_podcast_transcript(full_content)
    
    # Save transcript to file
    transcript_file = f"./resources/podcast_transcript_{timestamp}.txt"
    with open(transcript_file, 'w', encoding='utf-8') as f:
        f.write(transcript)
    print(f"✅ Transcript saved to {transcript_file}")
    
    # Step 3: Generate podcast audio
    print(f"\n🔊 Generating podcast audio using {args.tts_model} model...")
    tts_generator = TTSGenerator(model_type=args.tts_model)
    
    # Set output path
    if args.output:
        audio_path = args.output
    else:
        audio_path = f"./resources/podcast_{timestamp}.mp3"
    
    # Generate podcast audio
    tts_generator.generate_podcast(transcript_file, output_path=audio_path)
    print(f"✅ Podcast audio saved to {audio_path}")
    
    print("\n🎉 Podcast generation complete!")
    print(f"Topic: {args.topic}")
    print(f"Questions: {questions_file}")
    print(f"Content: {content_file}")
    print(f"Transcript: {transcript_file}")
    print(f"Audio: {audio_path}")

if __name__ == "__main__":
    main() 