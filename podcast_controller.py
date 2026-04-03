#!/usr/bin/env python
"""CLI controller for the full planeLLM podcast generation pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

TopicExplorer = None
PodcastWriter = None
TTSGenerator = None
timestamp_slug = None

REQUIRED_CONFIG_KEYS = ("compartment_id", "config_profile", "model_id")


def ensure_runtime_config_ready(parser: argparse.ArgumentParser, config_file: str) -> None:
    from plane_llm_utils import load_yaml_config

    try:
        config_data = load_yaml_config(config_file)
    except FileNotFoundError:
        parser.exit(
            1,
            f"Error: Config file not found: {config_file}\n"
            f"Copy config_example.yaml to {config_file} and set compartment_id, config_profile, and model_id before running the pipeline.\n",
        )
    except ValueError as exc:
        parser.exit(1, f"Error: {exc}\n")

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if not str(config_data.get(key, "")).strip()]
    if missing_keys:
        parser.exit(1, f"Error: Missing required config values in {config_file}: {', '.join(missing_keys)}\n")

    placeholder_values = []
    for key in REQUIRED_CONFIG_KEYS:
        lowered = str(config_data.get(key, "")).strip().lower()
        if lowered in {"compartment_ocid", "profile_name", "model_ocid"} or "example" in lowered:
            placeholder_values.append(key)
    if placeholder_values:
        parser.exit(
            1,
            f"Error: Replace placeholder values in {config_file} before running the pipeline: {', '.join(placeholder_values)}\n",
        )


def main():
    global TopicExplorer, PodcastWriter, TTSGenerator, timestamp_slug

    parser = argparse.ArgumentParser(description="Generate an educational podcast on any topic")
    parser.add_argument("--topic", required=True, help="Topic to generate a podcast about")
    parser.add_argument(
        "--tts-model",
        default="fish",
        choices=["bark", "parler", "fish", "coqui"],
        help="TTS model to use (default: fish)",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file (default: config.yaml)")
    parser.add_argument("--output", help="Output path for the audio file")
    parser.add_argument(
        "--transcript-length",
        default="medium",
        choices=["short", "medium", "long"],
        help="Length of the podcast transcript (default: medium)",
    )
    parser.add_argument(
        "--detailed-transcript",
        action="store_true",
        help="Process each question individually for more detailed content",
    )
    args = parser.parse_args()

    if TopicExplorer is None:
        from topic_explorer import TopicExplorer as TopicExplorerClass
        TopicExplorer = TopicExplorerClass
    if PodcastWriter is None:
        from lesson_writer import PodcastWriter as PodcastWriterClass
        PodcastWriter = PodcastWriterClass
    if TTSGenerator is None:
        from tts_generator import TTSGenerator as TTSGeneratorClass
        TTSGenerator = TTSGeneratorClass
    if timestamp_slug is None:
        from plane_llm_utils import timestamp_slug as timestamp_slug_fn
        timestamp_slug = timestamp_slug_fn

    ensure_runtime_config_ready(parser, args.config)
    os.makedirs("./resources", exist_ok=True)

    print(f"\n=== Step 1: Exploring topic '{args.topic}' ===")
    explorer = TopicExplorer(config_file=args.config)
    questions = explorer.generate_questions(args.topic)
    timestamp = timestamp_slug()

    questions_file = Path("./resources") / f"questions_{timestamp}.txt"
    with questions_file.open("w", encoding="utf-8") as file:
        file.write("\n".join(questions))
    print(f"Questions saved to {questions_file}")

    print("\nGenerating educational content...")
    content_parts = []
    for index, question in enumerate(questions, start=1):
        print(f"Exploring question {index}/{len(questions)}: {question}")
        content_parts.append(f"# {question}\n\n{explorer.explore_question(question)}\n")
    content = "\n".join(content_parts)

    content_file = Path("./resources") / f"content_{timestamp}.txt"
    with content_file.open("w", encoding="utf-8") as file:
        file.write(content)
    print(f"Raw content saved to {content_file}")

    print("\n=== Step 2: Creating podcast transcript ===")
    writer = PodcastWriter(config_file=args.config, transcript_length=args.transcript_length)
    transcript = writer.create_detailed_podcast_transcript(content) if args.detailed_transcript else writer.create_podcast_transcript(content)

    print("\n=== Step 3: Generating podcast audio ===")
    tts = TTSGenerator(model_type=args.tts_model, config_file=args.config)
    output_path = args.output or f"./resources/podcast_{timestamp}.mp3"
    audio_path = tts.generate_podcast(transcript, output_path=output_path)

    print("\n=== Podcast Generation Complete ===")
    print(f"Questions: {questions_file}")
    print(f"Content: {content_file}")
    print(f"Audio: {audio_path}")
    print("\nThank you for using planeLLM!")


if __name__ == "__main__":
    main()
