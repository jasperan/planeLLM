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
create_demo_bundle = None
build_runtime_preflight = None
resolve_runtime_config_data = None


def ensure_runtime_config_ready(parser: argparse.ArgumentParser, config_file: str) -> dict:
    preflight = build_runtime_preflight(config_file)
    config_data = resolve_runtime_config_data(config_file)

    if preflight["live_ready"]:
        return config_data

    if not preflight["config_file_present"] and preflight["config_profile_source"] != "environment":
        parser.exit(
            1,
            f"Error: Config file not found: {config_file}\n"
            f"Copy config_example.yaml to {config_file} and set compartment_id, config_profile, and model_id before running the pipeline.\n"
            "You can also set PLANELLM_COMPARTMENT_ID / PLANELLM_MODEL_ID and rely on ~/.oci/config for auth.\n",
        )

    if preflight["config_error"]:
        parser.exit(1, f"Error: {preflight['config_error']}\n")

    missing_keys = []
    if not preflight["compartment_id_present"]:
        missing_keys.append("compartment_id")
    if not preflight["model_id_present"]:
        missing_keys.append("model_id")
    if not preflight["config_profile"]:
        missing_keys.append("config_profile")
    if missing_keys:
        parser.exit(1, f"Error: Missing required config values in {config_file}: {', '.join(missing_keys)}\n")

    if preflight["config_file_present"] and (
        not preflight["compartment_id_present"] or not preflight["model_id_present"]
    ):
        parser.exit(
            1,
            f"Error: Replace placeholder values in {config_file} before running the pipeline.\n",
        )

    if preflight["config_profile"] and not preflight["oci_profile_available"]:
        parser.exit(
            1,
            f"Error: OCI profile '{preflight['config_profile']}' is not available via ~/.oci/config. "
            f"Run 'oci setup config' or update {config_file}.\n",
        )

    if preflight["config_profile"] and not preflight["oci_auth"]:
        parser.exit(
            1,
            f"Error: OCI profile '{preflight['config_profile']}' could not be authenticated via ~/.oci/config. "
            f"Run 'oci setup config' or update {config_file}. Original error: {preflight['oci_auth_error']}\n",
        )

    return config_data


def print_doctor_report(config_file: str) -> None:
    report = build_runtime_preflight(config_file)
    profiles = ", ".join(report["oci_profiles"]) if report["oci_profiles"] else "(none found)"

    print("planeLLM Doctor")
    print("================")
    print(f"Config file:        {report['config_file'] or config_file}")
    print(f"Config present:     {'yes' if report['config_file_present'] else 'no'}")
    print(f"Selected profile:   {report['config_profile'] or '(none)'} ({report['config_profile_source']})")
    print(f"OCI profiles:       {profiles}")
    print(f"OCI auth ready:     {'yes' if report['oci_auth'] else 'no'}")
    print(f"Compartment ID:     {'ready' if report['compartment_id_present'] else 'missing'}")
    print(f"Model ID:           {'ready' if report['model_id_present'] else 'missing'}")
    print(f"FFmpeg:             {'ready' if report['ffmpeg'] else 'missing'}")
    print(f"Fish SDK:           {'installed' if report['fish_sdk'] else 'missing'}")
    print(f"FISH_API_KEY:       {'set' if report['fish_api_key'] else 'missing'}")
    print(f"Resources tracked:  {report['resources_count']}")
    print(f"Recommended mode:   {report['recommended_mode']}")
    print(f"Next step:          {report['next_step']}")
    if report["issues"]:
        print("\nIssues")
        print("------")
        for issue in report["issues"]:
            print(f"- {issue}")


def main():
    global TopicExplorer, PodcastWriter, TTSGenerator, timestamp_slug, create_demo_bundle, build_runtime_preflight, resolve_runtime_config_data

    parser = argparse.ArgumentParser(description="Generate an educational podcast on any topic")
    parser.add_argument("--topic", help="Topic to generate a podcast about")
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
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Inspect first-run readiness, OCI profiles, and local runtime dependencies",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Create a deterministic demo bundle without OCI or Fish credentials",
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
    if build_runtime_preflight is None:
        from plane_llm_utils import build_runtime_preflight as build_runtime_preflight_fn
        build_runtime_preflight = build_runtime_preflight_fn
    if resolve_runtime_config_data is None:
        from plane_llm_utils import resolve_runtime_config_data as resolve_runtime_config_data_fn
        resolve_runtime_config_data = resolve_runtime_config_data_fn
    if create_demo_bundle is None:
        from demo_bundle import create_demo_bundle as create_demo_bundle_fn
        create_demo_bundle = create_demo_bundle_fn

    if args.doctor:
        print_doctor_report(args.config)
        return

    if not args.topic:
        parser.error("--topic is required unless --doctor is used")

    if args.demo:
        bundle = create_demo_bundle(args.topic, output_audio_path=args.output)
        print("\n=== Demo Bundle Complete ===")
        print(f"Questions: ./resources/{bundle['questions_file']}")
        print(f"Content: ./resources/{bundle['content_file']}")
        print(f"Transcript: ./resources/{bundle['transcript_file']}")
        if bundle["audio_path"]:
            print(f"Audio: {bundle['audio_path']}")
        else:
            print(f"Audio: {bundle['audio_message']}")
        print(f"\n{bundle['message']}")
        return

    config_data = ensure_runtime_config_ready(parser, args.config)
    os.makedirs("./resources", exist_ok=True)

    print(f"\n=== Step 1: Exploring topic '{args.topic}' ===")
    explorer = TopicExplorer(config_file=args.config, config_data=config_data)
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
    writer = PodcastWriter(config_file=args.config, config_data=config_data, transcript_length=args.transcript_length)
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
