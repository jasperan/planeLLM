"""Deterministic demo bundle generation for first-run planeLLM walkthroughs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

from pydub import AudioSegment
from pydub.generators import Sine

from plane_llm_utils import timestamp_slug

DEFAULT_DEMO_TOPIC = "How airplanes stay in the sky"


def _topic_questions(topic: str) -> list[str]:
    clean_topic = topic.strip() or DEFAULT_DEMO_TOPIC
    return [
        f"What is the big idea behind {clean_topic}?",
        f"What key mental models help someone understand {clean_topic} quickly?",
        f"Where does {clean_topic} show up in everyday life or industry?",
        f"What are common misconceptions about {clean_topic}?",
    ]


def _topic_content(topic: str, questions: list[str]) -> str:
    intro = (
        f"# {topic}\n\n"
        f"This is a deterministic demo lesson about {topic}. It exists so a new user can walk through "
        "planeLLM end to end before cloud credentials, OCI runtime details, or Fish Speech are fully configured.\n"
    )
    sections = [
        (
            questions[0],
            f"{topic} becomes easier to grasp when you treat it as a system with inputs, constraints, and outputs. "
            f"In this demo, the core idea is that {topic} can be taught by moving from intuition to mechanism to application."
        ),
        (
            questions[1],
            f"A useful model for {topic} is to ask: what forces or rules shape it, what tradeoffs matter most, "
            "and what observable behavior proves the explanation is working?"
        ),
        (
            questions[2],
            f"You can connect {topic} to concrete examples, operational decisions, and things a learner could explain back "
            "to a friend after hearing one short podcast episode."
        ),
        (
            questions[3],
            f"A common mistake is to reduce {topic} to a slogan. The better framing is to name the mechanism, "
            "the tradeoff, and the real-world consequence."
        ),
    ]
    parts = [intro]
    for heading, body in sections:
        parts.append(f"# {heading}\n\n{body}\n")
    return "\n".join(parts).strip() + "\n"


def _topic_transcript(topic: str, questions: list[str]) -> str:
    return "\n".join(
        [
            f"Speaker 1: Welcome to planeLLM. Today we're doing a quick demo episode about {topic}.",
            "Speaker 2: Great. Give me the shortest possible mental map first.",
            f"Speaker 1: Start with this question: {questions[0]}",
            f"Speaker 2: So the lesson is really about structure, not trivia.",
            f"Speaker 1: Exactly. Then we move to {questions[1]} so the learner has a framework.",
            f"Speaker 2: After that we connect it to the real world with {questions[2]}?",
            "Speaker 1: Right, and we close by fixing misconceptions before they harden.",
            f"Speaker 2: Which brings us to {questions[3]} and a cleaner explanation.",
            f"Speaker 1: That's the demo. Even without live OCI calls, this bundle shows how planeLLM stores reusable lesson assets for {topic}.",
        ]
    )


def _build_demo_audio() -> AudioSegment:
    lead = Sine(523).to_audio_segment(duration=250).fade_in(20).fade_out(60) - 8
    accent = Sine(659).to_audio_segment(duration=220).fade_in(20).fade_out(60) - 10
    tail = Sine(784).to_audio_segment(duration=320).fade_in(20).fade_out(80) - 12
    spacer = AudioSegment.silent(duration=120)
    return lead + spacer + accent + spacer + tail + AudioSegment.silent(duration=350)


def create_demo_bundle(
    topic: str,
    *,
    resources_dir: str | Path = "./resources",
    output_audio_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Create deterministic demo lesson assets under the resources directory."""
    resolved_topic = topic.strip() or DEFAULT_DEMO_TOPIC
    resources_path = Path(resources_dir).expanduser()
    resources_path.mkdir(parents=True, exist_ok=True)

    timestamp = timestamp_slug()
    questions = _topic_questions(resolved_topic)
    questions_text = "\n".join(f"{index}. {question}" for index, question in enumerate(questions, start=1))
    content = _topic_content(resolved_topic, questions)
    transcript = _topic_transcript(resolved_topic, questions)

    questions_file = f"questions_{timestamp}.txt"
    content_file = f"content_{timestamp}.txt"
    transcript_file = f"podcast_transcript_{timestamp}.txt"

    (resources_path / questions_file).write_text(questions_text, encoding="utf-8")
    (resources_path / content_file).write_text(content, encoding="utf-8")
    (resources_path / transcript_file).write_text(transcript, encoding="utf-8")

    audio_file = ""
    audio_path = ""
    audio_message = ""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        target_path = Path(output_audio_path).expanduser() if output_audio_path else resources_path / f"podcast_{timestamp}.mp3"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        _build_demo_audio().export(target_path, format="mp3")
        audio_file = target_path.name
        audio_path = str(target_path)
        audio_message = "Demo audio generated successfully."
    else:
        audio_message = "Demo text assets were created, but FFmpeg is unavailable so no audio file was exported."

    return {
        "success": True,
        "message": f"Created a demo bundle for '{resolved_topic}'.",
        "topic": resolved_topic,
        "questions": questions,
        "questions_text": questions_text,
        "content": content,
        "transcript": transcript,
        "questions_file": questions_file,
        "content_file": content_file,
        "transcript_file": transcript_file,
        "audio_file": audio_file,
        "audio_path": audio_path,
        "audio_message": audio_message,
    }
