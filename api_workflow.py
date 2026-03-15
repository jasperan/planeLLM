from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Mapping

from plane_llm_utils import safe_resource_path, timestamp_slug


def resource_file(resources: Path, file_name: str) -> Path:
    return safe_resource_path(resources.resolve(), file_name)


def create_transcript_response(*, resources: Path, request, get_writer: Callable[[], object]) -> dict:
    content_path = resource_file(resources, request.content_file)
    if not content_path.exists():
        raise FileNotFoundError(request.content_file)

    writer = get_writer()
    content = content_path.read_text(encoding="utf-8")
    transcript = writer.create_detailed_podcast_transcript(content) if request.detailed else writer.create_podcast_transcript(content)
    transcript_file = f"podcast_transcript_{timestamp_slug()}.txt"
    resource_file(resources, transcript_file).write_text(transcript, encoding="utf-8")
    return {
        "success": True,
        "message": "Transcript created successfully",
        "transcript_file": transcript_file,
        "transcript_preview": transcript[:500],
    }


def generate_audio_response(*, resources: Path, request, get_tts_generator_class: Callable[[], object]) -> dict:
    transcript_path = resource_file(resources, request.transcript_file)
    if not transcript_path.exists():
        raise FileNotFoundError(request.transcript_file)

    transcript = transcript_path.read_text(encoding="utf-8")
    output_path = str(resource_file(resources, f"podcast_{timestamp_slug()}.mp3"))
    tts_class = get_tts_generator_class()
    tts = tts_class(model_type=request.tts_model, fish_reference_id=request.fish_reference or None)
    result = tts.generate_podcast(transcript, output_path=output_path)
    return {
        "success": True,
        "message": "Audio generated successfully",
        "audio_file": Path(result).name,
    }


@contextmanager
def patched_environ(values: Mapping[str, str]) -> Iterator[None]:
    old = os.environ.copy()
    try:
        os.environ.clear()
        os.environ.update(values)
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)
