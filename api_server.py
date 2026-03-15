#!/usr/bin/env python
"""FastAPI server for planeLLM, exposing the podcast pipeline over HTTP."""

from __future__ import annotations

import importlib.util
import os
import shutil
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from plane_llm_utils import safe_resource_path, timestamp_slug


RESOURCES = Path("./resources").resolve()
RESOURCES.mkdir(exist_ok=True)
STATIC_DIR = Path("./static")
OCI_AVAILABLE = importlib.util.find_spec("oci") is not None
FISH_AUDIO_AVAILABLE = importlib.util.find_spec("fishaudio") is not None


def _allowed_origins() -> list[str]:
    configured = os.getenv("PLANELLM_ALLOWED_ORIGINS", "").strip()
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        "http://127.0.0.1:7860",
        "http://localhost:7860",
        "http://127.0.0.1:7880",
        "http://localhost:7880",
    ]


app = FastAPI(title="planeLLM API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.mount("/resources", StaticFiles(directory=str(RESOURCES)), name="resources")


class TopicRequest(BaseModel):
    topic: str


class TranscriptRequest(BaseModel):
    content_file: str
    detailed: bool = True


class AudioRequest(BaseModel):
    transcript_file: str
    tts_model: str = "fish"
    fish_reference: Optional[str] = ""
    fish_emotion: Optional[str] = ""


_explorer = None
_writer = None


def _get_explorer():
    global _explorer
    if _explorer is None:
        from topic_explorer import TopicExplorer

        _explorer = TopicExplorer()
    return _explorer


def _get_writer():
    global _writer
    if _writer is None:
        from lesson_writer import PodcastWriter

        _writer = PodcastWriter()
    return _writer


def _resource_file(file_name: str) -> Path:
    try:
        return safe_resource_path(RESOURCES, file_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _count_resources(suffix: str = "", keyword: str = "") -> int:
    if not RESOURCES.exists():
        return 0
    return sum(
        1
        for resource in RESOURCES.iterdir()
        if resource.is_file()
        and (not suffix or resource.suffix == suffix)
        and (not keyword or keyword in resource.name)
    )


@app.get("/api/status")
def get_status():
    return {
        "oci_config": OCI_AVAILABLE,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "fish_sdk": FISH_AUDIO_AVAILABLE,
        "resources_count": sum(
            [
                _count_resources(suffix=".txt", keyword="questions"),
                _count_resources(suffix=".txt", keyword="content"),
                _count_resources(suffix=".txt", keyword="podcast"),
                _count_resources(suffix=".mp3"),
            ]
        ),
    }


@app.get("/api/files")
def list_files():
    if not RESOURCES.exists():
        return {"questions": [], "content": [], "transcripts": [], "audio": []}

    all_files = sorted(resource.name for resource in RESOURCES.iterdir() if resource.is_file())
    return {
        "questions": [name for name in all_files if name.endswith(".txt") and "questions" in name],
        "content": [name for name in all_files if name.endswith(".txt") and ("content" in name or "raw_lesson" in name)],
        "transcripts": [name for name in all_files if name.endswith(".txt") and "podcast" in name],
        "audio": [name for name in all_files if name.endswith(".mp3")],
    }


@app.post("/api/topic/generate")
def generate_topic(req: TopicRequest):
    try:
        bundle = _get_explorer().generate_topic_bundle(req.topic)
        return {
            "success": True,
            "message": f"Generated {len(bundle['questions'])} questions and content for '{req.topic}'",
            "questions_file": bundle["questions_file"],
            "content_file": bundle["content_file"],
            "questions": bundle["questions"],
        }
    except Exception as exc:  # pragma: no cover - thin exception wrapper
        traceback.print_exc()
        return {
            "success": False,
            "message": str(exc),
            "questions_file": "",
            "content_file": "",
            "questions": [],
        }


@app.post("/api/transcript/create")
def create_transcript(req: TranscriptRequest):
    content_path = _resource_file(req.content_file)
    if not content_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.content_file}")

    try:
        writer = _get_writer()
        content = content_path.read_text(encoding="utf-8")
        transcript = writer.create_detailed_podcast_transcript(content) if req.detailed else writer.create_podcast_transcript(content)
        transcript_file = f"podcast_transcript_{timestamp_slug()}.txt"
        _resource_file(transcript_file).write_text(transcript, encoding="utf-8")
        return {
            "success": True,
            "message": "Transcript created successfully",
            "transcript_file": transcript_file,
            "transcript_preview": transcript[:500],
        }
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - thin exception wrapper
        traceback.print_exc()
        return {
            "success": False,
            "message": str(exc),
            "transcript_file": "",
            "transcript_preview": "",
        }


@app.post("/api/audio/generate")
def generate_audio(req: AudioRequest):
    transcript_path = _resource_file(req.transcript_file)
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.transcript_file}")

    try:
        from tts_generator import TTSGenerator

        transcript = transcript_path.read_text(encoding="utf-8")
        output_path = str(_resource_file(f"podcast_{timestamp_slug()}.mp3"))
        tts = TTSGenerator(model_type=req.tts_model, fish_reference_id=req.fish_reference or None)
        result = tts.generate_podcast(transcript, output_path=output_path)
        return {
            "success": True,
            "message": "Audio generated successfully",
            "audio_file": Path(result).name,
        }
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - thin exception wrapper
        traceback.print_exc()
        return {
            "success": False,
            "message": str(exc),
            "audio_file": "",
        }


@app.get("/")
def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("PLANELLM_HOST", "127.0.0.1"),
        port=int(os.getenv("PLANELLM_PORT", "7880")),
    )
