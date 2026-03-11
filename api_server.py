#!/usr/bin/env python
"""FastAPI server for planeLLM — exposes the podcast pipeline over HTTP."""

import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

os.makedirs("./resources", exist_ok=True)

app = FastAPI(title="planeLLM API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated audio/text files from resources/
app.mount("/resources", StaticFiles(directory="resources"), name="resources")

RESOURCES = Path("./resources")


# ---- Request / Response Models ----

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


# ---- Lazy-initialized singletons ----

_explorer = None
_writer = None


def _get_explorer():
    """Return a cached TopicExplorer instance (created on first call)."""
    global _explorer
    if _explorer is None:
        from topic_explorer import TopicExplorer
        _explorer = TopicExplorer()
    return _explorer


def _get_writer():
    """Return a cached PodcastWriter instance (created on first call)."""
    global _writer
    if _writer is None:
        from lesson_writer import PodcastWriter
        _writer = PodcastWriter()
    return _writer


# ---- Utility helpers ----

def _count_resources(suffix: str = "", keyword: str = "") -> int:
    """Count resource files matching an optional suffix and keyword."""
    if not RESOURCES.exists():
        return 0
    return sum(
        1 for f in RESOURCES.iterdir()
        if f.is_file()
        and (not suffix or f.suffix == suffix)
        and (not keyword or keyword in f.name)
    )


# ---- Endpoints ----

@app.get("/api/status")
def get_status():
    """Service health: OCI SDK, FFmpeg, Fish Audio SDK, resource counts."""
    oci_ok = True
    try:
        import oci  # noqa: F401
    except ImportError:
        oci_ok = False

    fish_ok = True
    try:
        import fishaudio  # noqa: F401
    except ImportError:
        fish_ok = False

    return {
        "oci_config": oci_ok,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "fish_sdk": fish_ok,
        "resources_count": sum([
            _count_resources(suffix=".txt", keyword="questions"),
            _count_resources(suffix=".txt", keyword="content"),
            _count_resources(suffix=".txt", keyword="podcast"),
            _count_resources(suffix=".mp3"),
        ]),
    }


@app.get("/api/files")
def list_files():
    """List available resource files by category."""
    if not RESOURCES.exists():
        return {"questions": [], "content": [], "transcripts": [], "audio": []}

    all_files = sorted(f.name for f in RESOURCES.iterdir() if f.is_file())
    return {
        "questions": [f for f in all_files if f.endswith(".txt") and "questions" in f],
        "content": [
            f for f in all_files
            if f.endswith(".txt") and ("content" in f or "raw_lesson" in f)
        ],
        "transcripts": [f for f in all_files if f.endswith(".txt") and "podcast" in f],
        "audio": [f for f in all_files if f.endswith(".mp3")],
    }


@app.post("/api/topic/generate")
def generate_topic(req: TopicRequest):
    """Stage 1: Topic exploration — generate questions and content."""
    try:
        explorer = _get_explorer()
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        questions = explorer.generate_questions(req.topic)

        questions_file = f"questions_{timestamp}.txt"
        (RESOURCES / questions_file).write_text("\n".join(questions), encoding="utf-8")

        content = ""
        for q in questions[:2]:
            content += f"# {q}\n\n{explorer.explore_question(q)}\n\n"

        content_file = f"content_{timestamp}.txt"
        (RESOURCES / content_file).write_text(content, encoding="utf-8")

        return {
            "success": True,
            "message": f"Generated {len(questions)} questions and content for '{req.topic}'",
            "questions_file": questions_file,
            "content_file": content_file,
            "questions": questions,
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "message": str(e),
            "questions_file": "",
            "content_file": "",
            "questions": [],
        }


@app.post("/api/transcript/create")
def create_transcript(req: TranscriptRequest):
    """Stage 2: Create podcast transcript from content file."""
    content_path = RESOURCES / req.content_file
    if not content_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.content_file}")

    try:
        writer = _get_writer()
        content = content_path.read_text(encoding="utf-8")

        if req.detailed:
            transcript = writer.create_detailed_podcast_transcript(content)
        else:
            transcript = writer.create_podcast_transcript(content)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        transcript_file = f"podcast_transcript_{timestamp}.txt"
        (RESOURCES / transcript_file).write_text(transcript, encoding="utf-8")

        return {
            "success": True,
            "message": "Transcript created successfully",
            "transcript_file": transcript_file,
            "transcript_preview": transcript[:500],
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "message": str(e),
            "transcript_file": "",
            "transcript_preview": "",
        }


@app.post("/api/audio/generate")
def generate_audio(req: AudioRequest):
    """Stage 3: Generate podcast audio from transcript file."""
    transcript_path = RESOURCES / req.transcript_file
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.transcript_file}")

    try:
        # Set Fish reference env var before creating TTSGenerator
        if req.fish_reference:
            os.environ["FISH_REFERENCE_ID"] = req.fish_reference

        # Fish handles emotion tags directly in the transcript text,
        # so no special server-side processing is needed unless
        # the caller explicitly provides a non-neutral emotion string.
        if req.fish_emotion and req.fish_emotion != "(neutral)":
            pass  # noted — no modification required

        from tts_generator import TTSGenerator

        tts = TTSGenerator(model_type=req.tts_model)

        transcript = transcript_path.read_text(encoding="utf-8")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(RESOURCES / f"podcast_{timestamp}.mp3")

        result = tts.generate_podcast(transcript, output_path=output_path)

        return {
            "success": True,
            "message": "Audio generated successfully",
            "audio_file": Path(result).name,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "message": str(e),
            "audio_file": "",
        }


STATIC_DIR = Path("./static")


@app.get("/")
def serve_frontend():
    """Serve the planeLLM web UI."""
    return FileResponse(STATIC_DIR / "index.html")


# Catch-all for static assets (CSS/JS/images if added later).
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7880)
