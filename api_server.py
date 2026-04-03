#!/usr/bin/env python
"""FastAPI server for planeLLM, exposing the podcast pipeline over HTTP."""

from __future__ import annotations

import importlib.util
import os
import traceback
from pathlib import Path
from typing import Optional

import api_workflow
from demo_bundle import DEFAULT_DEMO_TOPIC, create_demo_bundle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from plane_llm_utils import build_runtime_preflight

RESOURCES = Path("./resources").resolve()
RESOURCES.mkdir(exist_ok=True)
STATIC_DIR = Path("./static")
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


class DemoRequest(BaseModel):
    topic: str = DEFAULT_DEMO_TOPIC


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


def _get_tts_generator_class():
    from tts_generator import TTSGenerator

    return TTSGenerator


def _resource_file(file_name: str) -> Path:
    try:
        return api_workflow.resource_file(RESOURCES, file_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _is_requested_file_not_found(exc: FileNotFoundError, requested_name: str) -> bool:
    return bool(exc.args) and str(exc.args[0]) == requested_name


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
    preflight = build_runtime_preflight(resources_dir=RESOURCES)
    preflight["fish_sdk"] = preflight.get("fish_sdk", FISH_AUDIO_AVAILABLE)
    return preflight


@app.get("/api/preflight")
def get_preflight():
    preflight = build_runtime_preflight(resources_dir=RESOURCES)
    preflight["fish_sdk"] = preflight.get("fish_sdk", FISH_AUDIO_AVAILABLE)
    return preflight


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
    except FileNotFoundError as exc:
        return {
            "success": False,
            "message": str(exc),
            "questions_file": "",
            "content_file": "",
            "questions": [],
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
    try:
        return api_workflow.create_transcript_response(
            resources=RESOURCES,
            request=req,
            get_writer=_get_writer,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        missing_target = str(exc.args[0]) if exc.args else ""
        if _is_requested_file_not_found(exc, req.content_file):
            raise HTTPException(status_code=404, detail=f"File not found: {req.content_file}") from exc
        return {
            "success": False,
            "message": missing_target or str(exc),
            "transcript_file": "",
            "transcript_preview": "",
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
    try:
        return api_workflow.generate_audio_response(
            resources=RESOURCES,
            request=req,
            get_tts_generator_class=_get_tts_generator_class,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        missing_target = str(exc.args[0]) if exc.args else ""
        if _is_requested_file_not_found(exc, req.transcript_file):
            raise HTTPException(status_code=404, detail=f"File not found: {req.transcript_file}") from exc
        return {
            "success": False,
            "message": missing_target or str(exc),
            "audio_file": "",
        }
    except RuntimeError as exc:
        return {
            "success": False,
            "message": str(exc),
            "audio_file": "",
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


@app.post("/api/demo/bootstrap")
def bootstrap_demo(req: DemoRequest):
    try:
        bundle = create_demo_bundle(req.topic, resources_dir=RESOURCES)
        return {
            "success": True,
            "message": bundle["message"],
            "topic": bundle["topic"],
            "questions_file": bundle["questions_file"],
            "content_file": bundle["content_file"],
            "transcript_file": bundle["transcript_file"],
            "audio_file": bundle["audio_file"],
            "audio_message": bundle["audio_message"],
            "questions": bundle["questions"],
            "content_preview": str(bundle.get("content", ""))[:500],
        }
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - thin exception wrapper
        traceback.print_exc()
        return {
            "success": False,
            "message": str(exc),
            "topic": req.topic,
            "questions_file": "",
            "content_file": "",
            "transcript_file": "",
            "audio_file": "",
            "audio_message": "",
            "questions": [],
            "content_preview": "",
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
