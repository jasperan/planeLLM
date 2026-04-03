"""Shared helpers for planeLLM runtime, config, and file handling."""

from __future__ import annotations

import configparser
import importlib.util
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

OCI_GENAI_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
REQUIRED_OCI_CONFIG_KEYS = ("compartment_id", "config_profile", "model_id")
PLACEHOLDER_CONFIG_TOKENS = {
    "compartment_ocid",
    "profile_name",
    "model_ocid",
}
ENV_CONFIG_KEYS = {
    "compartment_id": ("PLANELLM_COMPARTMENT_ID", "OCI_COMPARTMENT_ID"),
    "config_profile": ("PLANELLM_CONFIG_PROFILE", "OCI_CLI_PROFILE", "OCI_PROFILE"),
    "model_id": ("PLANELLM_MODEL_ID", "OCI_MODEL_ID"),
}


def timestamp_slug() -> str:
    """Return a collision-resistant UTC timestamp for resource filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def load_yaml_config(config_file: Optional[str]) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    if not config_file:
        return {}

    path = Path(config_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")

    return data


def list_oci_profiles(config_file: Optional[str] = None) -> list[str]:
    """Return profile names from the local OCI CLI config file."""
    path = Path(config_file or "~/.oci/config").expanduser()
    if not path.exists():
        return []

    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    profiles: list[str] = []
    if parser.defaults():
        profiles.append(parser.default_section)
    profiles.extend(section for section in parser.sections() if section not in profiles)
    return profiles


def resolve_runtime_config_data(config_file: Optional[str] = "config.yaml") -> dict[str, Any]:
    """Resolve runtime config from config.yaml plus environment fallbacks."""
    config_data: dict[str, Any] = {}
    if config_file:
        config_path = Path(config_file).expanduser()
        if config_path.exists():
            config_data = load_yaml_config(str(config_path))

    resolved = dict(config_data)
    for key, env_names in ENV_CONFIG_KEYS.items():
        for env_name in env_names:
            value = str(os.environ.get(env_name, "")).strip()
            if value:
                resolved[key] = value
                break

    if not str(resolved.get("config_profile", "")).strip():
        profiles = list_oci_profiles()
        if "DEFAULT" in profiles:
            resolved["config_profile"] = "DEFAULT"
        elif len(profiles) == 1:
            resolved["config_profile"] = profiles[0]

    return resolved


def _value_present(value: Any) -> bool:
    lowered = str(value or "").strip().lower()
    return bool(lowered) and lowered not in PLACEHOLDER_CONFIG_TOKENS and "example" not in lowered


def build_runtime_preflight(
    config_file: Optional[str] = "config.yaml",
    *,
    resources_dir: str | Path = "./resources",
    oci_config_file: str | Path = "~/.oci/config",
) -> dict[str, Any]:
    """Build a first-run readiness report for live and demo modes."""
    resources_path = Path(resources_dir).expanduser()
    config_path = Path(config_file).expanduser() if config_file else None
    oci_config_path = Path(oci_config_file).expanduser()
    profiles = list_oci_profiles(str(oci_config_path))

    config_data: dict[str, Any] = {}
    config_error = ""
    if config_path and config_path.exists():
        try:
            config_data = load_yaml_config(str(config_path))
        except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
            config_error = str(exc)
    elif config_path:
        config_error = f"Config file not found: {config_path}"

    resolved_config = resolve_runtime_config_data(config_file)
    config_profile = str(resolved_config.get("config_profile") or "").strip()
    config_profile_source = "config.yaml"
    env_profile = os.environ.get("PLANELLM_CONFIG_PROFILE") or os.environ.get("OCI_CLI_PROFILE") or os.environ.get("OCI_PROFILE")
    if env_profile:
        config_profile_source = "environment"
    elif not str(config_data.get("config_profile") or "").strip() and config_profile:
        config_profile_source = "~/.oci/config"
    if not config_profile:
        if "DEFAULT" in profiles:
            config_profile = "DEFAULT"
            config_profile_source = "~/.oci/config"
        elif len(profiles) == 1:
            config_profile = profiles[0]
            config_profile_source = "~/.oci/config"
        else:
            config_profile_source = "missing"

    compartment_id = str(resolved_config.get("compartment_id") or "").strip()
    model_id = str(resolved_config.get("model_id") or "").strip()
    compartment_source = "environment" if os.environ.get("PLANELLM_COMPARTMENT_ID") else "config.yaml"
    model_source = "environment" if os.environ.get("PLANELLM_MODEL_ID") else "config.yaml"

    profile_available = bool(config_profile and config_profile in profiles)
    oci_sdk_available = importlib.util.find_spec("oci") is not None
    oci_auth = False
    oci_auth_error = ""
    if oci_sdk_available and profile_available:
        try:
            import oci

            oci.config.from_file(str(oci_config_path), config_profile)
            oci_auth = True
        except Exception as exc:  # pragma: no cover - environment-specific
            oci_auth_error = str(exc)

    ffmpeg_ready = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
    fish_sdk_available = importlib.util.find_spec("fishaudio") is not None
    fish_api_key = bool(os.environ.get("FISH_API_KEY", "").strip())
    compartment_ready = _value_present(compartment_id)
    model_ready = _value_present(model_id)
    live_ready = bool(compartment_ready and model_ready and config_profile and profile_available and oci_auth)
    demo_ready = ffmpeg_ready
    resource_count = 0
    if resources_path.exists():
        resource_count = sum(1 for resource in resources_path.iterdir() if resource.is_file())

    issues: list[str] = []
    if config_error:
        issues.append(config_error)
    if not profiles:
        issues.append(f"OCI CLI config not found at {oci_config_path}")
    elif config_profile and not profile_available:
        issues.append(f"OCI profile '{config_profile}' is not available in {oci_config_path}")
    if not compartment_ready:
        issues.append("compartment_id is missing or still a placeholder")
    if not model_ready:
        issues.append("model_id is missing or still a placeholder")
    if config_profile and profile_available and not oci_auth and oci_auth_error:
        issues.append(f"Unable to authenticate OCI profile '{config_profile}': {oci_auth_error}")
    if not ffmpeg_ready:
        issues.append("FFmpeg/ffprobe are not available for audio export")
    if not fish_api_key:
        issues.append("FISH_API_KEY is not set for the default cloud TTS backend")

    if live_ready:
        recommended_mode = "live"
        next_step = "Live generation is ready. Run the unified pipeline or one of the interactive apps."
    elif demo_ready:
        recommended_mode = "demo"
        next_step = (
            "Create a demo bundle now to exercise the full workflow, then add compartment_id/model_id "
            "for live OCI generation."
        )
    else:
        recommended_mode = "setup"
        next_step = "Install FFmpeg, then create a demo bundle or finish OCI runtime configuration."

    return {
        "config_file": str(config_path) if config_path else "",
        "config_file_present": bool(config_path and config_path.exists()),
        "config_error": config_error,
        "config_profile": config_profile,
        "config_profile_source": config_profile_source,
        "oci_profiles": profiles,
        "oci_profile_available": profile_available,
        "oci_sdk": oci_sdk_available,
        "oci_auth": oci_auth,
        "oci_auth_error": oci_auth_error,
        "compartment_id_present": compartment_ready,
        "compartment_id_source": compartment_source if compartment_id else "missing",
        "model_id_present": model_ready,
        "model_id_source": model_source if model_id else "missing",
        "oci_config": live_ready,
        "live_ready": live_ready,
        "ffmpeg": ffmpeg_ready,
        "fish_sdk": fish_sdk_available,
        "fish_api_key": fish_api_key,
        "demo_ready": demo_ready,
        "recommended_mode": recommended_mode,
        "next_step": next_step,
        "issues": issues,
        "resources_count": resource_count,
    }


def has_oci_runtime_config(config_file: Optional[str] = "config.yaml") -> bool:
    """Return True when the default OCI runtime config is locally usable."""
    return bool(build_runtime_preflight(config_file).get("live_ready"))


def build_genai_client(
    config_data: Mapping[str, Any],
    *,
    oci_config: Optional[Mapping[str, Any]] = None,
    client_cls: Any = None,
):
    """Build an OCI Generative AI inference client."""
    import oci

    client_cls = client_cls or oci.generative_ai_inference.GenerativeAiInferenceClient
    resolved_oci_config = (
        dict(oci_config)
        if oci_config is not None
        else oci.config.from_file(str(Path("~/.oci/config").expanduser()), config_data["config_profile"])
    )

    return client_cls(
        config=resolved_oci_config,
        service_endpoint=OCI_GENAI_ENDPOINT,
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240),
    )


def extract_chat_text(response: Any) -> str:
    """Extract the text payload from OCI responses and test doubles."""
    payload = getattr(response, "data", response)

    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    elif isinstance(payload, str):
        payload = json.loads(payload)
    elif not isinstance(payload, dict):
        try:
            payload = json.loads(str(payload))
        except json.JSONDecodeError:
            payload = getattr(payload, "__dict__", payload)

    if not isinstance(payload, dict):
        raise ValueError("Unsupported OCI response payload shape")

    try:
        return payload["chat_response"]["choices"][0]["message"]["content"][0]["text"]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Unable to extract chat text from OCI response payload") from exc


def explain_oci_error(exc: Exception, *, model_id: str = "") -> str:
    """Turn common OCI failures into actionable user-facing guidance."""
    message = str(exc)
    lowered = message.lower()
    if model_id and "entity with key" in lowered and "not found" in lowered:
        return (
            f"Configured model_id '{model_id}' was not found in OCI Generative AI. "
            "Update model_id to a currently available model for this tenancy and region."
        )
    return message


def safe_resource_path(base_dir: Path, file_name: str) -> Path:
    """Return a safe direct child path under base_dir."""
    if not file_name:
        raise ValueError("File name is required")

    candidate = Path(file_name)
    if candidate.name != file_name:
        raise ValueError("Nested or relative paths are not allowed")

    resolved_base = base_dir.resolve()
    resolved_path = (resolved_base / candidate.name).resolve()
    if resolved_path.parent != resolved_base:
        raise ValueError("Resolved path escapes the resources directory")

    return resolved_path
