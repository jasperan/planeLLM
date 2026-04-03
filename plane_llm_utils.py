"""Shared helpers for planeLLM runtime, config, and file handling."""

from __future__ import annotations

import json
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


def has_oci_runtime_config(config_file: Optional[str] = "config.yaml") -> bool:
    """Return True when the default OCI runtime config is locally usable."""
    try:
        import oci
    except ImportError:
        return False

    try:
        config_data = load_yaml_config(config_file)
    except (FileNotFoundError, ValueError, yaml.YAMLError):
        return False

    for key in REQUIRED_OCI_CONFIG_KEYS:
        value = str(config_data.get(key, "")).strip()
        if not value:
            return False
        lowered = value.lower()
        if lowered in PLACEHOLDER_CONFIG_TOKENS or "example" in lowered:
            return False

    try:
        oci.config.from_file(str(Path("~/.oci/config").expanduser()), str(config_data["config_profile"]))
    except Exception:
        return False

    return True


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
