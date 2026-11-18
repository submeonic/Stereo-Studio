"""Versioned, validated user-authored presets. Never deserialize executable objects."""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .editing import PhotoEdits
from .temporal import SPECS, validate

SCHEMA = "stereo-studio.user-preset"


def photo_state(edits, rotation=0, aspect=None):
    return {"adjustments": asdict(edits), "rotation": rotation, "aspect": aspect}


def validate_preset(data, kind):
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or data.get("version") != 1:
        raise ValueError("Not a supported Stereo Studio user preset (version 1 required).")
    if data.get("kind") != kind:
        raise ValueError(f"This file is not a preset for the {kind} panel.")
    state = data.get("settings")
    if not isinstance(state, dict):
        raise ValueError("Preset settings must be an object.")
    try:
        if kind == "photo":
            if set(state) != {"adjustments", "rotation", "aspect"}:
                raise ValueError("Unknown or missing photo settings.")
            edits = PhotoEdits(**state["adjustments"])
            edits.validate()
            if not isinstance(edits.auto_color, bool):
                raise ValueError("Auto color must be a boolean.")
            if state["rotation"] not in (0, 90, 180, 270):
                raise ValueError("Invalid rotation.")
            if state["aspect"] not in (None, "original", 1.0, 4 / 3, 1.5, 16 / 9, 9 / 16):
                raise ValueError("Unknown crop aspect preset.")
            state = photo_state(edits, state["rotation"], state["aspect"])
        elif kind == "effects":
            if set(state) != {"parameters", "enabled", "bypass"}:
                raise ValueError("Unknown or missing effect settings.")
            parameters = validate(state["parameters"])
            if set(parameters) != set(SPECS):
                raise ValueError("Effect preset must contain every effect group.")
            enabled = state["enabled"]
            if not isinstance(enabled, list) or any(
                not isinstance(x, str) or x not in SPECS for x in enabled
            ):
                raise ValueError("Unknown enabled effect.")
            if not isinstance(state["bypass"], bool):
                raise ValueError("Bypass must be a boolean.")
            state = dict(
                parameters=parameters, enabled=list(dict.fromkeys(enabled)), bypass=state["bypass"]
            )
        else:
            raise ValueError("Unknown preset kind.")
    except (TypeError, KeyError, OverflowError) as exc:
        raise ValueError("Malformed preset settings.") from exc
    return state


def load_preset(path, kind):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Preset is too large (maximum 1 MB).")
    return validate_preset(json.loads(path.read_text(encoding="utf-8")), kind)


def save_preset(path, kind, settings):
    data = dict(schema=SCHEMA, version=1, kind=kind, settings=settings)
    data["settings"] = validate_preset(data, kind)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
