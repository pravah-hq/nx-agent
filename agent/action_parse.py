from __future__ import annotations

import json
import re
from typing import Any

from agent.types import POLE_TYPES, Action, ActionType, PoleType


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def parse_action_response(
    text: str,
    *,
    allowed: list[str] | None = None,
) -> Action | None:
    payload = extract_json_object(text)
    if not payload:
        return None

    raw_action = str(payload.get("action", "")).strip().lower()
    try:
        action_type = ActionType(raw_action)
    except ValueError:
        return None

    if allowed and action_type.value not in allowed:
        return None

    pole_type: PoleType | None = None
    raw_type = payload.get("pole_type")
    if raw_type is not None and str(raw_type).lower() not in {"null", "none", ""}:
        candidate = str(raw_type).strip().lower()
        if candidate not in POLE_TYPES:
            return None
        pole_type = candidate  # type: ignore[assignment]

    stop_after = bool(payload.get("stop_after", False))
    if action_type != ActionType.CLASSIFY_OR_STOP:
        pole_type = None
        stop_after = False

    return Action(type=action_type, pole_type=pole_type, stop_after=stop_after)
