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


def parse_navigation_response(
    text: str,
    *,
    allowed: list[str] | None = None,
    neighbor_ids: list[str] | None = None,
    blocked_move_targets: frozenset[str] | None = None,
) -> tuple[Action | None, bool]:
    """Returns (action, wants_assess). wants_assess True when action is assess_classify."""
    payload = extract_json_object(text)
    if not payload:
        return None, False

    raw_action = str(payload.get("action", "")).strip().lower()
    if raw_action in {"classify_or_stop", "classify"}:
        return None, False

    if raw_action == "assess_classify":
        if allowed and "assess_classify" not in allowed:
            return None, False
        return None, True

    try:
        action_type = ActionType(raw_action)
    except ValueError:
        return None, False

    if allowed and action_type.value not in allowed:
        return None, False

    target_pano_id = payload.get("target_pano_id")
    if target_pano_id is not None and str(target_pano_id).lower() in {"null", "none", ""}:
        target_pano_id = None
    else:
        target_pano_id = str(target_pano_id) if target_pano_id else None

    if action_type == ActionType.MOVE:
        if not target_pano_id:
            return None, False
        if neighbor_ids is not None and target_pano_id not in neighbor_ids:
            return None, False
        if blocked_move_targets and target_pano_id in blocked_move_targets:
            return None, False

    return Action(type=action_type, target_pano_id=target_pano_id), False


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


def parse_pole_type_response(text: str) -> tuple[PoleType | None, str]:
    payload = extract_json_object(text)
    if not payload:
        return None, "no JSON"
    raw_type = payload.get("pole_type")
    if raw_type is None or str(raw_type).lower() in {"null", "none", ""}:
        return None, "missing pole_type"
    candidate = str(raw_type).strip().lower()
    if candidate not in POLE_TYPES:
        return None, f"invalid pole_type {candidate}"
    return candidate, ""  # type: ignore[return-value]


def parse_visibility_response(text: str) -> tuple[bool | None, str]:
    payload = extract_json_object(text)
    if not payload:
        return None, "no JSON"
    if "view_clear" not in payload:
        return None, "missing view_clear"
    view_clear = payload.get("view_clear") in (True, "true", "True", 1, "1")
    return view_clear, ""
