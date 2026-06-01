"""
Parse VLM JSON text into Actions and clear-view / pole-type results.

Models often wrap JSON in markdown fences — extract_json_object handles that.
To loosen/tighten classify gates, edit parse_pole_in_clear_view_response and
parse_pole_type_response.
"""

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
) -> Action | None:
    payload = extract_json_object(text)
    if not payload:
        return None

    raw_action = str(payload.get("action", "")).strip().lower()
    if raw_action in {"classify_or_stop", "classify", "assess_classify"}:
        return None

    try:
        action_type = ActionType(raw_action)
    except ValueError:
        return None

    if allowed and action_type.value not in allowed:
        return None

    target_pano_id = payload.get("target_pano_id")
    if target_pano_id is not None and str(target_pano_id).lower() in {"null", "none", ""}:
        target_pano_id = None
    else:
        target_pano_id = str(target_pano_id) if target_pano_id else None

    if action_type == ActionType.MOVE:
        if not target_pano_id:
            return None
        if neighbor_ids is not None and target_pano_id not in neighbor_ids:
            return None
        if blocked_move_targets and target_pano_id in blocked_move_targets:
            return None

    return Action(type=action_type, target_pano_id=target_pano_id)


def _truthy(value) -> bool:
    return value in (True, "true", "True", 1, "1")


def parse_pole_in_clear_view_response(
    text: str,
    *,
    expected_pole_id: str,
) -> tuple[bool | None, PoleType | None, str]:
    """
    VLM-only clear view: pole_in_clear_view (or view_clear) true from the model.

    identifiable_pole_type is optional — choose() can run a separate classify prompt.
    confirmed_target_pole_id is only checked when the model supplies it.
    """
    from agent.pole_ids import pole_ids_match

    payload = extract_json_object(text)
    if not payload:
        return None, None, "no JSON"

    raw_clear = _truthy(payload.get("pole_in_clear_view", False)) or _truthy(
        payload.get("view_clear", False)
    )
    if not raw_clear:
        return False, None, ""

    raw_type = payload.get("identifiable_pole_type") or payload.get("pole_type")
    pole_type: PoleType | None = None
    if raw_type is not None and str(raw_type).lower() not in {"null", "none", ""}:
        candidate = str(raw_type).strip().lower()
        if candidate not in POLE_TYPES:
            return False, None, f"invalid identifiable_pole_type {candidate}"
        pole_type = candidate  # type: ignore[assignment]

    confirmed = (
        payload.get("confirmed_target_pole_id")
        or payload.get("visible_pole_id")
        or payload.get("pole_id")
    )
    if confirmed is not None and str(confirmed).lower() not in {"null", "none", ""}:
        if not pole_ids_match(str(confirmed), expected_pole_id):
            return (
                False,
                None,
                f"confirmed id {confirmed} is not target {expected_pole_id}",
            )

    return True, pole_type, ""


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


def parse_pole_type_response(
    text: str,
    *,
    expected_pole_id: str,
) -> tuple[PoleType | None, str]:
    from agent.pole_ids import pole_ids_match

    payload = extract_json_object(text)
    if not payload:
        return None, "no JSON"
    raw_type = payload.get("pole_type")
    if raw_type is None or str(raw_type).lower() in {"null", "none", ""}:
        return None, "missing pole_type"
    candidate = str(raw_type).strip().lower()
    if candidate not in POLE_TYPES:
        return None, f"invalid pole_type {candidate}"

    classified_id = payload.get("classified_pole_id") or payload.get("pole_id")
    if classified_id is not None and str(classified_id).lower() not in {"null", "none", ""}:
        if not pole_ids_match(str(classified_id), expected_pole_id):
            return None, f"classified_pole_id {classified_id} != target {expected_pole_id}"

    if candidate == "low_tension_pole":
        conf = str(payload.get("confidence", "")).lower()
        reason = str(payload.get("reason", "")).lower()
        if conf == "low" or (
            "wire" not in reason
            and "tension" not in reason
            and "lt " not in reason
        ):
            return None, "low_tension_pole requires high confidence or wire evidence"

    return candidate, ""  # type: ignore[return-value]


def parse_visibility_response(text: str) -> tuple[bool | None, str]:
    """Legacy alias; prefer pole_in_clear_view."""
    payload = extract_json_object(text)
    if not payload:
        return None, "no JSON"
    key = "pole_in_clear_view" if "pole_in_clear_view" in payload else "view_clear"
    if key not in payload:
        return None, f"missing {key}"
    clear = payload.get(key) in (True, "true", "True", 1, "1")
    return clear, ""
