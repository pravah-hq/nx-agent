"""
Normalize and compare pole id strings from VLM JSON vs metadata.

VLM may return "57" or "POLE_000057"; both should match metadata pole_id.
"""

from __future__ import annotations

import re


def normalize_pole_id(pole_id: str) -> str:
    text = pole_id.strip().upper()
    if text.startswith("POLE_"):
        return text
    digits = re.sub(r"\D", "", text)
    if digits:
        return f"POLE_{digits.zfill(6)}" if len(digits) <= 6 else f"POLE_{digits}"
    return text


def pole_ids_match(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return normalize_pole_id(a) == normalize_pole_id(b)
