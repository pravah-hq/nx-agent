"""
Core types and constants for the agent.

Tweak geometry here to match the React app (src/App.tsx) or experiment with
stricter/looser visibility rules in environment.poles_in_view().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# --- Geometry (aligned with web UI) -----------------------------------------

# Max edge length in the pano graph; move only goes to neighbors within this.
PANO_PROXIMITY_MAX_M = 20
# When using aligned move (no explicit target_pano_id), neighbor must be within this
# bearing cone from the current view direction.
STEP_ALIGNMENT_MAX_DEG = 45
# Poles farther than this are never in poles_in_view (geometric viewshed).
VIEW_SHED_RADIUS_M = 42
# Horizontal field of view used for poles_in_view cone (degrees).
DEFAULT_HFOV_DEG = 100
# Agent heading is discrete: 12 bins × 30° each.
DIRECTION_BIN_COUNT = 12
DIRECTION_BIN_WIDTH_DEG = 30

# --- Pole labels (task definition) --------------------------------------------

PoleType = Literal[
    "distribution_transformer",
    "lamp_post",
    "billboard_pole",
    "low_tension_pole",
]

POLE_TYPES: tuple[PoleType, ...] = (
    "distribution_transformer",
    "lamp_post",
    "billboard_pole",
    "low_tension_pole",
)


class ActionType(str, Enum):
    """Discrete actions the environment.apply_action understands."""

    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    MOVE = "move"
    CLASSIFY_OR_STOP = "classify_or_stop"


@dataclass(frozen=True)
class Pano:
    """One street panorama node (position + equirectangular image metadata)."""

    id: str
    image_path: str
    session_id: str
    order_in_session: int
    lat: float
    lon: float
    heading_deg: float
    width: int
    height: int


@dataclass(frozen=True)
class Pole:
    """Ground-truth pole location from poles.geojson (not the predicted type)."""

    track_id: str
    pole_id: str
    lat: float
    lon: float
    quality: str | None
    pole_material: str | None
    n_sightings: int | None
    service_drop: bool | None


@dataclass(frozen=True)
class PoleInView:
    """
    Geometric visibility of a pole from current pano + direction_bin.
    Used by stub policy and as hints in VLM JSON; not the same as
    pole_in_clear_view (VLM: unclassified pole visible in street view).
    """

    track_id: str
    pole_id: str
    bearing_deg: float
    distance_m: float
    angle_from_view_deg: float
    in_center: bool


@dataclass
class PoleGuess:
    """Working hypothesis for the pole being hunted (type filled after classify)."""

    track_id: str
    pole_id: str
    pole_type: PoleType | None = None
    confidence: float | None = None
    note: str | None = None


@dataclass
class AgentState:
    """
    Full agent state each step.

    - pano_id + direction_bin: where you stand and which way you face
    - pole_in_consideration: track_id of the pole you are trying to classify next
    - classified: map track_id -> predicted PoleType
    """

    pano_id: str
    direction_bin: int
    pole_in_consideration: str | None = None
    pole_guess: PoleGuess | None = None
    classified: dict[str, PoleType] = field(default_factory=dict)

    def copy(self) -> AgentState:
        return AgentState(
            pano_id=self.pano_id,
            direction_bin=self.direction_bin,
            pole_in_consideration=self.pole_in_consideration,
            pole_guess=self.pole_guess,
            classified=dict(self.classified),
        )


@dataclass(frozen=True)
class Action:
    """Command issued by a policy; applied by World.apply_action."""

    type: ActionType
    pole_type: PoleType | None = None  # required for classify_or_stop (VLM/stub)
    stop_after: bool = False  # end autonomous run after this classify
    target_pano_id: str | None = None  # move destination (resolved from map point or VLM id)
    map_point_px: tuple[int, int] | None = None  # VLM pick on overview map (x, y), top-left origin


@dataclass
class StepRecord:
    """One step of a run (for logging / debugging)."""

    step: int
    action: Action
    state_before: AgentState
    state_after: AgentState
    pole_in_clear_view: bool
    message: str = ""
    vlm_calls: list[dict] = field(default_factory=list)
    visible_pole_id: str | None = None
