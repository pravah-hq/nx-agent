from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

PANO_PROXIMITY_MAX_M = 20
STEP_ALIGNMENT_MAX_DEG = 45
VIEW_SHED_RADIUS_M = 42
DEFAULT_HFOV_DEG = 100
DIRECTION_BIN_COUNT = 12
DIRECTION_BIN_WIDTH_DEG = 30

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
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    MOVE = "move"
    CLASSIFY_OR_STOP = "classify_or_stop"


@dataclass(frozen=True)
class Pano:
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
    track_id: str
    pole_id: str
    bearing_deg: float
    distance_m: float
    angle_from_view_deg: float
    in_center: bool


@dataclass
class PoleGuess:
    track_id: str
    pole_id: str
    pole_type: PoleType | None = None
    confidence: float | None = None
    note: str | None = None


@dataclass
class AgentState:
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
    type: ActionType
    # For classify_or_stop: set pole_type to classify; omit to only stop the run.
    pole_type: PoleType | None = None
    stop_after: bool = False
    # For move: VLM picks a neighbor pano id from the map (must be within 20 m).
    target_pano_id: str | None = None


@dataclass
class StepRecord:
    step: int
    action: Action
    state_before: AgentState
    state_after: AgentState
    pole_in_clear_view: bool
    message: str = ""
