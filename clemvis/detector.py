"""
The decision maker: scene assessment + a finite-state machine.

Two steps, deliberately separated:

1. `assess()` turns raw geometry (confirmed tracks + zone layout) into a
   small, readable verdict: is anything real happening, where, how big,
   which way is it going, and how serious is it?

2. `MotionStateMachine` turns a stream of verdicts into ACTIONS over time:

                confirmed target                capture allowed
   IDLE ──────────────────────▶ ALERT ─────────────────────▶ RECORDING
    ▲                             │  ▲                           │
    │   3 s of calm               │  │ still active              │ clip done
    └─────────────────────────────┘  └───────────────────────────┘

Rules that matter:
* only CONFIRMED tracks count — see tracking.py (time gate + travel gate),
  which is why curtains and chairs never reach this file
* hysteresis — the alarm releases only after N continuous seconds of calm,
  so it cannot flap on and off every frame
* capture cooldowns — one snapshot per intrusion, one clip per 30 s. This
  is what stops the system filling a disk in ten seconds.
* zones decide severity — motion in a `major` zone is escalated even when
  the object is small; motion in an `ignore` zone is discarded in app.py.
* the FSM performs NO I/O: it returns named actions and `app.py` executes
  them. That is what makes it unit-testable with no camera.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .config import DetectorConfig
from .tracking import STATIONARY, Track
from .zones import Region, region_at


class State(Enum):
    IDLE = auto()        # watching, nothing confirmed
    ALERT = auto()       # a real target is present, alarm engaged
    RECORDING = auto()   # evidence being written


# Named actions. The app turns these into side effects.
MOTION_STARTED = "motion_started"
ALARM_ON = "alarm_on"
ALARM_OFF = "alarm_off"
MAJOR_MOTION = "major_motion"
ENTER_RECORDING = "enter_recording"
EXIT_RECORDING = "exit_recording"
TAKE_SNAPSHOT = "take_snapshot"
BACK_TO_IDLE = "back_to_idle"


@dataclass(slots=True)
class Assessment:
    """What the system believes is happening right now."""

    active: bool = False                  # at least one confirmed target
    target_count: int = 0
    total_area: float = 0.0
    direction: str = STATIONARY
    severity: str = "normal"              # normal | major
    region: Region | None = None
    primary: Track | None = None

    @property
    def zone_name(self) -> str:
        return self.region.name if self.region else "-"

    @property
    def zone_label(self) -> str:
        return self.region.label if self.region else "Unzoned"

    def describe(self) -> str:
        """One-line human summary used by logs and the console."""
        zone = f"Zone {self.zone_name} — {self.zone_label}"
        return (
            f"area={int(self.total_area)}px  {zone}  "
            f"severity={self.severity.upper()}  direction={self.direction}"
        )


def assess(
    tracks: list[Track], config: DetectorConfig, width: int, height: int
) -> Assessment:
    """Fold confirmed tracks + the zone map into a single verdict."""
    if not tracks:
        return Assessment()

    primary = max(tracks, key=lambda t: t.area)
    region = region_at(config.active_regions, primary.cx, primary.cy, width, height)

    severity = "normal"
    if region is not None and region.severity == "major":
        severity = "major"           # restricted area → escalate regardless of size
    elif primary.area >= config.major_area:
        severity = "major"           # very large object → escalate

    return Assessment(
        active=True,
        target_count=len(tracks),
        total_area=sum(t.area for t in tracks),
        direction=primary.direction,
        severity=severity,
        region=region,
        primary=primary,
    )


class MotionStateMachine:
    def __init__(self, config: DetectorConfig) -> None:
        self.cfg = config
        self.state = State.IDLE
        self.alarm_on = False
        self.last_active: float | None = None
        self.snapshots_taken = 0
        self._last_snapshot = float("-inf")
        self._last_clip_end = float("-inf")
        self._clip_started = 0.0
        self._major_announced = False

    # ── helpers ───────────────────────────────────────────────────
    def _may_snapshot(self, now: float) -> bool:
        return (
            self.snapshots_taken < self.cfg.snapshots_per_burst
            and now - self._last_snapshot >= self.cfg.snapshot_cooldown_seconds
        )

    def _may_record(self, now: float) -> bool:
        return now - self._last_clip_end >= self.cfg.clip_cooldown_seconds

    # ── main transition ───────────────────────────────────────────
    def update(self, scene: Assessment, now: float) -> list[str]:
        """
        Advance the machine by one frame. `now` is injected rather than read
        from the clock, so tests can replay any timeline deterministically.
        """
        actions: list[str] = []

        if scene.active:
            self.last_active = now
        calm = (
            float("inf") if self.last_active is None else now - self.last_active
        )

        # ── IDLE → ALERT ──────────────────────────────────────────
        if self.state is State.IDLE and scene.active:
            self.state = State.ALERT
            self.alarm_on = True
            self._major_announced = False
            actions += [MOTION_STARTED, ALARM_ON]

        # ── escalation (announced once per intrusion) ─────────────
        if (
            self.state in (State.ALERT, State.RECORDING)
            and scene.active
            and scene.severity == "major"
            and not self._major_announced
        ):
            self._major_announced = True
            actions.append(MAJOR_MOTION)

        # ── evidence capture, rate-limited ────────────────────────
        if self.state is State.ALERT and scene.active:
            if self._may_snapshot(now):
                self.snapshots_taken += 1
                self._last_snapshot = now
                actions.append(TAKE_SNAPSHOT)
            if self._may_record(now):
                self.state = State.RECORDING
                self._clip_started = now
                actions.append(ENTER_RECORDING)

        # ── RECORDING → ALERT/IDLE when the clip window closes ────
        if (
            self.state is State.RECORDING
            and now - self._clip_started >= self.cfg.clip_seconds
        ):
            self._last_clip_end = now
            actions.append(EXIT_RECORDING)
            # Always fall back to ALERT; the hysteresis rule below owns the
            # alarm release, so a clip ending cannot short-circuit it.
            self.state = State.ALERT

        # ── ALERT → IDLE (hysteresis) ─────────────────────────────
        if (
            self.state is State.ALERT
            and not scene.active
            and calm > self.cfg.alarm_calm_seconds
        ):
            self.state = State.IDLE
            self.alarm_on = False
            actions += [ALARM_OFF, BACK_TO_IDLE]

        # ── budget reset after prolonged calm ─────────────────────
        if (
            not scene.active
            and calm > self.cfg.calm_reset_seconds
            and self.snapshots_taken
        ):
            self.snapshots_taken = 0
            self._major_announced = False

        return actions
