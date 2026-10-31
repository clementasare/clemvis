"""
Feature 1 — centroid tracking and motion direction.

For every blob the pipeline finds we compute its centre point (centroid) and
follow it across frames. Matching is nearest-neighbour: each new centroid is
linked to the closest existing track within `track_max_distance` pixels.
That is the simplest tracker that works, and it is easy to defend in an
interview (the alternative, the Hungarian algorithm, is overkill here).

Tracking buys us three things at once:

1. DIRECTION  — "LEFT → RIGHT", "UP → DOWN", … reported and drawn as an arrow.
2. CONFIRMATION — a track must survive `confirm_frames` frames before anyone
   is allowed to raise an alarm, which filters single-frame noise flashes.
3. TRAVEL GATE — a track must move `min_displacement` px away from where it
   first appeared. THIS is what rejects curtains, fans, hanging cables and
   wobbling chairs: they oscillate around a fixed origin, so their net
   displacement never grows, no matter how long they flap.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

STATIONARY = "STATIONARY"


@dataclass(slots=True)
class Blob:
    """A raw detection from the vision pipeline, in full-frame pixels."""

    x: int
    y: int
    w: int
    h: int
    area: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass(slots=True)
class Track:
    """One object followed over time."""

    id: int
    cx: float
    cy: float
    origin_cx: float
    origin_cy: float
    box: tuple[int, int, int, int]
    area: float
    age: int = 1                 # frames seen
    missing: int = 0             # consecutive frames not matched
    first_seen: float = 0.0
    last_seen: float = 0.0
    history: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=12)
    )

    # ── geometry ─────────────────────────────────────────────────
    @property
    def displacement(self) -> float:
        """Straight-line distance from where this track first appeared."""
        return math.hypot(self.cx - self.origin_cx, self.cy - self.origin_cy)

    @property
    def velocity(self) -> tuple[float, float]:
        """Average px/frame over the recent history window."""
        if len(self.history) < 2:
            return 0.0, 0.0
        x0, y0 = self.history[0]
        x1, y1 = self.history[-1]
        span = len(self.history) - 1
        return (x1 - x0) / span, (y1 - y0) / span

    @property
    def direction(self) -> str:
        """Human-readable dominant direction of travel."""
        if len(self.history) < 2:
            return STATIONARY
        x0, y0 = self.history[0]
        dx = self.cx - x0
        dy = self.cy - y0
        if math.hypot(dx, dy) < 12:          # jitter, not travel
            return STATIONARY
        if abs(dx) >= abs(dy):
            return "LEFT → RIGHT" if dx > 0 else "RIGHT → LEFT"
        return "UP → DOWN" if dy > 0 else "DOWN → UP"

    @property
    def arrow(self) -> tuple[int, int]:
        """Unit-ish vector for drawing the direction arrow on the HUD."""
        vx, vy = self.velocity
        mag = math.hypot(vx, vy)
        if mag < 0.4:
            return 0, 0
        return int(round(vx / mag)), int(round(vy / mag))


class CentroidTracker:
    """Nearest-neighbour multi-object tracker (no external dependencies)."""

    def __init__(
        self,
        max_distance: int = 180,
        ttl_frames: int = 10,
        confirm_frames: int = 5,
        min_displacement: int = 60,
    ) -> None:
        self.max_distance = max_distance
        self.ttl_frames = ttl_frames
        self.confirm_frames = confirm_frames
        self.min_displacement = min_displacement
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(self, blobs: list[Blob], now: float) -> list[Track]:
        """Advance all tracks by one frame; returns the live track list."""
        unmatched = list(blobs)

        # 1 ─ match existing tracks to the nearest unclaimed blob
        for track in self.tracks:
            best: Blob | None = None
            best_dist = float("inf")
            for blob in unmatched:
                dist = math.hypot(blob.cx - track.cx, blob.cy - track.cy)
                if dist < best_dist:
                    best, best_dist = blob, dist
            if best is not None and best_dist <= self.max_distance:
                unmatched.remove(best)
                track.history.append((track.cx, track.cy))
                track.cx, track.cy = best.cx, best.cy
                track.box = (best.x, best.y, best.w, best.h)
                track.area = best.area
                track.age += 1
                track.missing = 0
                track.last_seen = now
            else:
                track.missing += 1

        # 2 ─ leftover blobs become new tracks
        for blob in unmatched:
            self.tracks.append(
                Track(
                    id=self._next_id,
                    cx=blob.cx,
                    cy=blob.cy,
                    origin_cx=blob.cx,
                    origin_cy=blob.cy,
                    box=(blob.x, blob.y, blob.w, blob.h),
                    area=blob.area,
                    first_seen=now,
                    last_seen=now,
                )
            )
            self._next_id += 1

        # 3 ─ retire tracks that have been missing too long
        self.tracks = [t for t in self.tracks if t.missing <= self.ttl_frames]
        return self.tracks

    def is_confirmed(self, track: Track) -> bool:
        """Old enough AND actually travelled — the anti-false-alarm test."""
        return (
            track.age >= self.confirm_frames
            and track.missing == 0
            and track.displacement >= self.min_displacement
        )

    @property
    def confirmed(self) -> list[Track]:
        """Live tracks that passed both the time and travel gates."""
        out = [t for t in self.tracks if self.is_confirmed(t)]
        out.sort(key=lambda t: t.area, reverse=True)
        return out
