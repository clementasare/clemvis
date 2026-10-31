"""
Heads-up display drawing (plain cv2 primitives, no extra libraries).

Two very different outputs, on purpose:

* `annotate()`  — the rich LIVE view: zone outlines, target boxes, direction
                  arrows, state banner, telemetry. Useful while operating.
* `stamp()`     — the CLEAN evidence overlay: nothing but a small UTC
                  timestamp in the corner.

Saved snapshots and clips use `stamp()` by default, so the footage shows the
scene itself instead of a wall of boxes and text — you can actually see who
was there. Pass --annotate-evidence if you want the debug overlay baked in.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import cv2
import numpy as np

from .detector import Assessment, State
from .tracking import Track
from .zones import Region

# OpenCV colours are BGR, not RGB (classic gotcha).
GREEN = (120, 220, 140)
AMBER = (60, 190, 240)
RED = (80, 80, 235)
WHITE = (240, 240, 240)
GRAY = (160, 160, 160)
DIM = (90, 90, 90)

STATE_COLORS = {
    State.IDLE: GREEN,
    State.ALERT: AMBER,
    State.RECORDING: RED,
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def stamp(frame: np.ndarray) -> np.ndarray:
    """Minimal forensic overlay for saved evidence: UTC time, nothing else."""
    text = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    h = frame.shape[0]
    # thin dark outline keeps the text readable on any background
    cv2.putText(frame, text, (12, h - 12), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (12, h - 12), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    return frame


def _draw_zones(frame: np.ndarray, regions: list[Region]) -> None:
    h, w = frame.shape[:2]
    for region in regions:
        x, y, rw, rh = region.pixel_rect(w, h)
        color = RED if region.severity == "major" else (
            DIM if region.severity == "ignore" else GRAY
        )
        # dashed-looking top/bottom edges keep the view uncluttered
        cv2.line(frame, (x, y), (x + rw, y), color, 1)
        cv2.line(frame, (x, y + rh - 1), (x + rw, y + rh - 1), color, 1)
        cv2.line(frame, (x, y), (x, y + rh), color, 1)
        tag = f"{region.name}  {region.label}"
        if region.severity != "normal":
            tag += f"  [{region.severity.upper()}]"
        cv2.putText(frame, tag, (x + 8, y + 20), FONT, 0.45, color, 1, cv2.LINE_AA)


def _draw_target(frame: np.ndarray, track: Track, color) -> None:
    x, y, w, h = track.box
    # corner brackets instead of a full rectangle — far less visual noise
    L = max(12, min(w, h) // 5)
    for (ax, ay), (bx, by) in (
        ((x, y), (x + L, y)), ((x, y), (x, y + L)),
        ((x + w, y), (x + w - L, y)), ((x + w, y), (x + w, y + L)),
        ((x, y + h), (x + L, y + h)), ((x, y + h), (x, y + h - L)),
        ((x + w, y + h), (x + w - L, y + h)), ((x + w, y + h), (x + w, y + h - L)),
    ):
        cv2.line(frame, (ax, ay), (bx, by), color, 2)

    cx, cy = int(track.cx), int(track.cy)
    cv2.circle(frame, (cx, cy), 3, color, -1)

    # direction arrow from the centroid
    ax, ay = track.arrow
    if ax or ay:
        tip = (cx + ax * 60, cy + ay * 60)
        cv2.arrowedLine(frame, (cx, cy), tip, color, 2, tipLength=0.3)

    label = f"T{track.id} {track.direction}"
    ly = y - 8 if y > 24 else y + h + 18
    cv2.putText(frame, label, (x, ly), FONT, 0.45, color, 1, cv2.LINE_AA)


def annotate(
    frame: np.ndarray,
    *,
    state: State,
    scene: Assessment,
    tracks: list[Track],
    regions: list[Region],
    fps: float,
    recording: bool,
    muted: bool,
    warming_up: bool = False,
) -> np.ndarray:
    """Draw the full live HUD in-place and return the frame."""
    if regions:
        _draw_zones(frame, regions)

    color = RED if scene.severity == "major" else AMBER
    for track in tracks:
        _draw_target(frame, track, color)

    h, w = frame.shape[:2]
    state_color = STATE_COLORS.get(state, GRAY)

    banner = "WARMING UP" if warming_up else state.name
    cv2.putText(frame, f"[{banner}]", (12, 26), FONT, 0.7,
                GRAY if warming_up else state_color, 2, cv2.LINE_AA)

    if scene.active:
        info = (
            f"Zone {scene.zone_name} ({scene.zone_label})  "
            f"{scene.severity.upper()}  {scene.direction}"
        )
        cv2.putText(frame, info, (12, 50), FONT, 0.5, color, 1, cv2.LINE_AA)

    if recording and int(time.time() * 2) % 2 == 0:
        cv2.circle(frame, (w - 72, 20), 6, RED, -1)
        cv2.putText(frame, "REC", (w - 58, 26), FONT, 0.6, RED, 2, cv2.LINE_AA)

    stamp(frame)

    telemetry = f"{fps:5.1f} fps | targets {scene.target_count}"
    if muted:
        telemetry += " | MUTED"
    cv2.putText(frame, telemetry, (w - 250, h - 12), FONT, 0.45, GRAY, 1,
                cv2.LINE_AA)
    return frame
