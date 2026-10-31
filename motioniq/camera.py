"""
Frame sources — where video comes from.

Two interchangeable sources sharing one tiny interface:
`read() -> (ok, frame)`, `timestamp() -> float`, `release()`.

`DemoSource` is the important one for this project: it renders a SCRIPTED,
deterministic scene containing both real intruders and the exact nuisances
that fooled version 1 — a fluttering curtain, a wobbling chair and a light
switch. Because the script is known in advance it doubles as GROUND TRUTH
for the evaluation mode (see evaluate.py): we know precisely when something
real happened, so we can count hits, misses and false alarms.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetectorConfig


@dataclass(slots=True)
class ScriptedEvent:
    """One scheduled happening in the synthetic scene."""

    start: float
    end: float
    kind: str          # intruder | curtain | chair | light
    label: str
    should_trigger: bool   # ground truth: must this raise an alert?
    # motion description (normalised coordinates)
    x_from: float = 0.0
    x_to: float = 0.0
    y: float = 0.5
    size: int = 130


def _intruder(start, end, label, x_from, x_to, y=0.5, size=130) -> ScriptedEvent:
    return ScriptedEvent(start, end, "intruder", label, True, x_from, x_to, y, size)


# ── the standard 56-second rehearsal ──────────────────────────────
# 4 real intrusions, 3 nuisances. A good run detects 4 and ignores 3.
STANDARD: list[ScriptedEvent] = [
    _intruder(3.0, 8.0, "person crosses left to right", 0.05, 0.85),
    ScriptedEvent(11.0, 19.0, "curtain", "curtain flutters by the window",
                  False, 0.20, 0.20, 0.45, 90),
    _intruder(22.0, 27.0, "person crosses right to left", 0.85, 0.05),
    ScriptedEvent(30.0, 33.0, "light", "ceiling light switched on",
                  False, 0.0, 0.0, 0.0, 0),
    _intruder(36.0, 41.0, "person enters restricted zone", 0.55, 0.92),
    ScriptedEvent(44.0, 49.0, "chair", "office chair wobbles",
                  False, 0.45, 0.45, 0.70, 80),
    _intruder(52.0, 56.0, "person walks top to bottom", 0.45, 0.45, 0.5, 130),
]

# ── a 24-second version for unit tests / quick checks ─────────────
QUICK: list[ScriptedEvent] = [
    _intruder(2.0, 7.0, "person crosses left to right", 0.05, 0.85),
    ScriptedEvent(10.0, 16.0, "curtain", "curtain flutters",
                  False, 0.25, 0.25, 0.45, 90),
    _intruder(18.0, 23.0, "person crosses right to left", 0.85, 0.05),
]

SCENARIOS: dict[str, list[ScriptedEvent]] = {
    "standard": STANDARD,
    "quick": QUICK,
}


class CameraSource:
    """The real webcam. Thin wrapper around cv2.VideoCapture."""

    def __init__(self, config: DetectorConfig) -> None:
        self.cap = cv2.VideoCapture(config.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"camera {config.camera_index} unavailable "
                "(in use by another app, or no camera attached?)"
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        # A small buffer keeps latency low so the view is genuinely live.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.cap.read()
        return ok, frame if ok else None

    def timestamp(self) -> float:
        return time.time()

    def release(self) -> None:
        self.cap.release()


class DemoSource:
    """
    Synthetic camera: sensor noise + scripted objects.

    realtime=True  → frames are paced to demo_fps (watch it like a camera)
    realtime=False → a virtual clock runs as fast as the CPU allows
                     (used by evaluation and tests: deterministic, quick)
    """

    def __init__(self, config: DetectorConfig, seed: int = 11) -> None:
        self.width = config.width
        self.height = config.height
        self.fps = max(1.0, config.demo_fps)
        self.realtime = config.demo_realtime
        self.script = SCENARIOS.get(config.demo_scenario, STANDARD)
        self.duration = max(e.end for e in self.script) + 4.0
        self._rng = np.random.default_rng(seed)
        self._frame_idx = 0
        self._virtual_t = 0.0
        self._wall_start = time.monotonic()
        self._epoch = time.time()

    # ── ground truth helpers (used by evaluate.py) ────────────────
    @property
    def expected_events(self) -> list[ScriptedEvent]:
        return [e for e in self.script if e.should_trigger]

    @property
    def distractors(self) -> list[ScriptedEvent]:
        return [e for e in self.script if not e.should_trigger]

    def timestamp(self) -> float:
        """Wall clock in realtime mode, virtual clock otherwise."""
        return time.time() if self.realtime else self._epoch + self._virtual_t

    # ── frame synthesis ───────────────────────────────────────────
    def _render(self, t: float) -> np.ndarray:
        w, h = self.width, self.height
        ambient = 70.0
        # a light switch raises the whole scene's brightness
        for e in self.script:
            if e.kind == "light" and e.start <= t < e.end:
                ambient = 150.0

        gray = self._rng.normal(ambient, 2.0, (h, w)).astype(np.float32)

        for e in self.script:
            if not (e.start <= t < e.end) or e.kind == "light":
                continue
            progress = (t - e.start) / max(1e-6, e.end - e.start)

            if e.kind == "intruder":
                # travels steadily across the frame → large net displacement
                nx = e.x_from + (e.x_to - e.x_from) * progress
                ny = e.y
                if abs(e.x_to - e.x_from) < 0.01:      # vertical walker
                    ny = 0.15 + 0.7 * progress
            elif e.kind == "curtain":
                # oscillates ±18 px around a fixed point → net travel ≈ 0
                nx = e.x_from + (18.0 / w) * math.sin(progress * 14.0)
                ny = e.y
            else:  # chair
                nx = e.x_from + (8.0 / w) * math.sin(progress * 20.0)
                ny = e.y + (6.0 / h) * math.cos(progress * 17.0)

            s = e.size
            x = int(nx * w)
            y = int(ny * h)
            cv2.rectangle(gray, (x, y), (x + s, y + s), 205.0, -1)

        frame = np.clip(gray, 0, 255).astype(np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def read(self) -> tuple[bool, np.ndarray | None]:
        t = self._frame_idx / self.fps
        if t > self.duration:
            return False, None
        frame = self._render(t)
        self._frame_idx += 1
        self._virtual_t = t

        if self.realtime:
            behind = t - (time.monotonic() - self._wall_start)
            if behind > 0:
                time.sleep(behind)
        return True, frame

    def release(self) -> None:
        pass
