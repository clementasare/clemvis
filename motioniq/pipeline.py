"""
The vision pipeline — pure frame-in, blobs-out (no decisions made here).

    frame → downscale → grayscale → |frame − background| → Gaussian blur
          → binary threshold → dilate → contours → blobs

Why each stage exists (plain language):

* downscale      — vision runs on a ~480 px-wide copy. Four times fewer
*                  pixels ⇒ roughly four times faster, and motion of a
*                  person is still obvious. Coordinates are scaled back up,
*                  so every threshold stays in real full-frame pixels.
* grayscale      — colour is irrelevant for motion; 1 channel, not 3
* background     — we subtract a *learned average of the empty scene*, not
*   subtraction    the previous frame, so moving objects leave no ghost trail
* Gaussian blur  — erases single-pixel sensor noise before it can cross the
*                  threshold and become a fake detection
* threshold      — turns "how much did this pixel change" into yes/no
* dilate         — merges the fragments of one object into one blob
* contours       — OpenCV traces blob outlines; boundingRect gives the boxes

The background only learns while the scene is calm, so somebody standing
still does not slowly dissolve into the background and vanish.

NOTE: this module intentionally knows nothing about alarms, zones or
tracking. It reports *what changed*; `tracking.py` decides what is real and
`detector.py` decides what to do about it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import DetectorConfig
from .tracking import Blob


@dataclass(slots=True)
class DetectionResult:
    blobs: list[Blob] = field(default_factory=list)
    total_area: float = 0.0      # summed blob area, full-frame px²
    foreground_pixels: int = 0   # raw changed pixels (downscaled space)
    foreground_ratio: float = 0.0
    saturated: bool = False      # global change: light switch / exposure jump
    warming_up: bool = True      # background model still settling


class VisionPipeline:
    """Stateful: owns the background model. One instance per camera."""

    def __init__(self, config: DetectorConfig) -> None:
        self.cfg = config
        self._background: np.ndarray | None = None
        self._frames_seen = 0

    def reset(self) -> None:
        """Forget the background and re-learn it (operator key: [r])."""
        self._background = None
        self._frames_seen = 0

    @property
    def warming_up(self) -> bool:
        return self._frames_seen < self.cfg.warmup_frames

    def process(self, frame_bgr: np.ndarray) -> DetectionResult:
        cfg = self.cfg
        full_h, full_w = frame_bgr.shape[:2]

        # ── 0. downscale for speed ────────────────────────────────
        scale = min(1.0, cfg.process_width / float(full_w))
        if scale < 1.0:
            small = cv2.resize(
                frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        else:
            small = frame_bgr

        # BGR2GRAY. (The original script used COLOR_RGB2GRAY on a BGR frame,
        # which silently swaps the red/blue weights — fixed here.)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (cfg.blur_kernel, cfg.blur_kernel), 0)

        self._frames_seen += 1

        if self._background is None:
            self._background = gray.astype(np.float32)
            return DetectionResult(warming_up=True)

        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self._background))
        _, thresh = cv2.threshold(
            diff, cfg.diff_threshold, 255, cv2.THRESH_BINARY
        )

        foreground_pixels = int(cv2.countNonZero(thresh))
        foreground_ratio = foreground_pixels / float(thresh.size)
        saturated = foreground_ratio >= cfg.saturation_ratio

        dilated = cv2.dilate(thresh, None, iterations=cfg.dilate_iterations)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ── background learning gate ──────────────────────────────
        warming = self.warming_up
        if warming:
            # Settle fast during warm-up so start-up transients vanish.
            cv2.accumulateWeighted(gray, self._background, 0.25)
        elif saturated:
            # Lights changed: re-learn quickly instead of alarming forever.
            cv2.accumulateWeighted(gray, self._background, 0.40)
        elif foreground_ratio < 0.02:
            # Calm: drift slowly to absorb daylight and shadow movement.
            cv2.accumulateWeighted(gray, self._background, cfg.background_alpha)
        # else: real movement in progress → freeze the model.

        if warming or saturated:
            return DetectionResult(
                foreground_pixels=foreground_pixels,
                foreground_ratio=foreground_ratio,
                saturated=saturated,
                warming_up=warming,
            )

        # ── contours → blobs, converted back to full-frame pixels ──
        inv = 1.0 / scale if scale > 0 else 1.0
        area_scale = inv * inv
        blobs: list[Blob] = []
        total_area = 0.0
        for contour in contours:
            area_full = cv2.contourArea(contour) * area_scale
            if area_full < cfg.min_contour_area:
                continue  # too small — noise, insects, distant leaves
            x, y, w, h = cv2.boundingRect(contour)
            blobs.append(
                Blob(
                    x=int(x * inv),
                    y=int(y * inv),
                    w=int(w * inv),
                    h=int(h * inv),
                    area=area_full,
                )
            )
            total_area += area_full
        blobs.sort(key=lambda b: b.area, reverse=True)

        return DetectionResult(
            blobs=blobs,
            total_area=total_area,
            foreground_pixels=foreground_pixels,
            foreground_ratio=foreground_ratio,
            saturated=False,
            warming_up=False,
        )
