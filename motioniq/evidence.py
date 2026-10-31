"""
Evidence recorder — snapshots (JPEG) and clips (AVI/MJPEG).

Design decision: evidence is saved CLEAN. The frames handed to this module
carry only a small UTC timestamp (see hud.stamp), not the debugging overlay,
because the point of evidence is to show the scene — not the detector's
internal state. `--annotate-evidence` restores the full overlay when you are
tuning or recording a demo for the README.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import DetectorConfig

log = logging.getLogger("motioniq.evidence")


def _stamp_name() -> str:
    # millisecond precision keeps rapid filenames unique
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


class EvidenceRecorder:
    def __init__(self, config: DetectorConfig) -> None:
        self.cfg = config
        self.cfg.save_dir.mkdir(parents=True, exist_ok=True)
        self._writer: cv2.VideoWriter | None = None
        self._clip_path: Path | None = None
        self.frames_written = 0

    @property
    def recording(self) -> bool:
        return self._writer is not None

    @property
    def clip_path(self) -> Path | None:
        return self._clip_path

    def save_snapshot(self, frame: np.ndarray) -> Path:
        path = self.cfg.save_dir / f"snapshot_{_stamp_name()}.jpg"
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return path

    def start_clip(self, frame: np.ndarray) -> Path | None:
        """
        Begin a clip; returns its path, or None if the writer failed.

        v1 never called `isOpened()` — a missing codec produced zero-byte
        evidence with no warning. We check, report, and keep running.
        """
        if self._writer is not None:
            return self._clip_path
        h, w = frame.shape[:2]
        path = self.cfg.save_dir / f"movement_{_stamp_name()}.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), self.cfg.clip_fps, (w, h)
        )
        if not writer.isOpened():
            writer.release()
            log.error("VideoWriter failed to open — clip skipped (codec?)")
            return None
        self._writer = writer
        self._clip_path = path
        self.frames_written = 0
        return path

    def write_clip(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)
            self.frames_written += 1

    def stop_clip(self) -> Path | None:
        """Flush + release. MUST be called or the AVI stays corrupt."""
        path = self._clip_path
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._clip_path = None
        return path
