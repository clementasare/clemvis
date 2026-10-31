"""
motioniq — a console motion-detection & evidence system.

A small, dependency-light security camera built with OpenCV, NumPy and the
Python standard library. See README.md for the concepts and architecture.
"""

__version__ = "3.0.0"

from .config import PRESETS, DetectorConfig
from .detector import Assessment, MotionStateMachine, State, assess
from .pipeline import DetectionResult, VisionPipeline
from .tracking import Blob, CentroidTracker, Track
from .zones import Region, default_regions, parse_regions

__all__ = [
    "__version__",
    "DetectorConfig",
    "PRESETS",
    "VisionPipeline",
    "DetectionResult",
    "Blob",
    "CentroidTracker",
    "Track",
    "MotionStateMachine",
    "Assessment",
    "State",
    "assess",
    "Region",
    "default_regions",
    "parse_regions",
]
