"""
Pipeline tests — synthetic NumPy frames, no camera involved.

The vision code never touches hardware directly, which is exactly why it is
testable. Each test hands the pipeline crafted frames and checks the verdict.
"""
import cv2
import numpy as np

from motioniq.config import DetectorConfig
from motioniq.pipeline import VisionPipeline

W, H = 640, 360


def make_config(**overrides) -> DetectorConfig:
    base = dict(
        width=W,
        height=H,
        process_width=320,
        diff_threshold=25,
        min_contour_area=1500,
        major_area=20000,
        warmup_frames=3,
    )
    base.update(overrides)
    return DetectorConfig(**base)


def blank() -> np.ndarray:
    return np.full((H, W, 3), 70, np.uint8)


def with_blob(x=60, y=60, s=90) -> np.ndarray:
    f = blank()
    cv2.rectangle(f, (x, y), (x + s, y + s), (235, 235, 235), -1)
    return f


def warm(pipe: VisionPipeline, frames: int = 6) -> None:
    """Feed calm frames until the background model has settled."""
    for _ in range(frames):
        pipe.process(blank())


def test_warmup_suppresses_everything():
    pipe = VisionPipeline(make_config(warmup_frames=5))
    for _ in range(4):
        r = pipe.process(with_blob())
        assert r.warming_up and r.blobs == []


def test_calm_scene_reports_nothing():
    pipe = VisionPipeline(make_config())
    warm(pipe)
    for _ in range(5):
        assert pipe.process(blank()).blobs == []


def test_blob_is_found_with_area_and_box_in_full_frame_coords():
    pipe = VisionPipeline(make_config())
    warm(pipe)
    r = pipe.process(with_blob(x=60, y=60, s=90))
    assert len(r.blobs) >= 1
    b = r.blobs[0]
    # 90x90 blob ≈ 8100 px² — reported in FULL-frame pixels despite the
    # pipeline running at process_width=320
    assert b.area > 4000
    assert 30 <= b.x <= 90 and 30 <= b.y <= 90
    assert b.w > 50 and b.h > 50


def test_small_blob_is_rejected_by_area_floor():
    pipe = VisionPipeline(make_config(min_contour_area=20000, major_area=90000))
    warm(pipe)
    assert pipe.process(with_blob(s=60)).blobs == []


def test_light_switch_is_flagged_saturated_then_recovers():
    pipe = VisionPipeline(make_config())
    warm(pipe)
    bright = np.full((H, W, 3), 255, np.uint8)
    r = pipe.process(bright)
    assert r.saturated and r.blobs == []
    for _ in range(12):
        r = pipe.process(bright)
    assert not r.saturated and r.blobs == []


def test_reset_restarts_the_warmup():
    pipe = VisionPipeline(make_config(warmup_frames=3))
    warm(pipe)
    pipe.reset()
    assert pipe.process(blank()).warming_up
