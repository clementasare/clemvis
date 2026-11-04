"""
Tracking tests — Feature 1, and the false-alarm filter that matters most.

These encode the exact complaint that motivated v3: a curtain or a chair
moves constantly but never goes anywhere, so it must never be confirmed.
"""
import math

from clemvis.tracking import Blob, CentroidTracker


def tracker(**kw) -> CentroidTracker:
    base = dict(max_distance=180, ttl_frames=5, confirm_frames=4,
                min_displacement=60)
    base.update(kw)
    return CentroidTracker(**base)


def blob_at(cx, cy, size=100) -> Blob:
    return Blob(int(cx - size / 2), int(cy - size / 2), size, size, float(size * size))


def test_single_object_keeps_one_stable_id():
    trk = tracker()
    for i in range(6):
        trk.update([blob_at(100 + i * 20, 200)], now=i * 0.05)
    assert len(trk.tracks) == 1
    assert trk.tracks[0].id == 1
    assert trk.tracks[0].age == 6


def test_walker_is_confirmed_once_it_has_travelled():
    trk = tracker(confirm_frames=4, min_displacement=60)
    confirmed_at = None
    for i in range(10):
        trk.update([blob_at(100 + i * 25, 200)], now=i * 0.05)
        if trk.confirmed and confirmed_at is None:
            confirmed_at = i
    assert confirmed_at is not None
    # needs BOTH gates: 4 frames of persistence and 60 px of travel
    assert confirmed_at >= 3


def test_curtain_oscillation_is_never_confirmed():
    """The headline regression test: swaying motion must stay silent."""
    trk = tracker(min_displacement=60)
    for i in range(200):
        x = 300 + 18 * math.sin(i * 0.5)   # ±18 px forever
        trk.update([blob_at(x, 240)], now=i * 0.05)
        assert trk.confirmed == [], f"curtain confirmed at frame {i}"


def test_chair_wobble_is_never_confirmed():
    trk = tracker(min_displacement=60)
    for i in range(200):
        x = 400 + 8 * math.sin(i * 0.7)
        y = 300 + 6 * math.cos(i * 0.6)
        trk.update([blob_at(x, y)], now=i * 0.05)
        assert trk.confirmed == []


def test_single_frame_flash_is_never_confirmed():
    trk = tracker()
    trk.update([blob_at(100, 100)], now=0.0)
    for i in range(1, 5):
        trk.update([], now=i * 0.05)
    assert trk.confirmed == []


def test_direction_labels():
    trk = tracker()
    for i in range(8):
        trk.update([blob_at(100 + i * 30, 200)], now=i * 0.05)
    assert trk.tracks[0].direction == "LEFT → RIGHT"

    trk = tracker()
    for i in range(8):
        trk.update([blob_at(500 - i * 30, 200)], now=i * 0.05)
    assert trk.tracks[0].direction == "RIGHT → LEFT"

    trk = tracker()
    for i in range(8):
        trk.update([blob_at(300, 100 + i * 30)], now=i * 0.05)
    assert trk.tracks[0].direction == "UP → DOWN"

    trk = tracker()
    for i in range(8):
        trk.update([blob_at(300, 400 - i * 30)], now=i * 0.05)
    assert trk.tracks[0].direction == "DOWN → UP"


def test_stationary_object_reports_stationary():
    trk = tracker()
    for i in range(8):
        trk.update([blob_at(300, 200)], now=i * 0.05)
    assert trk.tracks[0].direction == "STATIONARY"


def test_lost_track_is_retired_after_ttl():
    trk = tracker(ttl_frames=3)
    trk.update([blob_at(100, 100)], now=0.0)
    for i in range(1, 6):
        trk.update([], now=i * 0.05)
    assert trk.tracks == []


def test_two_objects_get_separate_ids():
    trk = tracker()
    for i in range(5):
        trk.update(
            [blob_at(100 + i * 20, 150), blob_at(600 - i * 20, 400)],
            now=i * 0.05,
        )
    assert len({t.id for t in trk.tracks}) == 2
