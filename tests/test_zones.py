"""Zone tests — Feature 2 (regions of interest and severity)."""
import pytest

from clemvis.config import DetectorConfig
from clemvis.detector import assess
from clemvis.tracking import CentroidTracker, Blob
from clemvis.zones import Region, default_regions, parse_regions, region_at

W, H = 900, 600


def test_default_layout_is_three_vertical_thirds():
    regions = default_regions()
    assert [r.name for r in regions] == ["A", "B", "C"]
    assert region_at(regions, 50, 300, W, H).name == "A"
    assert region_at(regions, 450, 300, W, H).name == "B"
    assert region_at(regions, 850, 300, W, H).name == "C"


def test_restricted_zone_is_major_severity():
    assert default_regions()[2].severity == "major"


def test_pixel_rect_scales_with_resolution():
    region = Region("X", "Test", "normal", 0.5, 0.0, 1.0, 0.5)
    assert region.pixel_rect(800, 400) == (400, 0, 400, 200)


def test_parse_zone_spec_with_and_without_label():
    regions = parse_regions("A:Door:normal:0,0,0.5,1; B:major:0.5,0,1,1")
    assert regions[0].label == "Door" and regions[0].severity == "normal"
    assert regions[1].name == "B" and regions[1].severity == "major"


def test_bad_zone_specs_are_rejected():
    with pytest.raises(ValueError):
        parse_regions("A:normal:0,0,2,1")          # rect out of range
    with pytest.raises(ValueError):
        parse_regions("A:weird:0,0,1,1")           # unknown severity


def _confirmed_track_at(cx, cy):
    """Drive a tracker until one target at (cx, cy) is confirmed."""
    trk = CentroidTracker(confirm_frames=3, min_displacement=40)
    for i in range(8):
        x = cx - 120 + i * 20
        trk.update([Blob(int(x), int(cy - 50), 100, 100, 10000.0)], now=i * 0.05)
    return trk.confirmed


def test_motion_in_restricted_zone_escalates_to_major():
    cfg = DetectorConfig(width=W, height=H, major_area=10_000_000)
    scene = assess(_confirmed_track_at(820, 300), cfg, W, H)
    assert scene.active
    assert scene.zone_name == "C"
    assert scene.severity == "major"


def test_motion_in_normal_zone_stays_normal():
    cfg = DetectorConfig(width=W, height=H, major_area=10_000_000)
    scene = assess(_confirmed_track_at(430, 300), cfg, W, H)
    assert scene.zone_name == "B" and scene.severity == "normal"


def test_very_large_object_escalates_even_in_a_normal_zone():
    cfg = DetectorConfig(width=W, height=H, major_area=5000)
    scene = assess(_confirmed_track_at(430, 300), cfg, W, H)
    assert scene.severity == "major"


def test_zones_can_be_disabled():
    cfg = DetectorConfig(width=W, height=H, zones_enabled=False,
                         major_area=10_000_000)
    scene = assess(_confirmed_track_at(820, 300), cfg, W, H)
    assert scene.region is None and scene.severity == "normal"
