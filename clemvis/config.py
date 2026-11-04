"""
Central configuration.

Every tunable lives in ONE dataclass instead of being a magic number buried
in the main loop. CLI flags override these defaults; presets capture tested
combinations for common environments.

The defaults are deliberately CONSERVATIVE: it is far better to miss a
borderline event than to cry wolf at a curtain. Three independent filters
must all agree before anything is reported:

    1. size      — the blob must be big enough          (min_contour_area)
    2. time      — it must persist for several frames   (confirm_frames)
    3. travel    — it must actually go somewhere        (min_displacement)

Filter 3 is what kills curtains, fans and wobbling chairs: they oscillate
around a fixed point, so their NET displacement stays near zero forever.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from .zones import Region, default_regions


@dataclass(slots=True)
class DetectorConfig:
    # ── video source ──────────────────────────────────────────────
    camera_index: int = 0
    width: int = 1280              # capture resolution requested from camera
    height: int = 720
    process_width: int = 480       # vision runs on a downscaled copy → fast.
                                   # Detection coords are scaled back up, so
                                   # all thresholds stay in full-frame pixels.

    # ── vision pipeline ───────────────────────────────────────────
    blur_kernel: int = 5           # Gaussian blur size — smooths sensor noise
    diff_threshold: int = 32       # brightness change that counts (0-255)
    dilate_iterations: int = 2     # merges fragments of one object
    min_contour_area: int = 4000   # px² — ignore anything smaller
    major_area: int = 45000        # px² — "large object" escalation

    # ── adaptive background model ─────────────────────────────────
    background_alpha: float = 0.10  # learning rate while the scene is calm
    saturation_ratio: float = 0.35  # ≥35% pixels changed ⇒ light switch
    warmup_frames: int = 40         # settle the model before judging anything

    # ── temporal + travel filters (the false-alarm killers) ───────
    confirm_frames: int = 5         # must persist this many frames…
    min_displacement: int = 60      # …and travel this many px from its origin
    track_max_distance: int = 180   # max px a target may jump between frames
    track_ttl_frames: int = 10      # keep a lost track alive this long

    # ── state machine timing ──────────────────────────────────────
    alarm_calm_seconds: float = 3.0    # alarm releases after this much calm
    calm_reset_seconds: float = 15.0   # snapshot budget resets after this
    clip_seconds: float = 8.0          # evidence clip length
    clip_fps: float = 20.0

    # ── capture budget (stops the "100 files in 10 seconds" problem) ──
    clip_cooldown_seconds: float = 30.0      # min gap between two clips
    snapshots_per_burst: int = 1             # one photo per intrusion…
    snapshot_cooldown_seconds: float = 10.0  # …and never faster than this

    # ── detection zones (Feature 2) ───────────────────────────────
    zones_enabled: bool = True
    regions: list[Region] = field(default_factory=default_regions)

    # ── output ────────────────────────────────────────────────────
    save_dir: Path = Path("detections")
    db_path: Path | None = None
    alarm_enabled: bool = True
    show_window: bool = True
    annotate_evidence: bool = False   # False ⇒ CLEAN evidence (timestamp only)
    demo: bool = False
    demo_scenario: str = "standard"
    demo_fps: float = 20.0
    demo_realtime: bool = True
    duration: float | None = None

    def __post_init__(self) -> None:
        # cv2.GaussianBlur needs an odd kernel — normalize instead of
        # crashing deep inside the loop.
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1
        if self.min_contour_area >= self.major_area:
            raise ValueError(
                "min_contour_area must stay below major_area "
                f"({self.min_contour_area} !< {self.major_area})"
            )
        if self.process_width < 160:
            raise ValueError("process_width below 160 px destroys accuracy")

    @property
    def db_file(self) -> Path:
        """Ledger database location (inside save_dir unless overridden)."""
        return self.db_path if self.db_path else self.save_dir / "clemvis.db"

    @property
    def active_regions(self) -> list[Region]:
        return self.regions if self.zones_enabled else []


# Tested combinations. A preset name replaces half a dozen numbers and
# keeps tuning explainable in an interview.
PRESETS: dict[str, dict] = {
    # Living room / office: curtains, fans, screens, pets.
    "indoor": {
        "diff_threshold": 32,
        "min_contour_area": 4000,
        "major_area": 45000,
        "confirm_frames": 5,
        "min_displacement": 60,
        "background_alpha": 0.10,
    },
    # Garden / driveway: wind, foliage, shadows, weather.
    "outdoor": {
        "diff_threshold": 42,
        "min_contour_area": 7000,
        "major_area": 60000,
        "confirm_frames": 7,
        "min_displacement": 90,
        "background_alpha": 0.05,
    },
    # Deliberately twitchy: demos, small/slow subjects, testing.
    "sensitive": {
        "diff_threshold": 22,
        "min_contour_area": 1800,
        "major_area": 25000,
        "confirm_frames": 3,
        "min_displacement": 30,
        "background_alpha": 0.12,
    },
}


def apply_preset(config: DetectorConfig, name: str) -> DetectorConfig:
    """Return a copy of `config` with preset values applied."""
    if name not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown preset {name!r} (choose from: {known})")
    return replace(config, **PRESETS[name])
