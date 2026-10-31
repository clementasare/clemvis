"""
SurveillanceApp — the main loop that wires every component together.

    source → pipeline → tracker → assess → FSM ──actions──▶ alarm
                           │         │                      evidence
                           │         │                      ledger
                           └─────────┴──▶ hud.annotate ──▶ live window

The loop only ORCHESTRATES; every decision was already made (and unit
tested) in pipeline/tracking/detector. Teardown lives in `finally`, so
Ctrl+C can never strand the camera or leave a half-written clip.
"""
from __future__ import annotations

import logging
import sys
import time

import cv2

from . import __version__, hud
from .alarm import Alarm
from .camera import CameraSource, DemoSource
from .config import DetectorConfig
from .detector import (
    ALARM_OFF,
    ALARM_ON,
    BACK_TO_IDLE,
    ENTER_RECORDING,
    EXIT_RECORDING,
    MAJOR_MOTION,
    MOTION_STARTED,
    TAKE_SNAPSHOT,
    Assessment,
    MotionStateMachine,
    State,
    assess,
)
from .evidence import EvidenceRecorder
from .ledger import EventLedger
from .pipeline import VisionPipeline
from .tracking import CentroidTracker
from .zones import region_at

log = logging.getLogger("motioniq.app")

WINDOW_NAME = "motioniq - CAM 00"


class SurveillanceApp:
    def __init__(self, config: DetectorConfig) -> None:
        self.cfg = config

    def _banner(self) -> None:
        c = self.cfg
        src = "synthetic demo" if c.demo else f"camera #{c.camera_index}"
        print("=" * 60)
        print(f"  motioniq v{__version__} — console motion surveillance")
        print("=" * 60)
        print(f"  source      : {src} ({c.width}x{c.height}, "
              f"vision @ {c.process_width}px)")
        print(f"  filters     : area>={c.min_contour_area}px  "
              f"persist>={c.confirm_frames}f  travel>={c.min_displacement}px")
        print(f"  capture     : 1 clip / {c.clip_cooldown_seconds:.0f}s, "
              f"{c.snapshots_per_burst} snap / {c.snapshot_cooldown_seconds:.0f}s"
              f"  ({'annotated' if c.annotate_evidence else 'clean'})")
        if c.zones_enabled:
            zones = "  ".join(
                f"{r.name}:{r.severity}" for r in c.regions
            )
            print(f"  zones       : {zones}")
        print(f"  ledger      : {c.db_file}")
        print(f"  alarm       : {'on' if c.alarm_enabled else 'off'}"
              f"{'' if c.show_window else '   headless'}")
        if c.show_window:
            print("  keys        : [q] quit  [s] snapshot  [r] reset bg  "
                  "[m] mute  [z] zones")
        print("-" * 60)

    def run(self) -> int:
        cfg = self.cfg
        source = DemoSource(cfg) if cfg.demo else CameraSource(cfg)
        pipeline = VisionPipeline(cfg)
        tracker = CentroidTracker(
            max_distance=cfg.track_max_distance,
            ttl_frames=cfg.track_ttl_frames,
            confirm_frames=cfg.confirm_frames,
            min_displacement=cfg.min_displacement,
        )
        fsm = MotionStateMachine(cfg)
        alarm = Alarm(enabled=cfg.alarm_enabled)
        recorder = EvidenceRecorder(cfg)
        ledger = EventLedger(cfg.db_file)

        muted = False
        show_zones = cfg.zones_enabled
        events_logged = 0
        fps = 0.0
        last_tick = time.monotonic()
        started = last_tick

        def log_event(kind: str, **kw) -> None:
            nonlocal events_logged
            ledger.log(kind, **kw)
            events_logged += 1

        self._banner()
        log_event("system", note="armed (demo)" if cfg.demo else "armed")

        try:
            while True:
                ok, frame = source.read()
                if not ok or frame is None:
                    if cfg.demo:
                        break
                    log.warning("frame grab failed — retrying")
                    time.sleep(0.05)
                    continue

                now = source.timestamp()
                result = pipeline.process(frame)
                h, w = frame.shape[:2]

                # ── drop motion inside 'ignore' zones before tracking ──
                blobs = result.blobs
                if cfg.zones_enabled:
                    kept = []
                    for b in blobs:
                        region = region_at(cfg.active_regions, b.cx, b.cy, w, h)
                        if region is not None and region.severity == "ignore":
                            continue
                        kept.append(b)
                    blobs = kept

                tracker.update(blobs, now)
                confirmed = tracker.confirmed
                scene: Assessment = assess(confirmed, cfg, w, h)
                actions = fsm.update(scene, now)

                # ── evidence frame: CLEAN unless explicitly annotated ──
                if cfg.annotate_evidence:
                    evidence_frame = hud.annotate(
                        frame.copy(),
                        state=fsm.state,
                        scene=scene,
                        tracks=confirmed,
                        regions=cfg.active_regions if show_zones else [],
                        fps=fps,
                        recording=recorder.recording,
                        muted=muted or not cfg.alarm_enabled,
                        warming_up=result.warming_up,
                    )
                else:
                    evidence_frame = hud.stamp(frame.copy())

                # ── execute the FSM's action list ─────────────────
                for action in actions:
                    if action == MOTION_STARTED:
                        log.info("MOTION DETECTED | %s", scene.describe())
                        log_event(
                            "motion",
                            total_area=scene.total_area,
                            zone_count=scene.target_count,
                            fg_ratio=result.foreground_ratio,
                            note=f"zone={scene.zone_name} dir={scene.direction}",
                        )
                    elif action == ALARM_ON:
                        if not muted:
                            alarm.start()
                        log_event("alarm_start")
                    elif action == MAJOR_MOTION:
                        log.warning(
                            "MAJOR alert | Zone %s — %s | %s",
                            scene.zone_name, scene.zone_label, scene.direction,
                        )
                        log_event(
                            "major_motion",
                            total_area=scene.total_area,
                            zone_count=scene.target_count,
                            note=f"zone={scene.zone_name} dir={scene.direction}",
                        )
                    elif action == TAKE_SNAPSHOT:
                        path = recorder.save_snapshot(evidence_frame)
                        log_event("snapshot", note=path.name)
                        log.info("snapshot saved: %s", path.name)
                    elif action == ENTER_RECORDING:
                        path = recorder.start_clip(evidence_frame)
                        if path:
                            log.info("recording %.0fs clip: %s",
                                     cfg.clip_seconds, path.name)
                    elif action == EXIT_RECORDING:
                        path = recorder.stop_clip()
                        log_event("clip", note=path.name if path else "failed")
                        if path:
                            log.info("clip saved: %s", path.name)
                    elif action == ALARM_OFF:
                        alarm.stop()
                        log_event("alarm_stop")
                    elif action == BACK_TO_IDLE:
                        log.info("scene calm — back to IDLE")

                if recorder.recording:
                    recorder.write_clip(evidence_frame)

                # ── smoothed FPS ──────────────────────────────────
                tick = time.monotonic()
                dt = tick - last_tick
                last_tick = tick
                if dt > 0:
                    fps = (0.9 * fps + 0.1 / dt) if fps else 1.0 / dt

                # ── one-line live status ──────────────────────────
                state_name = "WARMUP" if result.warming_up else fsm.state.name
                sys.stdout.write(
                    f"\r  {state_name:<9} targets={scene.target_count:>2}"
                    f"  area={int(scene.total_area):>7}px"
                    f"  zone={scene.zone_name:<2}"
                    f"  dir={scene.direction:<12}"
                    f"  fps={fps:5.1f}  events={events_logged}   "
                )
                sys.stdout.flush()

                # ── live window + operator keys ───────────────────
                if cfg.show_window:
                    view = hud.annotate(
                        frame,
                        state=fsm.state,
                        scene=scene,
                        tracks=confirmed,
                        regions=cfg.active_regions if show_zones else [],
                        fps=fps,
                        recording=recorder.recording,
                        muted=muted or not cfg.alarm_enabled,
                        warming_up=result.warming_up,
                    )
                    cv2.imshow(WINDOW_NAME, view)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("r"):
                        pipeline.reset()
                        tracker.reset()
                        log.info("background model reset by operator")
                    elif key == ord("s"):
                        path = recorder.save_snapshot(evidence_frame)
                        log_event("snapshot", note=f"{path.name} (manual)")
                        log.info("manual snapshot: %s", path.name)
                    elif key == ord("m"):
                        muted = not muted
                        if muted and alarm.active:
                            alarm.stop()
                        log.info("alarm %s", "muted" if muted else "unmuted")
                    elif key == ord("z"):
                        show_zones = not show_zones

                if cfg.duration is not None and tick - started >= cfg.duration:
                    log.info("duration reached — shutting down")
                    break

        except KeyboardInterrupt:
            log.info("interrupted by operator")
        finally:
            # Flush the clip BEFORE releasing anything it depends on.
            recorder.stop_clip()
            alarm.stop()
            source.release()
            if cfg.show_window:
                cv2.destroyAllWindows()
            log_event("system", note="stopped cleanly")
            ledger.close()
            sys.stdout.write("\n")
            print("-" * 60)
            print("  motioniq offline — camera released, evidence saved.")
        return 0
