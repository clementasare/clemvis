"""
Feature 3 — performance and detection evaluation.

"The system detects motion reliably" is an opinion. This module turns it
into a measurement.

The synthetic scene (camera.py) is deterministic and its script is known, so
we have GROUND TRUTH: four real intrusions and three nuisances (curtain,
chair, light switch). We replay it with a virtual clock, record every alert
the state machine raises, and score the run:

    true positive  — an alert inside a real event's window (+ grace period)
    missed event   — a real event that produced no alert
    false alarm    — an alert that matches no real event (e.g. the curtain)
    latency        — seconds from the event starting to the alert firing
    throughput     — frames processed per second of CPU time

Reporting misses and false alarms honestly — and being able to explain
*why* they happen — is the part that matters for a graduate application.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .camera import DemoSource, ScriptedEvent
from .config import DetectorConfig
from .detector import MOTION_STARTED, MotionStateMachine, assess
from .pipeline import VisionPipeline
from .tracking import CentroidTracker

# An alert may legitimately arrive a little after the event starts (the
# confirmation window) and a little after it ends (trailing frames).
GRACE_AFTER_END = 2.5


@dataclass(slots=True)
class Alert:
    at: float               # seconds into the scenario
    zone: str
    severity: str
    direction: str
    area: float


@dataclass(slots=True)
class EvaluationReport:
    scenario: str
    duration: float
    frames: int
    wall_seconds: float
    expected: list[ScriptedEvent]
    distractors: list[ScriptedEvent]
    alerts: list[Alert]
    matched: dict[int, float] = field(default_factory=dict)  # event idx → latency
    false_alarms: list[Alert] = field(default_factory=list)

    # ── derived metrics ───────────────────────────────────────────
    @property
    def detected(self) -> int:
        return len(self.matched)

    @property
    def missed(self) -> int:
        return len(self.expected) - self.detected

    @property
    def fps(self) -> float:
        return self.frames / self.wall_seconds if self.wall_seconds else 0.0

    @property
    def recall(self) -> float:
        return self.detected / len(self.expected) if self.expected else 0.0

    @property
    def precision(self) -> float:
        total = self.detected + len(self.false_alarms)
        return self.detected / total if total else 0.0

    @property
    def mean_latency(self) -> float:
        lat = list(self.matched.values())
        return sum(lat) / len(lat) if lat else 0.0

    def render(self) -> str:
        """Format the report exactly as it appears in the README."""
        lines: list[str] = []
        add = lines.append
        add("=" * 64)
        add("  motioniq — EVALUATION REPORT")
        add("=" * 64)
        add(f"  Scenario            : {self.scenario}")
        add(f"  Test duration       : {self.duration:.0f} s "
            f"({self.frames} frames)")
        add(f"  Expected events     : {len(self.expected)}")
        add(f"  Detected events     : {self.detected}")
        add(f"  Missed events       : {self.missed}")
        add(f"  False alarms        : {len(self.false_alarms)}")
        add(f"  Nuisances in scene  : {len(self.distractors)} "
            f"({', '.join(d.kind for d in self.distractors)})")
        add("-" * 64)
        add(f"  Recall              : {self.recall * 100:5.1f} %")
        add(f"  Precision           : {self.precision * 100:5.1f} %")
        add(f"  Mean response time  : {self.mean_latency:5.2f} s")
        add(f"  Processing speed    : {self.fps:5.1f} FPS")
        add("-" * 64)
        add("  EVENT BREAKDOWN")
        for i, event in enumerate(self.expected):
            if i in self.matched:
                verdict = f"DETECTED  (+{self.matched[i]:.2f}s)"
            else:
                verdict = "MISSED"
            add(f"   [{event.start:5.1f}s] {event.label:<34} {verdict}")
        for d in self.distractors:
            hit = any(d.start <= a.at <= d.end + GRACE_AFTER_END
                      for a in self.false_alarms)
            verdict = "FALSE ALARM" if hit else "correctly ignored"
            add(f"   [{d.start:5.1f}s] {d.label:<34} {verdict}")
        add("=" * 64)
        return "\n".join(lines)


def run_evaluation(config: DetectorConfig) -> EvaluationReport:
    """Replay the scripted scene offline and score the detector."""
    cfg = config
    source = DemoSource(cfg)
    pipeline = VisionPipeline(cfg)
    tracker = CentroidTracker(
        max_distance=cfg.track_max_distance,
        ttl_frames=cfg.track_ttl_frames,
        confirm_frames=cfg.confirm_frames,
        min_displacement=cfg.min_displacement,
    )
    fsm = MotionStateMachine(cfg)

    alerts: list[Alert] = []
    frames = 0
    started = time.monotonic()

    while True:
        ok, frame = source.read()
        if not ok or frame is None:
            break
        frames += 1
        now = source.timestamp()
        elapsed = now - source._epoch  # seconds into the scenario

        result = pipeline.process(frame)
        tracks = tracker.update(result.blobs, now)
        confirmed = tracker.confirmed
        h, w = frame.shape[:2]
        scene = assess(confirmed, cfg, w, h)

        for action in fsm.update(scene, now):
            if action == MOTION_STARTED:
                alerts.append(
                    Alert(
                        at=elapsed,
                        zone=scene.zone_name,
                        severity=scene.severity,
                        direction=scene.direction,
                        area=scene.total_area,
                    )
                )

    wall = time.monotonic() - started
    report = EvaluationReport(
        scenario=cfg.demo_scenario,
        duration=source.duration,
        frames=frames,
        wall_seconds=wall,
        expected=source.expected_events,
        distractors=source.distractors,
        alerts=alerts,
    )

    # ── score: greedily match each alert to an unclaimed real event ──
    for alert in alerts:
        matched_idx: int | None = None
        for i, event in enumerate(report.expected):
            if i in report.matched:
                continue
            if event.start <= alert.at <= event.end + GRACE_AFTER_END:
                matched_idx = i
                break
        if matched_idx is None:
            report.false_alarms.append(alert)
        else:
            report.matched[matched_idx] = alert.at - report.expected[matched_idx].start

    return report
