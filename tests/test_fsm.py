"""
State machine tests — injected timestamps let us replay whole intrusion
timelines deterministically and assert every rule: hysteresis, capture
cooldowns, escalation and budget resets.
"""
from motioniq.config import DetectorConfig
from motioniq.detector import (
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
)

CALM = Assessment()


def active(severity="normal", area=20000.0, targets=1) -> Assessment:
    return Assessment(
        active=True,
        target_count=targets,
        total_area=area,
        direction="LEFT → RIGHT",
        severity=severity,
    )


def make_fsm(**overrides) -> MotionStateMachine:
    base = dict(
        alarm_calm_seconds=3.0,
        calm_reset_seconds=15.0,
        clip_seconds=8.0,
        clip_cooldown_seconds=30.0,
        snapshots_per_burst=1,
        snapshot_cooldown_seconds=10.0,
    )
    base.update(overrides)
    return MotionStateMachine(DetectorConfig(**base))


def test_confirmed_motion_raises_alert_and_alarm():
    fsm = make_fsm()
    actions = fsm.update(active(), now=0.0)
    assert fsm.state is State.RECORDING          # first clip allowed instantly
    assert MOTION_STARTED in actions and ALARM_ON in actions
    assert TAKE_SNAPSHOT in actions and ENTER_RECORDING in actions


def test_only_one_snapshot_per_intrusion():
    fsm = make_fsm()
    snaps = 0
    for t in (0.0, 0.5, 1.0, 2.0, 3.0):
        snaps += fsm.update(active(), now=t).count(TAKE_SNAPSHOT)
    assert snaps == 1


def test_second_clip_is_blocked_by_the_cooldown():
    fsm = make_fsm(clip_seconds=2.0, clip_cooldown_seconds=30.0)
    fsm.update(active(), now=0.0)                 # clip 1 starts
    assert EXIT_RECORDING in fsm.update(active(), now=2.1)
    # motion continues, but no second clip until the cooldown expires
    for t in (3.0, 10.0, 20.0, 31.0):
        assert ENTER_RECORDING not in fsm.update(active(), now=t)
    assert ENTER_RECORDING in fsm.update(active(), now=32.2)


def test_alarm_needs_sustained_calm_hysteresis():
    fsm = make_fsm(clip_seconds=1.0, clip_cooldown_seconds=9999)
    fsm.update(active(), now=0.0)                 # alarm engages
    # the clip closing at 1.1 s must NOT release the alarm early
    assert ALARM_OFF not in fsm.update(CALM, now=1.1)
    assert ALARM_OFF not in fsm.update(CALM, now=2.9)
    actions = fsm.update(CALM, now=3.1)           # 3 s of calm → release
    assert ALARM_OFF in actions and BACK_TO_IDLE in actions
    assert fsm.state is State.IDLE and not fsm.alarm_on


def test_major_severity_announced_once_per_intrusion():
    fsm = make_fsm()
    assert MAJOR_MOTION in fsm.update(active(severity="major"), now=0.0)
    for t in (0.5, 1.0, 2.0):
        assert MAJOR_MOTION not in fsm.update(active(severity="major"), now=t)


def test_normal_zone_motion_is_not_escalated():
    fsm = make_fsm()
    assert MAJOR_MOTION not in fsm.update(active(severity="normal"), now=0.0)


def test_snapshot_budget_resets_after_long_calm():
    fsm = make_fsm(clip_seconds=1.0, clip_cooldown_seconds=0.0)
    assert TAKE_SNAPSHOT in fsm.update(active(), now=0.0)
    fsm.update(CALM, now=1.5)
    fsm.update(CALM, now=20.0)                    # calm > 15 s → budget reset
    assert TAKE_SNAPSHOT in fsm.update(active(), now=40.0)


def test_calm_scene_never_triggers_anything():
    fsm = make_fsm()
    for t in range(20):
        assert fsm.update(CALM, now=float(t)) == []
    assert fsm.state is State.IDLE
