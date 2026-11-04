"""
Evaluation tests — Feature 3.

These run the whole stack (pipeline + tracker + FSM) against the scripted
synthetic scene and assert measurable quality: every real intrusion is
found, and the curtain never raises an alarm.
"""
from clemvis.config import DetectorConfig
from clemvis.evaluate import run_evaluation


def quick_config(**overrides) -> DetectorConfig:
    base = dict(
        width=640,
        height=360,
        process_width=320,
        demo=True,
        demo_scenario="quick",
        demo_realtime=False,
        show_window=False,
        alarm_enabled=False,
        min_contour_area=1500,
        major_area=40000,
        warmup_frames=10,
        confirm_frames=4,
        min_displacement=40,
    )
    base.update(overrides)
    return DetectorConfig(**base)


def test_quick_scenario_detects_every_real_event():
    report = run_evaluation(quick_config())
    assert len(report.expected) == 2
    assert report.detected == len(report.expected), report.render()
    assert report.missed == 0


def test_quick_scenario_raises_no_false_alarms():
    report = run_evaluation(quick_config())
    assert report.false_alarms == [], report.render()
    assert report.precision == 1.0


def test_response_time_is_fast():
    report = run_evaluation(quick_config())
    # confirmation costs a few frames; it must still react within ~2 s
    assert report.mean_latency < 2.0, report.render()


def test_report_renders_all_key_metrics():
    text = run_evaluation(quick_config()).render()
    for field in (
        "EVALUATION REPORT",
        "Expected events",
        "Detected events",
        "Missed events",
        "False alarms",
        "Recall",
        "Precision",
        "Processing speed",
    ):
        assert field in text


def test_raising_the_travel_gate_suppresses_detections():
    """Sanity check that the anti-false-alarm knob really is the gate."""
    report = run_evaluation(quick_config(min_displacement=100000))
    assert report.detected == 0 and report.false_alarms == []
