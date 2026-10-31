"""Ledger tests — SQLite round-trip using pytest's tmp_path sandbox."""
from motioniq.ledger import EventLedger


def test_log_read_and_summary(tmp_path):
    ledger = EventLedger(tmp_path / "test.db")
    first = ledger.log("motion", total_area=1234.0, zone_count=2, fg_ratio=0.05)
    second = ledger.log("snapshot", note="snapshot_001.jpg")
    ledger.log("system", note="armed")

    assert first == 1 and second == 2
    assert ledger.count() == 3

    summary = ledger.summary()
    assert summary == {"motion": 1, "snapshot": 1, "system": 1}

    recent = ledger.recent(limit=10)
    assert recent[0]["type"] == "system"          # newest first
    assert recent[1]["note"] == "snapshot_001.jpg"
    assert recent[2]["total_area"] == 1234.0
    ledger.close()


def test_ledger_survives_reopen(tmp_path):
    path = tmp_path / "keep.db"
    EventLedger(path).log("alarm_start")
    again = EventLedger(path)                     # fresh connection, same file
    assert again.count() == 1
    again.close()
