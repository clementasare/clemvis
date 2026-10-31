"""
The event ledger — every detection, alarm toggle and capture, persisted.

The original script used `print()` with emojis: gone the moment you close
the terminal. We use SQLite — a *serverless* database shipped inside the
Python standard library (sqlite3). One file, zero setup, real SQL, and it
survives restarts. For a portfolio project it's the perfect teaching point:
you get durability without installing PostgreSQL.

Table: events(id, ts_utc, type, total_area, zone_count, fg_ratio, note)
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc      TEXT NOT NULL,
    type        TEXT NOT NULL,
    total_area  REAL,
    zone_count  INTEGER,
    fg_ratio    REAL,
    note        TEXT NOT NULL DEFAULT ''
)
"""


class EventLedger:
    """Thin, honest wrapper: one method per operation, plain f-string SQL."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self.total_logged = 0

    def log(
        self,
        event_type: str,
        *,
        total_area: float | None = None,
        zone_count: int | None = None,
        fg_ratio: float | None = None,
        note: str = "",
    ) -> int:
        """Append one event; returns its row id. Timestamps are UTC ISO-8601."""
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        cur = self._conn.execute(
            "INSERT INTO events (ts_utc, type, total_area, zone_count,"
            " fg_ratio, note) VALUES (?, ?, ?, ?, ?, ?)",
            (ts, event_type, total_area, zone_count, fg_ratio, note),
        )
        self._conn.commit()
        self.total_logged += 1
        return int(cur.lastrowid or 0)

    def recent(self, limit: int = 15) -> list[sqlite3.Row]:
        cur = self._conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        )
        return list(cur.fetchall())

    def summary(self) -> dict[str, int]:
        cur = self._conn.execute(
            "SELECT type, COUNT(*) AS n FROM events GROUP BY type ORDER BY n DESC"
        )
        return {row["type"]: int(row["n"]) for row in cur.fetchall()}

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) AS n FROM events")
        return int(cur.fetchone()["n"])

    def close(self) -> None:
        self._conn.close()
