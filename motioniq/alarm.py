"""
Cross-platform acoustic alarm — no audio asset, no third-party packages.

The original script used `winsound.PlaySound(path, SND_LOOP)`, which only
exists on Windows and fails silently if the WAV file is missing. Instead we
*synthesize* a two-tone siren:

* Windows → `winsound.Beep(freq, ms)` alternates 988 Hz / 1319 Hz
* macOS/Linux → the standard terminal bell (\\a)

The siren runs in a *daemon thread* so the camera loop never blocks waiting
for a beep to finish. A threading.Event is the stop button; the daemon flag
guarantees the thread can't keep the process alive if we forget to stop it.
"""
from __future__ import annotations

import sys
import threading


def _windows_beeper():
    """Return winsound.Beep on Windows, None everywhere else."""
    try:
        import winsound  # type: ignore[import-not-found]

        return winsound.Beep
    except Exception:
        return None


class Alarm:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if not self.enabled or self.active:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._siren, name="motioniq-siren", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None

    def _siren(self) -> None:
        beep = _windows_beeper()
        while not self._stop.is_set():
            if beep is not None:
                try:
                    beep(988, 220)   # B5
                    if self._stop.is_set():
                        break
                    beep(1319, 220)  # E6 — alternating = classic "yelp"
                    continue
                except Exception:
                    beep = None      # bizarre audio failure → bell fallback
            # Portable fallback: terminal bell, roughly once a second.
            sys.stderr.write("\a")
            sys.stderr.flush()
            self._stop.wait(0.7)
