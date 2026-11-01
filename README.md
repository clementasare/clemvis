<div align="center">

# motioniq

A real-time computer vision and motion-tracking system featuring background subtraction, centroid tracking, zone-based monitoring, finite-state machine control, and performance evaluation.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-4dabf7.svg)](https://numpy.org/)
[![SQLite](https://img.shields.io/badge/storage-SQLite-lightgrey.svg)](https://www.sqlite.org/)
[![Tests](https://img.shields.io/badge/tests-39%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

[Features](#features) · [How it works](#how-it-works) · [Performance](#performance-evaluation) · [Installation](#installation) · [Usage](#usage) · [Architecture](#architecture)

<!-- To add a demo image, drop a screenshot or GIF into docs/ and uncomment:
<p align="center"><img src="docs/demo.gif" alt="motioniq detecting motion" width="720"></p>
-->

</div>

---

## Overview

**motioniq** is a desktop surveillance system written entirely in Python. It reads a live video feed, determines when a *person or object is genuinely moving* through the scene, raises an alarm, records video evidence, and stores a permanent, queryable log of everything that happened — all inside a terminal and a single OpenCV window.

It is designed to solve the two problems that make simple motion detectors useless in practice:

1. **False alarms.** A naive detector fires at anything that changes — curtains in a draft, a wobbling chair, a cloud passing the window, a light being switched on. motioniq rejects all of these by requiring motion to be large enough, persistent enough, and to actually *travel* somewhere.
2. **Unusable output.** Detectors that save every frame produce hundreds of files nobody will ever review. motioniq captures a strictly rate-limited set of clean, timestamped artifacts that remain useful as evidence.

The system is fully self-contained: no cloud services, no API keys, no database server, no web framework. Its only runtime dependencies are **OpenCV** and **NumPy**; persistence uses **SQLite**, which ships with the Python standard library.

```
10:02:15 INFO  MOTION DETECTED | area=24984px  Zone A — Entrance  severity=NORMAL  direction=LEFT → RIGHT
10:02:15 INFO  snapshot saved: snapshot_20261030_100215_844.jpg
10:02:15 INFO  recording 8s clip: movement_20261030_100215_851.avi
10:02:18 WARN  MAJOR alert | Zone C — Restricted area | LEFT → RIGHT
10:02:23 INFO  clip saved: movement_20261030_100215_851.avi

  RECORDING targets= 1  area=  20693px  zone=C   dir=LEFT → RIGHT  fps= 20.2  events=5
```

---

## Features

| | Feature | Description |
|---|---|---|
| 🎯 | **Reliable motion detection** | Adaptive background subtraction rejects noise, gradual lighting drift and camera shake |
| 🧠 | **False-alarm filtering** | Size, persistence and net-travel gates reduce false detections caused by transient motion and stationary oscillating objects. |
| ➡️ | **Motion direction tracking** | Centroid tracking reports whether a target moves left, right, up or down, and draws a live vector |
| 🗺️ | **Detection zones** | The view is divided into regions of interest; motion in a restricted zone escalates the alert |
| 🚨 | **Audible alarm** | A synthesized two-tone siren with hysteresis, so it cannot flicker on and off |
| 📸 | **Clean evidence capture** | Rate-limited snapshots and video clips carrying only a UTC timestamp — no overlay clutter |
| 📊 | **Performance evaluation** | Results are from a deterministic synthetic benchmark and should not be interpreted as real-world detection accuracy. Real-world performance depends on lighting, camera placement, scene conditions and threshold configuration. |
| 🗄️ | **Event ledger** | Every detection is written to an SQLite database with timestamps, area, zone and direction |
| 🖥️ | **Runs anywhere** | Windows, macOS and Linux, on any machine with a webcam — including a Raspberry Pi |
| ✅ | **Tested** | 39 unit tests covering the vision pipeline, tracker, zones, state machine, ledger and evaluator |

---

## How it works

### The vision pipeline

Each frame passes through a classical computer-vision chain — the same sequence of operations used in industrial change-detection systems:

```
 frame ─▶ downscale ─▶ grayscale ─▶ |frame − background| ─▶ Gaussian blur
       ─▶ threshold ─▶ dilate ─▶ contour extraction ─▶ tracked targets
```

| Stage | Purpose |
|---|---|
| **Downscale** | Detection runs on a ~480 px-wide copy, roughly 4× faster than full resolution while remaining accurate |
| **Grayscale** | Colour carries no motion information; one channel is three times less data |
| **Background subtraction** | The frame is compared against a *learned model of the empty scene* rather than the previous frame, so moving objects leave no ghost trail |
| **Gaussian blur** | Removes single-pixel sensor noise before it can cross the detection threshold |
| **Threshold** | Converts a continuous difference image into a binary foreground mask |
| **Dilate** | Merges the fragments of one object into a single connected blob |
| **Contours** | Extracts bounding boxes, centroids and areas for each moving region |

The background model is adaptive: it updates slowly while the scene is calm (absorbing daylight changes), freezes while motion is in progress (so a person who stops moving is not absorbed into the background), and re-learns rapidly after a sudden illumination change.

### Rejecting false alarms

Motion must pass **three independent gates** before it is reported:

| Gate | Rule | What it eliminates |
|---|---|---|
| **Size** | Contour area ≥ 4,000 px² | Sensor noise, insects, distant foliage |
| **Persistence** | Present for ≥ 5 consecutive frames | Momentary flicker, compression artefacts |
| **Travel** | Displaced ≥ 60 px from its point of origin | Curtains, fans, hanging cables, wobbling furniture |

The travel gate is the decisive one. A curtain flutters constantly but never leaves the point it hangs from — its *net* displacement stays near zero no matter how long it moves. A person walking through a scene genuinely abandons their starting position. Two further guards handle edge cases: a warm-up period ignores the first frames while the model settles, and a saturation detector discards any frame in which more than 35 % of pixels change at once (a light switch or auto-exposure jump), then re-learns the background quickly.

### Motion direction tracking

Every detected blob is linked to the nearest tracked centroid from the previous frame. Over its lifetime, a track accumulates a position history from which the system derives displacement, velocity and a human-readable direction — `LEFT → RIGHT`, `RIGHT → LEFT`, `UP → DOWN`, `DOWN → UP` or `STATIONARY`. This is displayed as an arrow in the live view and recorded with every event.

### Detection zones

The camera view is divided into regions of interest, each with an assigned severity:

```
┌───────────────┬───────────────┬────────────────┐
│  A — Entrance │  B — Main area│ C — Restricted │
│    normal     │    normal     │     MAJOR      │
└───────────────┴───────────────┴────────────────┘
```

Zones are defined in normalised coordinates, so a layout is independent of camera resolution. Motion in a `major` zone escalates the alert regardless of object size, while motion in an `ignore` zone is discarded before tracking even begins — an effective way to permanently mask a busy road or a window. Zone layouts can be redefined from the command line without touching code.

### The decision state machine

All alarm and recording behaviour is governed by an explicit finite-state machine:

```
                 confirmed target                capture permitted
    IDLE ──────────────────────▶ ALERT ──────────────────────▶ RECORDING
     ▲                              │  ▲                            │
     │      3 s of sustained calm   │  │ target still active        │ clip complete
     └──────────────────────────────┘  └────────────────────────────┘
```

The alarm is released only after three continuous seconds of calm — a hysteresis mechanism that prevents it from strobing when a detection hovers near the threshold. The machine performs no input or output of its own; it emits named actions that the application layer executes. This separation is what allows the entire decision logic to be unit-tested without a camera attached.

### Evidence capture

Snapshots and clips are deliberately rate-limited — one snapshot per intrusion with a 10-second cooldown, and at most one clip every 30 seconds — so a sustained intrusion produces a small, reviewable set of files rather than an unreadable flood. Saved footage carries only a UTC timestamp, keeping the image clean and useful as evidence; a diagnostic overlay can be enabled when tuning is required.

---

## Performance evaluation

The repository includes a synthetic, fully deterministic test scene whose script is known in advance — making it usable as **ground truth**. The scene contains four genuine intrusions *and* three events that commonly defeat motion detectors: a fluttering curtain, a wobbling office chair, and a ceiling light being switched on.

`python main.py evaluate` replays the scene and scores the detector:

```
================================================================
  motioniq — EVALUATION REPORT
================================================================
  Scenario            : standard
  Test duration       : 60 s (1201 frames)
  Expected events     : 4
  Detected events     : 4
  Missed events       : 0
  False alarms        : 0
  Nuisances in scene  : 3 (curtain, light, chair)
----------------------------------------------------------------
  Recall              : 100.0 %
  Precision           : 100.0 %
  Mean response time  :  0.66 s
  Processing speed    :  55.4 FPS
----------------------------------------------------------------
  EVENT BREAKDOWN
   [  3.0s] person crosses left to right       DETECTED  (+0.40s)
   [ 22.0s] person crosses right to left       DETECTED  (+0.35s)
   [ 36.0s] person enters restricted zone      DETECTED  (+1.05s)
   [ 52.0s] person walks top to bottom         DETECTED  (+0.85s)
   [ 11.0s] curtain flutters by the window     correctly ignored
   [ 30.0s] ceiling light switched on          correctly ignored
   [ 44.0s] office chair wobbles               correctly ignored
================================================================
```

The processing speed of roughly 55 FPS comfortably exceeds a 30 FPS camera, so the system analyses every frame with headroom to spare. Response time — the delay between an intrusion beginning and the alarm firing — averages 0.66 seconds, most of which is the deliberate persistence gate that filters transient noise.

---

## Installation

**Requirements:** Python 3.10 or newer, and a webcam (or use the built-in synthetic scene).

```bash
git clone https://github.com/clementasare/motioniq.git
cd motioniq

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

For development, including the test suite:

```bash
pip install -r requirements-dev.txt
```

The package can also be installed, which provides a `motioniq` console command:

```bash
pip install .
```

---

## Usage

### Running the camera

```bash
python main.py
```

Opens the camera, displays the live annotated view, and begins monitoring. Press `q` or `Esc` to stop; `Ctrl+C` also shuts down cleanly and flushes any recording in progress.

### Keyboard controls

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `s` | Take a snapshot immediately |
| `r` | Reset the learned background model |
| `m` | Mute / unmute the alarm |
| `z` | Toggle the zone overlay on the live view |

### All commands

| Command | Purpose |
|---|---|
| `python main.py` | Monitor the default webcam with a live window |
| `python main.py --camera 1` | Monitor a secondary camera |
| `python main.py --demo` | Run a scripted synthetic scene — no hardware required |
| `python main.py --demo --headless --duration 24` | Run the synthetic scene without a window for a fixed duration |
| `python main.py evaluate` | Benchmark detection performance and print a full report |
| `python main.py report` | Print the recorded event ledger |
| `python main.py --preset outdoor` | Apply a tested tuning profile |
| `python main.py --no-alarm` | Monitor silently |
| `python main.py --annotate-evidence` | Include the diagnostic overlay in saved footage |
| `python main.py --help` | List every option with its default |

### Tuning profiles

Three profiles are bundled, each tested for a different environment:

| Preset | Threshold | Min area | Persistence | Min travel | Intended for |
|---|---|---|---|---|---|
| `indoor` *(default)* | 32 | 4,000 px² | 5 frames | 60 px | Homes, offices, hallways |
| `outdoor` | 42 | 7,000 px² | 7 frames | 90 px | Driveways, gardens, windy sites |
| `sensitive` | 22 | 1,800 px² | 3 frames | 30 px | Small or slow subjects, testing |

Any individual parameter can be overridden without a preset:

```bash
python main.py --min-travel 120 --confirm-frames 8
```

### Custom detection zones

Zones are defined as `name:label:severity:x0,y0,x1,y1` in normalised coordinates from 0 to 1, separated by semicolons. The label is optional.

```bash
python main.py --zones "A:Entrance:normal:0,0,0.33,1; B:Main area:normal:0.33,0,0.66,1; C:Restricted:major:0.66,0,1,1"
```

To mask a distracting region — a window or a busy street — assign it `ignore` severity and it will be excluded from analysis entirely:

```bash
python main.py --zones "A:Window:ignore:0,0,0.3,1; B:Room:normal:0.3,0,1,1"
```

Zones can be disabled with `--no-zones`.

### Viewing the event log

Every detection, alarm transition, snapshot and clip is appended to an SQLite database at `detections/motioniq.db`.

```bash
python main.py report
```

```
============================================================================================
  motioniq LEDGER — detections/motioniq.db
============================================================================================
  18 events: motion=4, alarm_start=4, alarm_stop=4, snapshot=1, major_motion=3, ...
--------------------------------------------------------------------------------------------
     ID  TIME (UTC)                 TYPE              AREA  TGT  NOTE
--------------------------------------------------------------------------------------------
      2  2026-10-30T10:02:15.839+00 motion           24984    1  zone=A dir=LEFT → RIGHT
      5  2026-10-30T10:02:18.940+00 major_motion     20337    1  zone=C dir=LEFT → RIGHT
      ...
============================================================================================
```

Because the ledger is a standard SQLite file, it can also be queried directly with any SQL client or scripted analysis.

---

## Configuration reference

| Option | Default | Meaning |
|---|---|---|
| `--threshold` | 32 | Brightness change (0–255) required for a pixel to count as changed |
| `--min-area` | 4000 | Minimum contour area in px² |
| `--major-area` | 45000 | Total area that escalates an alert to MAJOR |
| `--confirm-frames` | 5 | Frames a target must persist before alerting |
| `--min-travel` | 60 | Pixels a target must travel from its origin |
| `--process-width` | 480 | Width of the internal analysis frame |
| `--clip-cooldown` | 30 | Minimum seconds between two recorded clips |
| `--snapshot-cooldown` | 10 | Minimum seconds between two snapshots |
| `--save-dir` | `detections` | Output directory for evidence and the database |
| `--db` | *(save-dir)/motioniq.db* | Path of the event database |
| `--no-alarm` | off | Disable the audible siren |
| `--annotate-evidence` | off | Burn the diagnostic overlay into saved files |
| `--no-zones` | off | Disable region-of-interest logic |
| `--camera` | 0 | Camera device index |
| `--demo` | off | Use the synthetic scene instead of hardware |
| `--scenario` | standard | Which synthetic script to replay (`standard` / `quick`) |
| `--headless` | off | Run without an OpenCV window |
| `--duration` | ∞ | Stop automatically after N seconds |
| `-v` | off | Verbose debug logging |

---

## Architecture

```
┌────────────┐  frames  ┌──────────┐  blobs  ┌───────────┐ tracks ┌──────────┐
│   camera   │ ───────▶ │ pipeline │ ──────▶ │  tracker  │ ─────▶ │  assess  │
└────────────┘          └──────────┘         └───────────┘        └────┬─────┘
                 background subtraction   confirm + travel gates     │
                                                                    ▼
                                                              ┌──────────┐
                                                              │   FSM    │
                                                              └────┬─────┘
                                          actions: alarm · snapshot · clip · log
                                                                   ▼
                                             alarm.py · evidence.py · ledger.py
```

Each stage is a separate module with a single responsibility, and the decision layers perform no input or output — which is why they can be tested in isolation without a camera.

### Project structure

```
motioniq/
├── pipeline.py     Vision: background subtraction and contour extraction
├── tracking.py     Centroid tracker, direction, confirmation and travel gates
├── zones.py        Regions of interest and severity assignment
├── detector.py     Scene assessment and the finite-state machine
├── evaluate.py     Benchmark harness measuring recall and precision
├── camera.py       Webcam source and the deterministic synthetic scene
├── app.py          Main loop: wiring and action execution
├── evidence.py     Snapshot and clip recording
├── alarm.py        Cross-platform audible siren
├── ledger.py       SQLite event persistence
├── hud.py          Live display overlay and evidence timestamping
├── config.py       All tunables, defaults and presets
└── cli.py          Command-line interface

tests/
├── test_pipeline.py     Vision pipeline behaviour
├── test_tracking.py     Tracker and false-alarm gates
├── test_zones.py        Region-of-interest logic
├── test_fsm.py          State machine transitions
├── test_ledger.py       Database round-trips
└── test_evaluation.py   End-to-end benchmark assertions

main.py              Entry point
pyproject.toml       Packaging metadata
requirements.txt     Runtime dependencies
```

---

## Testing

The test suite runs entirely without a camera. Vision tests feed crafted NumPy frames to the pipeline; tracking tests replay synthetic motion paths, including a 200-frame curtain oscillation that must never be confirmed; state machine tests inject explicit timestamps to replay complete intrusion timelines deterministically.

```bash
pytest -q
```

```
.......................................                          [100%]
39 passed in 10.46s
```

The synthetic camera scene serves a second purpose here: because it is deterministic and its script is known, it doubles as ground truth for the evaluation benchmark, allowing end-to-end quality to be asserted rather than assumed.

---

## Design decisions

**Adaptive background over frame differencing.** Subtracting consecutive frames detects a moving object twice — on arrival and on departure — and leaves a ghost trail. motioniq maintains a running average of the empty scene and subtracts that, updating it only while the scene is calm so that stationary people are not absorbed.

**Evidence saved without clutter.** Recorded footage carries only a UTC timestamp rather than bounding boxes and telemetry. Evidence should show the scene; the diagnostic overlay exists for tuning and can be enabled explicitly.

**Vision at reduced resolution.** Analysing a downscaled frame costs a fraction of the computation while detection accuracy is unaffected, since coordinates are rescaled back to full-frame pixel units.

**A state machine over scattered flags.** Encoding behaviour as explicit states and transitions makes every rule inspectable and testable, rather than emerging unpredictably from interacting global variables.

**SQLite over a database server.** An embedded database gives durable, transactional, queryable storage with zero deployment overhead — appropriate for an edge device, and swappable for PostgreSQL by changing a single module.

---

## Limitations

Stated plainly, because they define the project's scope:

- **Visible-light, static camera.** The system requires a fixed viewpoint; a moving or panning camera invalidates the background model.
- **Detection, not recognition.** motioniq reports that something moved, where and in which direction — it does not classify what the object is. A pet and a person produce the same alert.
- **Nearest-neighbour tracking.** The tracker links detections by proximity. Two targets that fully overlap for several frames may have their identities merged.
- **Benchmark scope.** The 100 % recall and precision figures are measured against the synthetic scene, which validates the detection logic. Real environments are less controlled and typically require per-site tuning of the travel and area thresholds.
- **No remote access.** Evidence is written to local disk; there is no network streaming, notification or web interface.

---

## Roadmap

- [ ] MOG2 / KNN background models for outdoor robustness
- [ ] Lucas–Kanade optical flow for sub-pixel velocity estimation
- [ ] Person classification to distinguish people from animals
- [ ] MQTT / ROS 2 publisher for integration into a robot stack
- [ ] Raspberry Pi deployment profile with Picamera2
- [ ] YAML configuration and a graphical zone editor

---

## Contributing

Issues and pull requests are welcome. The codebase is intentionally small and modular — each module fits on one screen — with the aim that a new reader can understand any single component in isolation.

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## License

Released under the [MIT License](LICENSE).

Motion-detection approach derived from the standard background-subtraction idiom documented in the [OpenCV](https://docs.opencv.org/) computer-vision library.
