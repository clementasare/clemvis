<div align="center">

## CLEMVIS

A real-time computer vision and motion-tracking system featuring background subtraction, centroid tracking, zone-based monitoring, finite-state machine control, and performance evaluation.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-4dabf7.svg)](https://numpy.org/)
[![SQLite](https://img.shields.io/badge/storage-SQLite-lightgrey.svg)](https://www.sqlite.org/)
[![Tests](https://img.shields.io/badge/tests-39%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

[Features](#features) · [How it works](#how-it-works) · [Performance](#performance-evaluation) · [Installation](#installation) · [Usage](#usage) · [Architecture](#architecture)
</div>

---

## Overview

CLEMVIS is a motion detection security camera written in Python with OpenCV. It watches a webcam feed, follows moving objects across the frame, reports the direction they travel, saves snapshots and short clips, and keeps a searchable event history in SQLite.

The project started as a small frame differencing script. That version worked, but it reacted to everything: curtains in a draft, a wobbling chair, lighting changes. It also wrote so many files during any movement that reviewing the output later was pointless. The current version works on both problems:

1. **False alarms.** Movement is only reported when it is large enough, persists long enough, and travels across the scene instead of oscillating in place.
2. **Flooded output.** Snapshots and clips are rate limited, and saved files contain clean footage with a timestamp rather than the debugging overlay.

Everything runs locally. There is no cloud service, no API keys, and no database server. The only runtime dependencies are OpenCV and NumPy. The event storage uses SQLite, which is part of the Python standard library.

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
| 🧠 | **False-alarm filtering** | Size, persistence and travel gates reject sensor flashes and oscillating household objects |
| ➡️ | **Motion direction tracking** | Centroid tracking reports left, right, up or down movement and draws a live vector on the target |
| 🗺️ | **Detection zones** | Regions of interest with separate severities; restricted zones escalate the alert |
| 🚨 | **Audible alarm** | A synthesized two-tone siren with hysteresis, running in a background thread |
| 📸 | **Clean evidence capture** | Rate-limited snapshots and clips carrying only a UTC timestamp, no overlay clutter |
| 📊 | **Performance evaluation** | A deterministic synthetic benchmark measures recall, precision, response time and frame rate |
| 🗄️ | **Event ledger** | Every detection is written to SQLite with timestamp, area, zone and direction |
| 🖥️ | **Runs anywhere** | Windows, macOS and Linux, on any machine with a webcam, including a Raspberry Pi |
| ✅ | **Tested** | 39 unit tests covering the pipeline, tracker, zones, state machine, ledger and evaluator |

---

## How it works

### The vision pipeline

Each frame goes through the same processing chain:

```
 frame ─▶ downscale ─▶ grayscale ─▶ |frame − background| ─▶ Gaussian blur
       ─▶ threshold ─▶ dilate ─▶ contour extraction ─▶ tracked targets
```

| Stage | Purpose |
|---|---|
| **Downscale** | Detection runs on a roughly 480 px wide copy, about four times less work per frame while motion stays obvious |
| **Grayscale** | Colour carries no motion information; one channel instead of three |
| **Background subtraction** | The frame is compared against a learned model of the empty scene, so moving objects leave no ghost trail |
| **Gaussian blur** | Removes single-pixel sensor noise before it can cross the detection threshold |
| **Threshold** | Converts the difference image into a binary foreground mask |
| **Dilate** | Merges the fragments of one object into a single connected blob |
| **Contours** | Extracts bounding boxes, centroids and areas for each moving region |

The background model is adaptive. While the scene is calm it drifts slowly, which absorbs daylight changes. While motion is present it freezes, so a person who stands still does not get absorbed into the background. After a sudden illumination change it relearns quickly.

### Rejecting false alarms

Movement must pass three independent gates before it is reported:

| Gate | Rule | What it eliminates |
|---|---|---|
| **Size** | Contour area ≥ 4,000 px² | Sensor noise, insects, distant foliage |
| **Persistence** | Present for ≥ 5 consecutive frames | Single-frame flashes and compression artefacts |
| **Travel** | Displaced ≥ 60 px from its point of origin | Curtains, fans, hanging cables, wobbling furniture |

The travel gate does most of the work here. A curtain can swing for minutes, but since it never leaves its anchor point, its net displacement stays close to zero. Someone walking across a room abandons their starting position, so they pass the gate. Two more guards cover the remaining edge cases. A warm-up period ignores the first frames while the background settles, and a saturation check discards any frame where more than 35 percent of pixels change at once, which is what a light switch or an automatic exposure jump looks like. After such a frame the background adapts fast instead of firing alarms.

### Motion direction tracking

Every detected blob is matched to the nearest tracked centroid from the previous frame. Across its lifetime a track builds up a position history, and from that history the program derives displacement, velocity, and a direction label: `LEFT → RIGHT`, `RIGHT → LEFT`, `UP → DOWN`, `DOWN → UP`, or `STATIONARY`. The direction appears as an arrow in the live view and is written into every ledger entry.

### Detection zones

The camera view is split into regions of interest, each with its own severity:

```
┌───────────────┬───────────────┬────────────────┐
│  A — Entrance │  B — Main area│ C — Restricted │
│    normal     │    normal     │     MAJOR      │
└───────────────┴───────────────┴────────────────┘
```

Zone rectangles use normalised coordinates between 0 and 1, so the same layout works at any camera resolution. Motion in a `major` zone escalates the alert no matter how small the object is. Motion in an `ignore` zone is dropped before tracking starts, which is the simplest way to mask a window, a busy road, or a stubborn curtain. The layout can be changed from the command line without editing code.

### The decision state machine

All alarm and recording behaviour is governed by a small finite-state machine:

```
                 confirmed target                capture permitted
    IDLE ──────────────────────▶ ALERT ──────────────────────▶ RECORDING
     ▲                              │  ▲                            │
     │      3 s of sustained calm   │  │ target still active        │ clip complete
     └──────────────────────────────┘  └────────────────────────────┘
```

The alarm only switches off after three continuous seconds of calm. That hysteresis keeps the siren from flickering when a detection sits near the threshold. The state machine performs no input or output itself. It returns named actions, and the main application executes them. That separation is what makes the decision logic testable without a camera.

### Evidence capture

Snapshots and clips are rate limited: one snapshot per intrusion with a 10 second cooldown and no more than one clip every 30 seconds. Sustained movement therefore produces a small, reviewable set of files rather than a flood. Saved footage carries only a UTC timestamp, so the recording shows the scene itself. When tuning, the full diagnostic overlay can still be written into saved files with `--annotate-evidence`.

---

## Performance evaluation

The repository ships a synthetic, deterministic test scene. Since its script is known in advance, it acts as ground truth. The scene contains four genuine intrusions and three events that commonly fool motion detectors: a fluttering curtain, a wobbling office chair and a ceiling light being switched on.

`python main.py evaluate` replays the scene and scores the detector:

```
================================================================
  CLEMVIS — EVALUATION REPORT
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

These numbers need context. They come from a controlled, synthetic benchmark, so they confirm that the implemented logic behaves as designed; they do not promise that any real room will score 100 percent. Real performance depends on lighting, camera placement, scene clutter, and tuning, which is exactly what the presets and flags are for.

Processing at about 55 FPS gives comfortable headroom above a 30 FPS camera feed. The 0.66 s mean response time is mostly the persistence gate waiting a few frames before it trusts a detection.

---

## Installation

**Requirements:** Python 3.10 or newer, and a webcam. The built-in synthetic scene works without one.

```bash
git clone https://github.com/clementasare/clemvis.git
cd clemvis

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

For development, including the test suite:

```bash
pip install -r requirements-dev.txt
```

The package can also be installed, which provides a `clemvis` console command:

```bash
pip install .
```

---

## Usage

### Running the camera

```bash
python main.py
```

Opens the camera, shows the live annotated view, and starts monitoring. The first few seconds are a warm-up while the background model settles, and detections are intentionally suppressed during that time. Press `q` or `Esc` to stop. `Ctrl+C` also shuts down cleanly and flushes any recording in progress.

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
| `python main.py --demo` | Run a scripted synthetic scene, no hardware required |
| `python main.py --demo --headless --duration 24` | Run the synthetic scene without a window for a fixed duration |
| `python main.py evaluate` | Benchmark detection performance and print a full report |
| `python main.py report` | Print the recorded event ledger |
| `python main.py --preset outdoor` | Apply a tested tuning profile |
| `python main.py --no-alarm` | Monitor silently |
| `python main.py --annotate-evidence` | Include the diagnostic overlay in saved footage |
| `python main.py --help` | List every option with its default |

### Tuning profiles

Three profiles are bundled, each tuned for a different environment:

| Preset | Threshold | Min area | Persistence | Min travel | Intended for |
|---|---|---|---|---|---|
| `indoor` *(default)* | 32 | 4,000 px² | 5 frames | 60 px | Homes, offices, hallways |
| `outdoor` | 42 | 7,000 px² | 7 frames | 90 px | Driveways, gardens, windy sites |
| `sensitive` | 22 | 1,800 px² | 3 frames | 30 px | Small or slow subjects, testing |

Any individual parameter can be overridden without using a preset:

```bash
python main.py --min-travel 120 --confirm-frames 8
```

Adjust one value at a time and re-run `evaluate` between changes, so it stays clear which setting produced which effect.

### Custom detection zones

Zones are defined as `name:label:severity:x0,y0,x1,y1` in normalised coordinates from 0 to 1, separated by semicolons. The label is optional.

```bash
python main.py --zones "A:Entrance:normal:0,0,0.33,1; B:Main area:normal:0.33,0,0.66,1; C:Restricted:major:0.66,0,1,1"
```

A distracting region like a window or a busy street can be masked completely by giving it `ignore` severity:

```bash
python main.py --zones "A:Window:ignore:0,0,0.3,1; B:Room:normal:0.3,0,1,1"
```

Zones can be disabled entirely with `--no-zones`.

### Viewing the event log

Every detection, alarm transition, snapshot and clip is appended to an SQLite database at `detections/clemvis.db`.

```bash
python main.py report
```

```
============================================================================================
  CLEMVIS LEDGER — detections/clemvis.db
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

Since the ledger is a standard SQLite file, it can also be queried directly with any SQL client or a short script.

---

## Configuration reference

| Option | Default | Meaning |
|---|---|---|
| `--threshold` | 32 | Brightness change (0 to 255) required for a pixel to count as changed |
| `--min-area` | 4000 | Minimum contour area in px² |
| `--major-area` | 45000 | Total contour area that escalates an alert to MAJOR |
| `--confirm-frames` | 5 | Frames a target must persist before alerting |
| `--min-travel` | 60 | Pixels a target must travel from its origin |
| `--process-width` | 480 | Width of the internal analysis frame |
| `--clip-cooldown` | 30 | Minimum seconds between two recorded clips |
| `--snapshot-cooldown` | 10 | Minimum seconds between two snapshots |
| `--save-dir` | `detections` | Output directory for evidence and the database |
| `--db` | *(save-dir)/clemvis.db* | Path of the event database |
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

Each stage is a separate module with one job, and the decision layers never touch the camera, files, or sound on their own. That is why they can be tested without hardware.

### Project structure

```
clemvis/
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

The test suite runs without a camera. Vision tests prepare NumPy frames with known content and check the pipeline output. Tracking tests replay synthetic motion over hundreds of frames, including a curtain that oscillates for 200 frames and must never be confirmed. State machine tests supply timestamps directly and replay whole intrusion timelines deterministically.

```bash
pytest -q
```

```
.......................................                          [100%]
39 passed in 10.46s
```

The synthetic scene doubles as ground truth for the evaluation benchmark, so end-to-end behaviour is tested rather than assumed.

---

## Design decisions

**Adaptive background over frame differencing.** Comparing two consecutive frames flags a moving object twice, once where it was and once where it is, and leaves a ghost trail behind it. CLEMVIS keeps a running average of the empty scene and compares against that instead. The model only learns while the scene is calm, so a stationary person is not absorbed.

**Evidence saved without clutter.** Recorded footage carries only a UTC timestamp rather than boxes and telemetry. Evidence should show the scene. The diagnostic overlay exists for tuning and can be switched on explicitly with `--annotate-evidence`.

**Vision at reduced resolution.** Analysing a downscaled frame costs a fraction of the computation. Coordinates are rescaled back to full-frame units, so all thresholds stay in real pixels and detection accuracy is unaffected.

**A state machine over scattered flags.** Explicit states and transitions make every rule inspectable and testable, instead of behaviour emerging unpredictably from interacting global variables.

**SQLite over a database server.** An embedded database gives durable, transactional, queryable storage with no deployment overhead, which fits a single-device application. It can be swapped for a client-server database later if the project ever needs one.

---

## Limitations

These boundaries are part of the design, so they are stated openly:

- **Fixed camera.** The system assumes a static viewpoint; panning or moving the camera breaks the background model.
- **Detection, not recognition system.** CLEMVIS reports that something moved, where it happened, and which direction it went. It does not classify the object. A pet and a person produce the same alert.
- **Nearest-neighbour tracking.** Targets are linked by proximity, so two objects that overlap for several frames can swap identities.
- **Benchmark scope.** The evaluation numbers validate the detection logic against a controlled synthetic scene. Real deployments need per-site tuning of the area and travel thresholds.
- **Local only.** Evidence is written to local disk; there is no streaming, remote notification or web interface in this version.

---

## Possible future work

Areas I would like to explore next include:

- OpenCV MOG2 or KNN background subtraction for outdoor environments
- optical flow for more detailed velocity estimates
- person classification to distinguish people from other moving objects
- a Kalman filter for predicted target movement
- improved assignment when several targets are present
- MQTT or ROS 2 event publishing
- Raspberry Pi camera support
- a graphical tool for drawing zones


---

## Contributions

Issues and pull requests are welcome. The modules are kept small on purpose — each one is short enough to read and understand in one sitting — so please keep new features focused the same way.

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## License

Released under the [MIT License](LICENSE).

The motion detection approach follows the standard background-subtraction workflow documented in the [OpenCV](https://docs.opencv.org/) computer vision library.
