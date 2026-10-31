"""
Command-line interface.

    python main.py                       # webcam + window + alarm
    python main.py --demo                # scripted scene, no hardware
    python main.py evaluate              # measure detection performance
    python main.py report                # print the event ledger
    python main.py --preset outdoor --no-alarm
    python main.py --zones "A:Door:normal:0,0,0.5,1; B:Safe:major:0.5,0,1,1"

argparse gives us --help, validation and sane defaults for free.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from . import __version__
from .config import PRESETS, DetectorConfig, apply_preset
from .ledger import EventLedger
from .zones import parse_regions


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="motioniq",
        description="Console motion-detection & evidence system (OpenCV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # source
    p.add_argument("--camera", type=int, default=0, help="camera device index")
    p.add_argument("--demo", action="store_true",
                   help="use the scripted synthetic scene (no hardware)")
    p.add_argument("--scenario", choices=("standard", "quick"),
                   default="standard", help="which synthetic script to play")
    p.add_argument("--headless", action="store_true",
                   help="console only, no GUI window")
    p.add_argument("--duration", type=float, metavar="SEC",
                   help="stop automatically after SEC seconds")

    # tuning
    p.add_argument("--preset", choices=sorted(PRESETS),
                   help="tested tuning bundle for the environment")
    p.add_argument("--threshold", type=int, metavar="0-255",
                   help="brightness-change threshold (lower = more sensitive)")
    p.add_argument("--min-area", type=int, metavar="PX",
                   help="ignore blobs smaller than this (px²)")
    p.add_argument("--major-area", type=int, metavar="PX",
                   help="area that escalates an alert to MAJOR")
    p.add_argument("--confirm-frames", type=int, metavar="N",
                   help="frames a target must persist before alerting")
    p.add_argument("--min-travel", type=int, metavar="PX",
                   help="px a target must travel from its origin "
                        "(raise this to ignore curtains/fans)")
    p.add_argument("--process-width", type=int, metavar="PX",
                   help="width the vision runs at (lower = faster)")

    # zones
    p.add_argument("--zones", type=str, metavar="SPEC",
                   help="zone layout: name:label:severity:x0,y0,x1,y1; ...")
    p.add_argument("--no-zones", action="store_true",
                   help="disable detection zones entirely")

    # capture behaviour
    p.add_argument("--clip-cooldown", type=float, metavar="SEC",
                   help="minimum seconds between two recorded clips")
    p.add_argument("--snapshot-cooldown", type=float, metavar="SEC",
                   help="minimum seconds between two snapshots")
    p.add_argument("--annotate-evidence", action="store_true",
                   help="burn the debug overlay into saved files "
                        "(default: clean footage with a timestamp only)")

    # output
    p.add_argument("--save-dir", type=Path, default=Path("detections"),
                   help="where snapshots / clips / ledger live")
    p.add_argument("--db", type=Path,
                   help="ledger path (default: <save-dir>/motioniq.db)")
    p.add_argument("--no-alarm", action="store_true", help="silent mode")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {__version__}")

    sub = p.add_subparsers(dest="command")

    rep = sub.add_parser("report", help="print the persisted event ledger")
    rep.add_argument("--db", type=Path, default=None)
    rep.add_argument("--limit", type=int, default=20)

    ev = sub.add_parser(
        "evaluate",
        help="score detection performance against the scripted scene",
    )
    ev.add_argument("--scenario", choices=("standard", "quick"),
                    default="standard")
    ev.add_argument("--preset", choices=sorted(PRESETS))
    ev.add_argument("--min-travel", type=int, metavar="PX")
    ev.add_argument("--confirm-frames", type=int, metavar="N")
    return p


def print_report(db_path: Path, limit: int) -> None:
    if not db_path.exists():
        print(f"No ledger at {db_path} — run the system first "
              f"(python main.py --demo).")
        return
    ledger = EventLedger(db_path)
    try:
        print("=" * 92)
        print(f"  motioniq LEDGER — {db_path}")
        print("=" * 92)
        print(f"  {ledger.count()} events: "
              + ",  ".join(f"{k}={v}" for k, v in ledger.summary().items()))
        print("-" * 92)
        print(f"  {'ID':>5}  {'TIME (UTC)':<26} {'TYPE':<13} "
              f"{'AREA':>8} {'TGT':>4}  NOTE")
        print("-" * 92)
        for row in reversed(ledger.recent(limit)):
            area = f"{int(row['total_area'])}" if row["total_area"] is not None else "-"
            tgt = f"{row['zone_count']}" if row["zone_count"] is not None else "-"
            print(f"  {row['id']:>5}  {row['ts_utc'][:26]:<26} {row['type']:<13} "
                  f"{area:>8} {tgt:>4}  {row['note'][:30]}")
        print("=" * 92)
    finally:
        ledger.close()


def _tune(cfg: DetectorConfig, args: argparse.Namespace) -> DetectorConfig:
    """Apply preset first, then explicit flags (flags always win)."""
    if getattr(args, "preset", None):
        cfg = apply_preset(cfg, args.preset)
    if getattr(args, "threshold", None) is not None:
        cfg.diff_threshold = args.threshold
    if getattr(args, "min_area", None) is not None:
        cfg.min_contour_area = args.min_area
    if getattr(args, "major_area", None) is not None:
        cfg.major_area = args.major_area
    if getattr(args, "confirm_frames", None) is not None:
        cfg.confirm_frames = args.confirm_frames
    if getattr(args, "min_travel", None) is not None:
        cfg.min_displacement = args.min_travel
    if getattr(args, "process_width", None) is not None:
        cfg.process_width = args.process_width
    if getattr(args, "clip_cooldown", None) is not None:
        cfg.clip_cooldown_seconds = args.clip_cooldown
    if getattr(args, "snapshot_cooldown", None) is not None:
        cfg.snapshot_cooldown_seconds = args.snapshot_cooldown
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "report":
        print_report(args.db or Path("detections") / "motioniq.db", args.limit)
        return 0

    if args.command == "evaluate":
        from .evaluate import run_evaluation

        cfg = DetectorConfig(
            demo=True,
            demo_scenario=args.scenario,
            demo_realtime=False,     # virtual clock: run as fast as the CPU can
            show_window=False,
            alarm_enabled=False,
        )
        cfg = _tune(cfg, args)
        print(run_evaluation(cfg).render())
        return 0

    cfg = DetectorConfig(
        camera_index=args.camera,
        demo=args.demo,
        demo_scenario=args.scenario,
        show_window=not args.headless,
        duration=args.duration,
        save_dir=args.save_dir,
        db_path=args.db,
        alarm_enabled=not args.no_alarm,
        annotate_evidence=args.annotate_evidence,
        zones_enabled=not args.no_zones,
    )
    if args.zones:
        cfg = replace(cfg, regions=parse_regions(args.zones))
    cfg = _tune(cfg, args)

    from .app import SurveillanceApp  # local import keeps `report` cv2-free

    return SurveillanceApp(cfg).run()
