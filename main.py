#!/usr/bin/env python3
"""Convenience launcher: `python main.py` ≡ `python -m motioniq`."""
from motioniq.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
