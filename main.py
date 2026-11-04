#!/usr/bin/env python3
"""Convenience launcher: `python main.py` ≡ `python -m clemvis`."""
from clemvis.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
