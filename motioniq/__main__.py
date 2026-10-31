"""Enables `python -m motioniq` as an entry point."""
from .cli import main

raise SystemExit(main())
