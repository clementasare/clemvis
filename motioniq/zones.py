"""
Feature 2 — configurable detection zones (regions of interest).

A rectangle defined in NORMALISED coordinates (0.0–1.0) so a zone layout
works at any camera resolution. Each zone carries a severity:

    normal  → ordinary alert
    major   → escalated alert ("restricted area")
    ignore  → motion here is discarded entirely (point the camera at a
              busy road or a tree and simply mask it out)

Default layout splits the view into vertical thirds:

    ┌───────────┬───────────┬───────────┐
    │ A         │ B         │ C         │
    │ Entrance  │ Main area │ Restricted│
    │ normal    │ normal    │ MAJOR     │
    └───────────┴───────────┴───────────┘

Why this matters for robotics: a region of interest lets the system make
decisions based on WHERE something happens, not merely whether it happened —
and masking irrelevant regions is the cheapest false-positive filter there is.
"""
from __future__ import annotations

from dataclasses import dataclass

SEVERITIES = ("ignore", "normal", "major")


@dataclass(slots=True, frozen=True)
class Region:
    name: str          # short id, e.g. "C"
    label: str         # human label, e.g. "Restricted area"
    severity: str      # ignore | normal | major
    x0: float          # normalised rect, 0..1
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"severity must be one of {SEVERITIES}, got {self.severity!r}"
            )
        if not (0.0 <= self.x0 < self.x1 <= 1.0 and 0.0 <= self.y0 < self.y1 <= 1.0):
            raise ValueError(f"region {self.name!r} has an invalid rectangle")

    def pixel_rect(self, width: int, height: int) -> tuple[int, int, int, int]:
        """(x, y, w, h) of this zone for a frame of the given size."""
        x = int(self.x0 * width)
        y = int(self.y0 * height)
        return x, y, int(self.x1 * width) - x, int(self.y1 * height) - y

    def contains(self, cx: float, cy: float, width: int, height: int) -> bool:
        """Is the pixel point (cx, cy) inside this zone?"""
        nx = cx / max(1, width)
        ny = cy / max(1, height)
        return self.x0 <= nx < self.x1 and self.y0 <= ny < self.y1


def default_regions() -> list[Region]:
    return [
        Region("A", "Entrance", "normal", 0.00, 0.0, 0.33, 1.0),
        Region("B", "Main area", "normal", 0.33, 0.0, 0.66, 1.0),
        Region("C", "Restricted area", "major", 0.66, 0.0, 1.00, 1.0),
    ]


def region_at(
    regions: list[Region], cx: float, cy: float, width: int, height: int
) -> Region | None:
    """First zone containing the point, or None if it falls outside them all."""
    for region in regions:
        if region.contains(cx, cy, width, height):
            return region
    return None


def parse_regions(spec: str) -> list[Region]:
    """
    Parse a CLI zone spec:

        --zones "A:Entrance:normal:0,0,0.3,1; C:Vault:major:0.7,0,1,1"

    Fields: name:label:severity:x0,y0,x1,y1   (label may be omitted)
    """
    regions: list[Region] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(":")]
        if len(parts) == 3:                      # name:severity:rect
            name, severity, rect = parts
            label = name
        elif len(parts) == 4:                    # name:label:severity:rect
            name, label, severity, rect = parts
        else:
            raise ValueError(
                f"bad zone {chunk!r} — use name:label:severity:x0,y0,x1,y1"
            )
        try:
            x0, y0, x1, y1 = (float(v) for v in rect.split(","))
        except ValueError as exc:
            raise ValueError(f"bad rectangle in zone {chunk!r}") from exc
        regions.append(Region(name, label, severity.lower(), x0, y0, x1, y1))
    if not regions:
        raise ValueError("no zones parsed from spec")
    return regions
