"""Data structures describing a meshable grain geometry."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Grain:
    """Gmsh-readable description of one grain.

    ``outer_loop`` and ``inner_loops`` hold signed segment indices (a negative
    id means the segment is traversed against its stored knot sequence);
    ``phi1``, ``Phi``, ``phi2`` are the Bunge Euler angles (radians) of the mean
    orientation of the grain.
    """

    id: int
    phase: str
    outer_loop: np.ndarray
    inner_loops: list = field(default_factory=list)
    phi1: float = 0.0
    Phi: float = 0.0
    phi2: float = 0.0
