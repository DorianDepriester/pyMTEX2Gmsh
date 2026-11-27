"""Local element sizes in specified grains.

Port of the Matlab ``LocalSize.m`` class, intended to be passed as the
``local_size`` argument of :meth:`GmshGeo.mesh`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LocalSizeEntry:
    grain_id: int
    size_at_boundaries: float
    slope: float = 0.0


class LocalSize:
    """Local element sizes and size gradients in specific grains.

    ``LocalSize([5, 7], 30)`` sets the element size to 30 in grains labelled 5
    and 7. ``LocalSize([5, 7], [20, 30])`` uses a different size per grain.
    ``LocalSize(5, 0.1, 0.5)`` additionally prescribes a size gradient with
    slope 0.5, starting from 0.1 at the grain boundary.
    """

    def __init__(self, grain_ids, sizes, slope=0.0):
        ids = np.atleast_1d(np.asarray(grain_ids, dtype=np.int64)).ravel()
        sizes = _broadcast(np.asarray(sizes, dtype=float), len(ids), "sizes at boundaries")
        slopes = _broadcast(np.asarray(slope, dtype=float), len(ids), "slopes")
        self.entries = [
            LocalSizeEntry(int(ids[i]), float(sizes[i]), float(slopes[i])) for i in range(len(ids))
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, i) -> LocalSizeEntry:
        return self.entries[i]

    @property
    def grain_ids(self) -> np.ndarray:
        return np.array([e.grain_id for e in self.entries], dtype=np.int64)

    @property
    def sizes(self) -> np.ndarray:
        return np.array([e.size_at_boundaries for e in self.entries], dtype=float)

    @property
    def slopes(self) -> np.ndarray:
        return np.array([e.slope for e in self.entries], dtype=float)

    def __repr__(self) -> str:
        if not self.entries:
            return "Empty set for local element sizes"
        rows = "\n".join(
            f"  {e.grain_id:>6}  {e.size_at_boundaries:>12g}  {e.slope:>8g}" for e in self.entries
        )
        return f"LocalSize (GrainID, SizeAtBoundaries, Slope):\n{rows}"


def _broadcast(values: np.ndarray, n: int, what: str) -> np.ndarray:
    if values.size == 1:
        return np.full(n, float(values.ravel()[0]))
    if values.size != n:
        raise ValueError(f"The {what} must be either a unique value or an array of the same length as the grain IDs.")
    return values.astype(float).ravel()
