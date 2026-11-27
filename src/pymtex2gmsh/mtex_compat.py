"""Small compatibility helpers around the Python port of MTEX (``mtex``).

These mirror the helper functions of the original Matlab toolbox
(``grainVertices.m``, ``grainNumPixel.m`` and ``Vtx_onBounds.m``) and hide the
small API differences of the ``mtex`` Python package.
"""

from __future__ import annotations

import numpy as np


def grain_vertices(grains) -> np.ndarray:
    """Coordinates of all grain vertices, as an ``(n, 2)`` array.

    The returned array is indexed the same way as ``grains.boundary.F``,
    ``grains.poly`` and ``grains.triplePoints`` (0-based, as in ``mtex``).
    """
    V = np.asarray(grains.V, dtype=float)
    return V[:, :2]


def set_grain_vertices(grains, V: np.ndarray):
    """Update the x, y coordinates of the grain vertices from an ``(n, 2)`` array."""
    V = np.asarray(V, dtype=float)
    grains.boundary.V[:, :2] = V
    return grains


def grain_num_pixel(grains) -> np.ndarray:
    """Number of pixels (measurements) in each grain."""
    if hasattr(grains, "numPixel"):
        return np.asarray(grains.numPixel)
    return np.asarray(grains.grainSize)


def vtx_on_bounds(gB) -> np.ndarray:
    """0-based indices of the vertices lying on the border of the map (ROI)."""
    outer_boundary_id = np.any(np.asarray(gB.grainId) == 0, axis=1)
    return np.unique(np.asarray(gB.F)[outer_boundary_id])
