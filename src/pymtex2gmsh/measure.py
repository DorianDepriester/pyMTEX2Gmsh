"""Geometric measures of a :class:`GmshGeo` (port of ``segmentLenghts.m``)."""

from __future__ import annotations

import numpy as np

from .bspline import bspline_auto


def segment_lengths(G) -> np.ndarray:
    """Curvilinear length of each segment.

    Segments are first approximated with B-splines, then the curvilinear length
    of each curve is computed.
    """
    lengths = np.zeros(len(G.segments))
    for i, seg in enumerate(G.segments):
        Vi = G.V[np.asarray(seg, dtype=np.int64) - 1]
        bs = bspline_auto(Vi)
        lengths[i] = np.linalg.norm(np.diff(bs, axis=0), axis=1).sum()
    return lengths
