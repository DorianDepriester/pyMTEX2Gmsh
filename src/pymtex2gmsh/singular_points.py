"""Detection of the singular points of a grain map.

Port of the Matlab ``findSingularPoints.m``: triple junctions, quadruple (or
higher order) junctions, corners of the ROI and double points on the border of
the ROI. All returned indices are 1-based (Gmsh convention), i.e. they index
rows of ``GmshGeo.V`` starting at 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mtex_compat import grain_vertices, vtx_on_bounds


@dataclass
class SingularPoints:
    """Indices (1-based) of the singular points of a geometry."""

    triple_points: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    quadruple_points: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    corners: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    double_points_on_border: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    symmetric: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def as_list(self) -> list[np.ndarray]:
        return [
            self.triple_points,
            self.quadruple_points,
            self.corners,
            self.double_points_on_border,
        ]

    def all(self) -> np.ndarray:
        return np.unique(np.concatenate([a for a in self.as_list() + [self.symmetric] if len(a)]))


def find_singular_points(grains) -> SingularPoints:
    """List the singular points of ``grains`` (an ``mtex.grain2d`` object).

    The singular points can be triple junctions, quadruple junctions (or of
    even higher order), corners of the ROI, or double points at the borders of
    the ROI. Returned indices are 1-based.
    """
    gB = grains.boundary
    F = np.asarray(gB.F, dtype=np.int64)
    grainId = np.asarray(gB.grainId, dtype=np.int64)
    V = grain_vertices(grains)
    n_vtx = len(V)

    # Triple points (0-based vertexId in mtex -> 1-based here)
    tp = np.asarray(grains.triplePoints.vertexId, dtype=np.int64) + 1

    # Quadruple points: vertices of degree > 3 touching more than 2 grains,
    # i.e. sum((I_VF * I_FG) == 2, 2) > 2 with MTEX's incidence matrices
    deg = np.bincount(F.ravel(), minlength=n_vtx)
    n_g = int(grainId.max(initial=0)) + 1
    v_pair = np.concatenate([F[:, 0], F[:, 0], F[:, 1], F[:, 1]])
    g_pair = np.concatenate([grainId[:, 0], grainId[:, 1], grainId[:, 0], grainId[:, 1]])
    key = v_pair * (n_g + 1) + g_pair
    _, counts = np.unique(key, return_counts=True)
    u_key = np.unique(key)
    i_vg = np.zeros((n_vtx, n_g), dtype=bool)
    i_vg[u_key // (n_g + 1), u_key % (n_g + 1)] = counts == 2
    it_p = (i_vg.sum(axis=1) > 2) & (deg > 3)
    qp = np.flatnonzero(it_p) + 1

    # Corners of the ROI: closest vertices to the corners are moved onto them
    corners = np.zeros((2, 2), dtype=np.int64)
    mins, maxs = V.min(axis=0), V.max(axis=0)
    mean_v2v = float(np.mean(np.asarray(grains.perimeter) / np.array([len(p) for p in grains.poly])))
    tol = mean_v2v / 10
    Vc = V.copy()
    for i in range(2):
        for j in range(2):
            c = np.array([mins[i], maxs[j]])
            d2 = (Vc[:, 0] - c[0]) ** 2 + (Vc[:, 1] - c[1]) ** 2
            id_dmin = int(np.argmin(d2))
            if np.sqrt(d2[id_dmin]) < tol:
                corners[i, j] = id_dmin + 1
                Vc[id_dmin] = c
    corners = corners[corners != 0]

    # Double points on the border of the ROI
    border = vtx_on_bounds(gB) + 1
    it_p_border = np.flatnonzero(deg > 2) + 1
    dpb = np.intersect1d(it_p_border, border)
    dpb = dpb[~np.isin(dpb, corners)]

    return SingularPoints(
        triple_points=np.asarray(tp, dtype=np.int64),
        quadruple_points=np.asarray(qp, dtype=np.int64),
        corners=np.asarray(corners, dtype=np.int64),
        double_points_on_border=np.asarray(dpb, dtype=np.int64),
    )
