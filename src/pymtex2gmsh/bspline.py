"""B-spline approximation of a set of knots.

Port of the Matlab ``BSpline.m`` and ``bspline_auto.m`` (de Boor's algorithm).
"""

from __future__ import annotations

import numpy as np


def bspline(knots: np.ndarray, order: int = 3, nint: int = 10, periodic: bool = False) -> np.ndarray:
    """B-spline approximation from a set of coordinates.

    ``knots[i, j]`` is the ``j``-th coordinate of the ``i``-th knot.
    ``order`` is the spline order (polynomial degree ``order - 1``, default 3
    for quadratic B-splines), ``nint`` the number of points per interval.
    With ``periodic=True``, periodic conditions are used at the end knots.
    """
    knots = np.asarray(knots, dtype=float)
    knots_in = knots
    if periodic:
        knots = np.vstack([knots[-order:], knots, knots[:order]])

    p, q = knots.shape
    if p <= 2:
        return knots.copy()

    n = (nint - 1) * (p - 1) + 1  # overall number of queried points
    y = np.linspace(0, 1, n)

    order = min(order, p)
    t = np.concatenate([np.zeros(order - 1), np.linspace(0, 1, p - order + 2), np.ones(order - 1)])

    BS = np.zeros((n, q))
    m_min, m_max = 1, n  # 1-based bounds of the periodic part, as in the Matlab code
    for m in range(n - 1):
        t0 = y[m]
        k = int(np.searchsorted(t, t0, side="right") - 1)  # 0-based, last index with t[k] <= t0
        if k >= p:
            return BS[:m]
        Xl = np.zeros((order, order, q))
        Xl[:, 0, :] = knots[k - order + 1 : k + 1, :]
        if k <= order:
            m_min = max(m + 1, m_min)
        if k >= p - order + 1:
            m_max = min(m + 1, m_max)
        for i in range(1, order):
            for j in range(i, order):
                num = t0 - t[k - order + j + 1]
                if num == 0:
                    wt = 0.0
                else:
                    s = t[k + j - i + 1] - t[k - order + j + 1]
                    wt = num / s
                Xl[j, i, :] = (1 - wt) * Xl[j - 1, i - 1, :] + wt * Xl[j, i - 1, :]
        BS[m, :] = Xl[order - 1, order - 1, :]
    BS[-1, :] = knots[-1, :]

    if periodic:
        BS = BS[m_min - 1 : m_max - 1, :]
        if BS.shape[0] == 0:
            # Degenerate (too short) periodic input: fall back to the knots
            return knots_in.copy()
        BS[-1, :] = BS[0, :]
    return BS


def bspline_auto(V: np.ndarray) -> np.ndarray:
    """Automatic choice of the B-spline order for a polyline ``V``."""
    V = np.asarray(V, dtype=float)
    if np.array_equal(V[0], V[-1]):
        order = 3 if len(V) == 4 else 4
        return bspline(V[:-1], order=order, periodic=True)
    return bspline(V, order=4)
