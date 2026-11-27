import numpy as np
import pytest
from scipy.interpolate import BSpline

from pymtex2gmsh.bspline import bspline, bspline_auto


@pytest.mark.parametrize("p", [3, 4, 6, 9])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_bspline_matches_scipy(p, order):
    if order > p:
        pytest.skip("order reduced to p")
    rng = np.random.default_rng(42)
    knots = rng.normal(size=(p, 2))
    nint = 10
    BS = bspline(knots, order=order, nint=nint)
    t = np.concatenate([np.zeros(order - 1), np.linspace(0, 1, p - order + 2), np.ones(order - 1)])
    n = (nint - 1) * (p - 1) + 1
    y = np.linspace(0, 1, n)
    ref = np.column_stack([BSpline(t, knots[:, j], order - 1, extrapolate=False)(y) for j in range(2)])
    ref[0] = knots[0]
    ref[-1] = knots[-1]
    np.testing.assert_allclose(BS, ref, atol=1e-12)
    assert BS.shape == (n, 2)


def test_bspline_endpoints():
    rng = np.random.default_rng(1)
    knots = rng.normal(size=(7, 2))
    BS = bspline(knots, order=3)
    np.testing.assert_allclose(BS[0], knots[0])
    np.testing.assert_allclose(BS[-1], knots[-1])


def test_bspline_periodic_closed():
    rng = np.random.default_rng(2)
    knots = rng.normal(size=(8, 2))
    BS = bspline(knots, order=3, periodic=True)
    np.testing.assert_allclose(BS[0], BS[-1])


def test_bspline_auto_closed_and_open():
    closed = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    BS = bspline_auto(closed)
    np.testing.assert_allclose(BS[0], BS[-1])
    open_poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    BS2 = bspline_auto(open_poly)
    np.testing.assert_allclose(BS2[0], open_poly[0])
    np.testing.assert_allclose(BS2[-1], open_poly[-1])
