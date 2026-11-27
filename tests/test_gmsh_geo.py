import os
import re

import numpy as np
import pytest

from pymtex2gmsh import GmshGeo, LocalSize


@pytest.fixture(scope="module")
def G(grains_synth):
    return GmshGeo(grains_synth)


def test_constructor(G, grains_synth):
    assert len(G) == len(grains_synth)
    assert len(G.segments) > 0
    assert G.V.shape[1] == 2
    assert G.resolution > 0
    assert "ROI Border" in G.interfaces
    ids = [g.id for g in G.grains]
    assert sorted(ids) == sorted(int(i) for i in grains_synth.id)


def test_size(G):
    s = G.size()
    assert s.number_of_grains == len(G)
    assert s.roi[0] > 0 and s.roi[1] > 0


def test_rectangular_roi(G):
    V = G.V
    assert np.isclose(V[:, 0].min(), G.V[:, 0].min())
    corners = G.singular_points.corners
    assert len(corners) == 4
    c = V[corners - 1]
    assert np.isclose(c[:, 0].min(), V[:, 0].min())
    assert np.isclose(c[:, 0].max(), V[:, 0].max())
    assert np.isclose(c[:, 1].min(), V[:, 1].min())
    assert np.isclose(c[:, 1].max(), V[:, 1].max())


def test_getitem_by_phase_and_index(G):
    phase = G.grains[0].phase
    sub = G[phase]
    assert all(g.phase == phase for g in sub.grains)
    assert 0 < len(sub) <= len(G)
    sub2 = G[[0, 1, 2]]
    assert len(sub2) == 3
    sub3 = G[1]
    assert len(sub3) == 1
    with pytest.raises(IndexError):
        G[len(G) + 5]


def test_segment_lengths(G):
    L = G.segment_lengths()
    assert L.shape == (len(G.segments),)
    assert np.all(L >= 0)
    assert np.any(L > 0)


def test_regularize(G):
    G2 = G.copy().regularize()
    assert len(G2.segments) <= len(G.segments)
    # loops still reference existing segments
    for g in G2.grains:
        assert np.abs(g.outer_loop).max() <= len(G2.segments)


def test_savegeo(tmp_path, G):
    path = tmp_path / "geom.geo"
    assert G.savegeo(str(path)) == 0
    txt = path.read_text()
    assert 'SetFactory("OpenCASCADE");' in txt
    assert re.search(r"^Point\(\d+\)=\{", txt, re.M)
    assert re.search(r"^(Line|BSpline|Spline)\(\d+\)=\{", txt, re.M)
    assert re.search(r"^Line Loop\(\d+\)=\{", txt, re.M)
    assert re.search(r"^Plane Surface\(\d+\)=\{", txt, re.M)
    assert re.search(r"Physical (Surface|Volume)", txt)
    assert "Mesh 2;" in txt or "Mesh 3;" in txt


def test_savegeo_appends_extension(tmp_path, G):
    path = tmp_path / "geom"
    G.savegeo(str(path))
    assert (tmp_path / "geom.geo").exists()


def test_export_grain_props(tmp_path, G):
    f = tmp_path / "props.csv"
    G.export_grain_props(str(f))
    lines = f.read_text().splitlines()
    assert lines[0].split("\t") == ["GrainID", "Phase", "phi1", "Phi", "phi2"]
    assert len(lines) == len(G) + 1

    f2 = tmp_path / "props_prisms.csv"
    G.export_grain_props(str(f2), "PRISMS")
    header = f2.read_text().splitlines()[0].split("\t")
    assert header == ["GrainID", "r_x", "r_y", "r_z", "PhaseID"]

    f3 = tmp_path / "props_rodr.csv"
    G.export_grain_props(str(f3), "GrainID", "Rodrigues", "PhaseID")
    header = f3.read_text().splitlines()[0].split("\t")
    assert header == ["GrainID", "r_x", "r_y", "r_z", "PhaseID"]


def test_local_size():
    ls = LocalSize([5, 7], 30)
    assert len(ls) == 2
    np.testing.assert_array_equal(ls.grain_ids, [5, 7])
    np.testing.assert_allclose(ls.sizes, [30, 30])
    np.testing.assert_allclose(ls.slopes, [0, 0])
    ls2 = LocalSize([5, 7], [20, 30], [0.5, 0.7])
    np.testing.assert_allclose(ls2.sizes, [20, 30])
    np.testing.assert_allclose(ls2.slopes, [0.5, 0.7])
    with pytest.raises(ValueError):
        LocalSize([5, 7], [1, 2, 3])


def test_add_symmetry(G):
    Gs, a_seg = G.add_symmetry("both")
    assert len(Gs.V) >= len(G.V)
    assert a_seg.ndim == 2 and a_seg.shape[1] == 2
    # either new symmetric vertices were created, or correspondences between
    # existing opposite vertices were found (perfectly symmetric map)
    assert len(Gs.singular_points.symmetric) > 0 or len(a_seg) > 0
    with pytest.raises(ValueError):
        G.add_symmetry("z")


def test_plot(tmp_path, G):
    import matplotlib

    matplotlib.use("Agg")
    ax = G.plot()
    ax = G.plot_singular_points(ax)
    fig = ax.figure
    fig.savefig(tmp_path / "geo.png")
    assert (tmp_path / "geo.png").exists()


def test_mesh_options_validation(G):
    with pytest.raises(ValueError):
        G.savegeo("/tmp/x.geo", element_type="Banana")
    with pytest.raises(TypeError):
        G.savegeo("/tmp/x.geo", unknown_option=3)
    with pytest.raises(ValueError):
        G.savegeo("/tmp/x.geo", periodic="z")
