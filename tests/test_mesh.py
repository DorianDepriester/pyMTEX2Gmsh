"""Integration tests: actually run Gmsh on generated .geo files."""

import os

import numpy as np
import pytest

from pymtex2gmsh import GmshGeo, LocalSize
from conftest import requires_gmsh

pytestmark = requires_gmsh


def _mesh_stats(path, dim):
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.open(path)
    n_nodes, _, _ = gmsh.model.mesh.getNodes()
    n_el = 0
    for d, t in gmsh.model.getEntities(dim):
        _, tags, _ = gmsh.model.mesh.getElements(dim, t)
        n_el += sum(len(x) for x in tags)
    n_groups = len(gmsh.model.getPhysicalGroups(dim))
    gmsh.finalize()
    return len(n_nodes), n_el, n_groups


@pytest.fixture(scope="module")
def G(grains_synth):
    return GmshGeo(grains_synth)


def test_mesh_tri_2d(tmp_path, G):
    out = tmp_path / "mesh_tri.msh"
    assert G.mesh(str(out), element_type="Tri", element_size=8.0) == 0
    n_nodes, n_el, n_groups = _mesh_stats(str(out), 2)
    assert n_nodes > 100
    assert n_el > 100
    assert n_groups == len(G)


def test_mesh_wedge_3d(tmp_path, G):
    out = tmp_path / "mesh_wedge.msh"
    assert G.mesh(str(out), element_size=8.0) == 0  # Wedge by default
    n_nodes, n_el, n_groups = _mesh_stats(str(out), 3)
    assert n_el > 0
    assert n_groups == len(G)


def test_mesh_hex_and_tet(tmp_path, G):
    for et in ("Hex", "Tet"):
        out = tmp_path / f"mesh_{et}.msh"
        assert G.mesh(str(out), element_type=et, element_size=8.0) == 0
        _, n_el, _ = _mesh_stats(str(out), 3)
        assert n_el > 0


def test_mesh_quad(tmp_path, G):
    out = tmp_path / "mesh_quad.msh"
    assert G.mesh(str(out), element_type="Quad", element_size=8.0) == 0
    _, n_el, _ = _mesh_stats(str(out), 2)
    assert n_el > 0


def test_mesh_gradient(tmp_path, G):
    out = tmp_path / "mesh_grad.msh"
    assert G.mesh(str(out), element_type="Tri", element_size=4.0, gradient=0.5) == 0
    assert out.exists() and out.stat().st_size > 0


def test_mesh_curvature(tmp_path, G):
    out = tmp_path / "mesh_curv.msh"
    assert G.mesh(str(out), element_type="Tri", element_size=4.0, curvature=5) == 0


def test_mesh_periodic(tmp_path, G):
    for axis in ("x", "y", "both"):
        out = tmp_path / f"mesh_per_{axis}.msh"
        assert G.mesh(str(out), element_type="Tri", element_size=8.0, periodic=axis) == 0
        txt = G.savegeo(str(tmp_path / f"per_{axis}.geo"), element_type="Tri", element_size=8.0, periodic=axis)
        assert "Periodic Line" in (tmp_path / f"per_{axis}.geo").read_text()


def test_mesh_local_size(tmp_path, G):
    ids = [g.id for g in G.grains[:2]]
    ls = LocalSize(ids, 4.0, 0.3)
    out = tmp_path / "mesh_ls.msh"
    assert G.mesh(str(out), element_type="Tri", element_size=8.0, local_size=ls) == 0
    geo = tmp_path / "ls.geo"
    G.savegeo(str(geo), element_type="Tri", element_size=8.0, local_size=ls)
    txt = geo.read_text()
    assert "Field[" in txt and "Background Field" in txt
    assert f"e_Grain_{ids[0]}=" in txt


def test_mesh_medium(tmp_path, G):
    s = G.size().roi
    medium = (s[0] * 1.5, s[1] * 1.5, 8.0)
    out = tmp_path / "mesh_med.msh"
    assert G.mesh(str(out), element_type="Tet", element_size=8.0, medium=medium, medium_element_size=16.0) == 0
    n_nodes, n_el, n_groups = _mesh_stats(str(out), 3)
    assert n_el > 0
    assert n_groups == len(G) + 1  # grains + Medium


def test_mesh_grain_boundaries_only(tmp_path, G):
    out = tmp_path / "gb.msh"
    assert G.mesh(str(out), element_size=8.0, grain_boundaries_only=True) == 0
    assert out.exists() and out.stat().st_size > 0


def test_mesh_externalize_gbs(tmp_path, G):
    out = tmp_path / "mesh.msh"
    gb = tmp_path / "gb_only.msh"
    assert G.mesh(str(out), element_type="Tri", element_size=8.0, externalize_gbs=str(gb)) == 0
    assert gb.exists() and gb.stat().st_size > 0


def test_mesh_partition(tmp_path, G):
    out = tmp_path / "mesh_part.msh"
    assert G.mesh(str(out), element_type="Tri", element_size=8.0, partition=2) == 0
    assert out.exists() and out.stat().st_size > 0


def test_mesh_element_order_2(tmp_path, G):
    out = tmp_path / "mesh_o2.msh"
    assert G.mesh(str(out), element_type="Tri", element_size=8.0, element_order=2) == 0
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.open(str(out))
    types, _, _ = gmsh.model.mesh.getElements(2, 1)
    assert 9 in types  # second order triangles
    gmsh.finalize()


def test_mesh_abaqus_inp(tmp_path, G):
    out = tmp_path / "mesh.inp"
    assert G.mesh(str(out), element_type="Tri", element_size=8.0) == 0
    assert out.exists() and out.stat().st_size > 0


def test_mesh_small_dataset(tmp_path, grains_small):
    G = GmshGeo(grains_small)
    out = tmp_path / "small_tri.msh"
    assert G.mesh(str(out), element_type="Tri", element_size=200.0) == 0
    n_nodes, n_el, n_groups = _mesh_stats(str(out), 2)
    assert n_nodes > 500
    assert n_el > 500
    assert n_groups == len(G)
