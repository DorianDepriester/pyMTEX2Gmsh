import numpy as np

from pymtex2gmsh.graph import euler_path


def test_euler_path_square():
    F = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    paths = euler_path(F)
    assert len(paths) == 1
    p = paths[0]
    assert p[0] == p[-1]
    assert len(p) == 5  # 4 edges + closing vertex


def test_euler_path_figure_eight_is_split():
    # two triangles sharing vertex 0: degrees are 4 at 0, so the Eulerian
    # circuit is ambiguous and must be split into the two short loops
    F = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [3, 4], [4, 0]])
    paths = euler_path(F)
    assert len(paths) == 2
    assert sorted(len(p) for p in paths) == [4, 4]
    for p in paths:
        assert p[0] == p[-1]
    covered = sum(len(p) - 1 for p in paths)
    assert covered == len(F)


def test_euler_path_disconnected():
    F = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]])
    paths = euler_path(F)
    assert len(paths) == 2
    assert sum(len(p) - 1 for p in paths) == len(F)
    # largest loop first
    assert len(paths[0]) >= len(paths[1])


def test_euler_path_covers_all_edges():
    rng = np.random.default_rng(0)
    # random even-degree graph: disjoint squares sharing vertices
    F = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [4, 5], [5, 6], [6, 0]])
    paths = euler_path(F)
    assert sum(len(p) - 1 for p in paths) == len(F)
