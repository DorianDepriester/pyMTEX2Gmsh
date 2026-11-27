"""ROI regularization and symmetry (periodicity) helpers.

Ports of the Matlab ``fixRectangularROI.m`` and ``addSymmetry.m``.
"""

from __future__ import annotations

import copy

import numpy as np


def fix_rectangular_roi(G) -> None:
    """Move some vertices so that the border of the ROI becomes rectangular.

    First computes the smallest rectangular box bounding all the geometry, then
    moves the outer vertices to the closest edge of the box. Also moves (or
    creates) 4 vertices so that the ROI has sharp corners. Modifies ``G`` in
    place.
    """
    V = G.V
    tol = G.resolution / 10
    border = np.asarray(G.interfaces["ROI Border"], dtype=np.int64).ravel()
    border_segs = [np.asarray(G.segments[i - 1], dtype=np.int64).ravel() for i in border]
    segments = G.segments
    mins, maxs = V.min(axis=0), V.max(axis=0)
    outer_loops = [np.asarray(g.outer_loop, dtype=np.int64) for g in G.grains]

    ids_vb = np.unique(np.concatenate(border_segs)) - 1  # 0-based rows of V

    # Move vertices to the border edges
    d = np.column_stack(
        [
            V[ids_vb, 0] - mins[0],
            V[ids_vb, 0] - maxs[0],
            V[ids_vb, 1] - mins[1],
            V[ids_vb, 1] - maxs[1],
        ]
    )
    a = np.argmin(np.abs(d), axis=1)
    edge_val = np.array([mins[0], maxs[0], mins[1], maxs[1]])
    for k in range(4):
        j = k // 2
        V[ids_vb[a == k], j] = edge_val[k]

    # Find/create sharp corners
    corners = np.zeros((2, 2), dtype=np.int64)
    for i in range(2):
        for j in range(2):
            C = np.array([(mins if i == 0 else maxs)[0], (mins if j == 0 else maxs)[1]])
            d2 = (V[ids_vb, 0] - C[0]) ** 2 + (V[ids_vb, 1] - C[1]) ** 2
            id_dmin = int(np.argmin(d2))
            if d2[id_dmin] < tol**2:
                corners[i, j] = ids_vb[id_dmin] + 1
                V[ids_vb[id_dmin]] = C
            else:
                # No candidate vertex found -> need to create a new one
                id_border_min = -1
                dmin = np.inf
                for k, seg_k in enumerate(border_segs):
                    V1 = V[seg_k[0] - 1]
                    V2 = V[seg_k[1] - 1]
                    u = V2 - V1
                    v = C - V1
                    u2 = float(u @ u)
                    proj = float(u @ v)
                    dist = abs(np.linalg.det(np.array([u, v]))) / np.sqrt(u2) if u2 > 0 else np.inf
                    if dist < dmin and 0 < proj < u2:
                        id_border_min = k
                        dmin = dist
                V = np.vstack([V, C[None, :]])
                G.V = V
                id_new_v = len(V)
                corners[i, j] = id_new_v
                id_seg_min = int(border[id_border_min])
                old_seg = np.asarray(segments[id_seg_min - 1], dtype=np.int64)
                segments[id_seg_min - 1] = np.array([old_seg[0], id_new_v])
                segments.append(np.array([id_new_v, old_seg[-1]], dtype=np.int64))
                id_new_seg = len(segments)
                for k in range(len(outer_loops)):
                    p = np.flatnonzero(outer_loops[k] == id_seg_min)
                    if len(p):
                        p = p[0]
                        outer_loops[k] = np.insert(outer_loops[k], p + 1, id_new_seg)
                border = np.append(border, id_new_seg)

    G.interfaces["ROI Border"] = border
    G.V = V
    for k, g in enumerate(G.grains):
        g.outer_loop = outer_loops[k]
    G.segments = segments
    G.singular_points.corners = corners.flatten(order="F")
    G.singular_points.corners = G.singular_points.corners[G.singular_points.corners != 0]
    _remove_degenerate_segments(G)


def _remove_degenerate_segments(G) -> None:
    """Drop the segments whose two endpoints were snapped to the same point.

    Snapping the border vertices onto the edges of the bounding box can make a
    border segment collapse onto a single vertex; such zero-length segments are
    removed from the segment list, the grain loops and the interfaces (recent
    Gmsh versions reject ``Line`` entities with identical end points).
    """
    degen = np.array(
        [i + 1 for i, s in enumerate(G.segments) if len(s) == 2 and s[0] == s[1]], dtype=np.int64
    )
    if not len(degen):
        return
    n_seg = len(G.segments)
    keep = np.ones(n_seg, dtype=bool)
    keep[degen - 1] = False
    remap = np.zeros(n_seg + 1, dtype=np.int64)
    remap[np.flatnonzero(keep) + 1] = np.arange(1, keep.sum() + 1)
    G.segments = [np.asarray(G.segments[i], dtype=np.int64) for i in np.flatnonzero(keep)]
    for g in G.grains:
        loop = np.sign(g.outer_loop) * remap[np.abs(g.outer_loop)]
        g.outer_loop = loop[loop != 0]
        new_inner = []
        for l in g.inner_loops:
            l2 = np.sign(l) * remap[np.abs(l)]
            if np.any(l2 != 0):
                new_inner.append(l2[l2 != 0])
        g.inner_loops = new_inner
    # Grains whose whole outer loop collapsed have no surface anymore: drop them
    G.grains = [g for g in G.grains if len(g.outer_loop) > 0]
    G.interfaces = {
        name: remap[np.asarray(ids, dtype=np.int64)].astype(np.int64)
        for name, ids in G.interfaces.items()
    }
    G.interfaces = {name: ids[ids > 0] for name, ids in G.interfaces.items() if np.any(ids > 0)}


def add_symmetry(G, direction: str):
    """Add symmetry conditions on the geometry for periodic meshing.

    ``direction`` is ``'x'``, ``'y'`` or ``'both'``. Returns ``(G_sym, A_seg)``
    where ``G_sym`` is a new :class:`GmshGeo` with dummy singular points inserted
    on opposite sides of the ROI, and ``A_seg`` the ``(n, 2)`` array of segment
    id pairs that are symmetric to each other.
    """
    direction = str(direction).lower()
    xsym = direction in ("x", "both")
    ysym = direction in ("y", "both")
    if not (xsym or ysym):
        raise ValueError("Wrong argument for the periodic direction. It can be 'x', 'y', 'both' or 'none'.")

    Gr = copy.deepcopy(G)
    fix_rectangular_roi(Gr)
    dpb = np.concatenate([Gr.singular_points.double_points_on_border, Gr.singular_points.symmetric])
    V = Gr.V
    Vdpb = V[dpb - 1].copy()

    npt = len(dpb)
    mins, maxs = Vdpb.min(axis=0), Vdpb.max(axis=0)
    tol = Gr.resolution / 10
    n_old_v = len(V)

    vsym: list[np.ndarray] = []
    a_v = np.zeros((0, 2), dtype=np.int64)
    for i in range(npt):
        new_coords = None
        if xsym:
            if abs(Vdpb[i, 0] - mins[0]) < tol:
                new_coords = np.array([maxs[0], Vdpb[i, 1]])
            elif abs(Vdpb[i, 0] - maxs[0]) < tol:
                new_coords = np.array([mins[0], Vdpb[i, 1]])
        if ysym:
            if abs(Vdpb[i, 1] - mins[1]) < tol:
                new_coords = np.array([Vdpb[i, 0], maxs[1]])
            elif abs(Vdpb[i, 1] - maxs[1]) < tol:
                new_coords = np.array([Vdpb[i, 0], mins[1]])
        if new_coords is not None:
            d2 = (Vdpb[:, 0] - new_coords[0]) ** 2 + (Vdpb[:, 1] - new_coords[1]) ** 2
            duplicate = d2 <= tol**2
            if not duplicate.any():
                vsym.append(new_coords)
                a_v = np.vstack([a_v, [dpb[i], n_old_v + len(vsym)]])
            else:
                dup = int(np.flatnonzero(duplicate)[0])
                Vdpb[dup] = new_coords
                hit = np.argwhere(a_v == dpb[i])
                if len(hit) == 0 or a_v[hit[0, 0], 1 - hit[0, 1]] != dpb[dup]:
                    a_v = np.vstack([a_v, [dpb[i], dpb[dup]]])

    vsym = np.asarray(vsym, dtype=float).reshape(-1, 2)
    n_new_v = len(vsym)

    # Split segments where vertices are added
    V = np.vstack([V, vsym])
    border = np.asarray(Gr.interfaces["ROI Border"], dtype=np.int64).ravel().copy()
    segments = Gr.segments
    outer_loops = [np.asarray(g.outer_loop, dtype=np.int64) for g in Gr.grains]
    for i in range(n_new_v):
        Vi = vsym[i]
        j = 0
        while j < len(border):
            id_seg = int(border[j])
            seg_j = np.asarray(segments[id_seg - 1], dtype=np.int64)
            A = V[seg_j[0] - 1]
            u = Vi - A
            v = V[seg_j[1] - 1] - A
            proj = float(u @ v)
            if abs(np.linalg.det(np.array([u, v]))) < 1e-10 and 0 < proj < float(v @ v):
                id_v = i + n_old_v + 1
                segments[id_seg - 1] = np.array([seg_j[0], id_v])
                id_new_seg = len(segments) + 1
                segments.append(np.array([id_v, seg_j[-1]], dtype=np.int64))
                border = np.append(border, id_new_seg)
                for k in range(len(outer_loops)):
                    p = np.flatnonzero(outer_loops[k] == id_seg)
                    if len(p):
                        outer_loops[k] = np.insert(outer_loops[k], p[0] + 1, id_new_seg)
                break
            j += 1

    # Compute adjacency between segments
    n_seg = len(border)
    a_seg = np.zeros((0, 2), dtype=np.int64)
    for i in range(n_seg):
        border_i = int(border[i])
        ref_vs = np.asarray(segments[border_i - 1], dtype=np.int64)
        oppo_seg = np.zeros(2, dtype=np.int64)
        ok = True
        for k in range(2):
            hit = np.argwhere(a_v == ref_vs[k])
            if len(hit) == 0:
                ok = False
                break
            oppo_seg[k] = a_v[hit[0, 0], 1 - hit[0, 1]]
        if ok and np.all(oppo_seg) and border_i not in a_seg:
            match = [
                idx + 1
                for idx, s in enumerate(segments)
                if np.array_equal(np.asarray(s), oppo_seg) or np.array_equal(np.asarray(s), oppo_seg[::-1])
            ]
            if match:
                a_seg = np.vstack([a_seg, [border_i, match[0]]])

    Gr.interfaces["ROI Border"] = border
    Gr.V = V
    Gr.V[dpb - 1] = Vdpb
    for k, g in enumerate(Gr.grains):
        g.outer_loop = outer_loops[k]
    Gr.segments = segments
    id_sym = np.arange(n_new_v, dtype=np.int64) + n_old_v + 1
    sp = Gr.singular_points
    sp.symmetric = np.concatenate([sp.symmetric, id_sym])
    return Gr, a_seg
