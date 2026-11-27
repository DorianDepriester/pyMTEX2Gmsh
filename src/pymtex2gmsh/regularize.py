"""Collapse of too-short segments (port of ``regularize.m``)."""

from __future__ import annotations

import numpy as np

from .measure import segment_lengths


def _find_root(parent, x):
    r = x
    while parent[r] != r:
        r = parent[r]
    return r


def regularize(G, epsilon: float | None = None):
    """Collapse the segments shorter than ``epsilon`` into vertices.

    The endpoints of each removed segment are merged into a single vertex
    located at the barycenter of its knots; all the remaining segments, the
    grain loops, the interfaces and the singular points are updated
    accordingly. ``epsilon`` defaults to the spatial resolution of ``G``.
    Modifies ``G`` in place and returns it.
    """
    if epsilon is None:
        epsilon = G.resolution

    seg = G.segments
    n_seg = len(seg)
    L = segment_lengths(G)
    short_segment = set(np.flatnonzero(L < epsilon).tolist())  # 0-based positions

    if not short_segment:
        return G

    outer_loops = [np.asarray(g.outer_loop, dtype=np.int64) for g in G.grains]
    inner_loops = [[np.asarray(l, dtype=np.int64) for l in g.inner_loops] for g in G.grains]

    border_segs = set()
    if "ROI Border" in G.interfaces:
        border_segs = set(int(s) for s in np.asarray(G.interfaces["ROI Border"]).ravel())
    border_vtx = set()
    if border_segs:
        for s in border_segs:
            border_vtx.update(int(v) for v in np.asarray(seg[s - 1]).ravel())

    collapsible = {i + 1 for i in short_segment} - border_segs  # 1-based segment ids

    # Protect the loops that would be left with less than 3 segments
    for i in range(len(G.grains)):
        loops = [outer_loops[i]] + inner_loops[i]
        for loop in loops:
            loop_abs = set(int(abs(s)) for s in loop)
            n_short = len(loop_abs & collapsible)
            if len(loop_abs) - n_short < 3:
                collapsible -= loop_abs

    # Protect the segments with both endpoints on the ROI border
    for k in list(collapsible):
        s = np.asarray(seg[k - 1], dtype=np.int64)
        if int(s[0]) in border_vtx and int(s[-1]) in border_vtx:
            collapsible.discard(k)

    if not collapsible:
        return G

    # Group the collapsible segments into chains of adjacent segments (union-find)
    col = sorted(collapsible)
    index = {s: i for i, s in enumerate(col)}
    parent = np.arange(len(col))
    for a_pos, ka in enumerate(col):
        s = np.asarray(seg[ka - 1], dtype=np.int64)
        for v in (int(s[0]), int(s[-1])):
            for kb in col[a_pos + 1 :]:
                sm = np.asarray(seg[kb - 1], dtype=np.int64)
                if int(sm[0]) == v or int(sm[-1]) == v:
                    ra, rb = _find_root(parent, a_pos), _find_root(parent, index[kb])
                    if ra != rb:
                        parent[rb] = ra
    root = np.array([_find_root(parent, i) for i in range(len(col))])

    # vmap maps each endpoint of a collapsed segment to the vertex of its chain
    V = G.V
    n_vtx = len(V)
    vmap = np.zeros(n_vtx + 1, dtype=np.int64)  # 1-based vertex ids
    new_vertices = np.zeros((0, 2))
    for r in np.unique(root):
        members = [col[i] for i in np.flatnonzero(root == r)]
        ends = set()
        for m in members:
            sm = np.asarray(seg[m - 1], dtype=np.int64)
            ends.add(int(sm[0]))
            ends.add(int(sm[-1]))
        b = [e for e in ends if e in border_vtx]
        if b:
            mid = b[0]
        else:
            knots = set()
            for m in members:
                knots.update(int(v) for v in np.asarray(seg[m - 1]).ravel())
            new_vertices = np.vstack([new_vertices, V[np.array(sorted(knots)) - 1].mean(axis=0)[None, :]])
            mid = n_vtx + len(new_vertices)
        for e in ends:
            vmap[e] = mid

    # Remove the collapsible segments from the loops
    for i in range(len(G.grains)):
        outer_loops[i] = outer_loops[i][~np.isin(np.abs(outer_loops[i]), list(collapsible))]
        inner_loops[i] = [l[~np.isin(np.abs(l), list(collapsible))] for l in inner_loops[i]]


    # Replace the collapsed endpoints in the endpoints of all the segments
    deg = np.zeros(n_seg, dtype=bool)
    for j in range(n_seg):
        k = np.asarray(seg[j], dtype=np.int64)
        if vmap[k[0]] and vmap[k[-1]] and vmap[k[0]] == vmap[k[-1]]:
            deg[j] = True
        if vmap[k[0]]:
            k[0] = vmap[k[0]]
        if vmap[k[-1]]:
            k[-1] = vmap[k[-1]]
        seg[j] = k

    # Remove the collapsed (and degenerate) segments and remap all the ids
    keep = np.ones(n_seg, dtype=bool)
    keep[[c - 1 for c in collapsible]] = False
    keep[deg] = False
    remap = np.zeros(n_seg + 1, dtype=np.int64)
    remap[np.flatnonzero(keep) + 1] = np.arange(1, keep.sum() + 1)
    seg = [seg[i] for i in np.flatnonzero(keep)]

    for i in range(len(G.grains)):
        loop = np.sign(outer_loops[i]) * remap[np.abs(outer_loops[i])]
        outer_loops[i] = loop[loop != 0]
        new_inner = []
        for l in inner_loops[i]:
            l2 = np.sign(l) * remap[np.abs(l)]
            new_inner.append(l2[l2 != 0])
        inner_loops[i] = new_inner

    interfaces = {}
    for name, ids in G.interfaces.items():
        new_ids = remap[np.asarray(ids, dtype=np.int64)]
        interfaces[name] = new_ids[new_ids > 0]
    G.interfaces = interfaces

    sp = G.singular_points
    for attr in ("triple_points", "quadruple_points", "corners", "double_points_on_border", "symmetric"):
        ids = np.asarray(getattr(sp, attr), dtype=np.int64)
        m = vmap[ids]
        ids = np.where(m != 0, m, ids)
        setattr(sp, attr, ids)

    G.segments = seg
    G.V = np.vstack([V, new_vertices])
    for i, g in enumerate(G.grains):
        g.outer_loop = outer_loops[i]
        g.inner_loops = inner_loops[i]
    return G
