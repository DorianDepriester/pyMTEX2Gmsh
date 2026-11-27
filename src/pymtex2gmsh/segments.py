"""Computation of the grain boundary segments between singular points.

Port of the Matlab ``computeSegments.m``: each grain boundary loop is split at
the singular points (triple junctions, corners, ...) into sequences of vertex
indices ("segments"), shared segments being stored only once (up to reversal,
hence the signed segment ids used in the loops).

All vertex indices handled here are 1-based (Gmsh convention); the 0-based
indices of ``mtex`` are shifted on entry.
"""

from __future__ import annotations

import numpy as np

from .graph import euler_path
from .mtex_compat import vtx_on_bounds
from .singular_points import SingularPoints, find_singular_points


def _add_sequence(segments: list, seq: np.ndarray, phase_id: np.ndarray) -> int:
    """Add ``seq`` to the segment list if not already present (up to reversal).

    Returns the signed index of the segment in the list (negative if the
    stored sequence is the reverse of ``seq``).
    """
    if len(seq) == 0:
        return 0
    seq = np.asarray(seq, dtype=np.int64)
    if seq[0] == seq[-1]:
        # Ensure that self loops are unique: reorder starting from the lowest vertex index
        id_min = int(np.argmin(seq))
        if id_min != 0:
            seq_ordered = np.roll(seq[:-1], -id_min)
            seq = np.concatenate([seq_ordered, seq_ordered[:1]])
    for idx, (seg_idx, _) in enumerate(segments, start=1):
        if len(seg_idx) != len(seq):
            continue
        if np.array_equal(seq, seg_idx):
            return idx
        if np.array_equal(seq[::-1], seg_idx):
            return -idx
    segments.append((seq, np.asarray(phase_id)))
    return len(segments)


def compute_segments(grains):
    """Compute the singular-point-to-singular-point segments defining each grain.

    Returns ``(segments, outer_loops, inner_loops, singular_points, phase_ids)``
    where ``segments`` is a list of ``(vertex_indices, phase_id_pair)`` with
    1-based vertex indices, ``outer_loops[i]`` / ``inner_loops[i]`` are lists of
    signed segment indices for grain ``i``, and ``singular_points`` is a
    :class:`SingularPoints`.
    """
    sp: SingularPoints = find_singular_points(grains)
    sp_all = sp.all()

    gB = grains.boundary
    F_all = np.asarray(gB.F, dtype=np.int64) + 1
    grain_id_all = np.asarray(gB.grainId, dtype=np.int64)
    phase_id_all = np.asarray(gB.phaseId, dtype=np.int64)

    ng = len(grains)
    grain_ids = np.asarray(grains.id, dtype=np.int64)

    partitions: list[list[list[np.ndarray]]] = []
    for ig in range(ng):
        mask = np.any(grain_id_all == grain_ids[ig], axis=1)
        F = F_all[mask] - 1  # back to 0-based for the graph routines
        polys = euler_path(F)
        surface: list[list[np.ndarray]] = []
        for poly in polys:
            poly = np.asarray(poly, dtype=np.int64) + 1
            sp_loc = sp_all[np.isin(sp_all, poly)]
            if len(sp_loc):
                if poly[-1] == poly[0]:
                    poly = poly[:-1]
                id_start = int(np.flatnonzero(poly == sp_loc[0])[0])
                poly = np.roll(poly, -id_start)
            if poly[-1] != poly[0]:
                poly = np.append(poly, poly[0])
            if len(sp_loc):
                segments: list[np.ndarray] = []
                seq = np.array([poly[0]], dtype=np.int64)
                id_poly = 1
                while id_poly < len(poly):
                    seq = np.append(seq, poly[id_poly])
                    if len(seq) > 1 and np.isin(poly[id_poly], sp_loc):
                        segments.append(seq)
                        seq = np.array([poly[id_poly]], dtype=np.int64)
                    id_poly += 1
                if len(seq) > 1:
                    segments.append(seq)
            else:
                segments = [poly]
            surface.append(segments)
        partitions.append(surface)

    # Indexing all segments
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    vtx_ids = vtx_on_bounds(gB) + 1

    outer_loops: list[np.ndarray] = []
    inner_loops: list[list[np.ndarray]] = []
    for i in range(ng):
        mask = np.any(grain_id_all == grain_ids[i], axis=1)
        F = F_all[mask]
        phase_ids = phase_id_all[mask]
        parti = partitions[i]
        in_loops: list[np.ndarray] = []
        for j, loopi in enumerate(parti):
            id_segments = np.zeros(len(loopi), dtype=np.int64)
            for k, segment in enumerate(loopi):
                a, b = int(segment[0]), int(segment[1])
                cand = np.flatnonzero(
                    ((F[:, 0] == a) & (F[:, 1] == b)) | ((F[:, 0] == b) & (F[:, 1] == a))
                )
                phase_id = phase_ids[cand[0]] if len(cand) else np.array([0, 0])
                if np.isin(segment, vtx_ids).all():
                    # All the vertices on the domain boundary: keep only the end points
                    segment = segment[[0, -1]]
                id_segments[k] = _add_sequence(segments, segment, phase_id)
            if j == 0:
                outer_loops.append(id_segments)
            else:
                in_loops.append(id_segments)
        inner_loops.append(in_loops)

    phase_of_segment = [pid for _, pid in segments]
    segment_seqs = [seq for seq, _ in segments]
    return segment_seqs, outer_loops, inner_loops, sp, phase_of_segment
