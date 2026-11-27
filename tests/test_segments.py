import numpy as np

from pymtex2gmsh import compute_segments, find_singular_points
from pymtex2gmsh.mtex_compat import vtx_on_bounds


def _seg_ends(segments, cid):
    s = np.asarray(segments[abs(cid) - 1])
    return (int(s[0]), int(s[-1])) if cid > 0 else (int(s[-1]), int(s[0]))


def test_find_singular_points(grains_synth):
    sp = find_singular_points(grains_synth)
    assert len(sp.triple_points) > 0
    assert np.all(sp.triple_points >= 1)
    assert len(sp.double_points_on_border) > 0
    # corners of the ROI are found only after fix_rectangular_roi; here just check types
    assert sp.quadruple_points.ndim == 1


def test_compute_segments_loops_are_closed(grains_synth):
    segments, outer, inner, sp, phases = compute_segments(grains_synth)
    assert len(segments) == len(phases)
    assert len(outer) == len(grains_synth)
    for loops in [outer] + inner:
        for loop in loops:
            for l in ([loop] if not isinstance(loop, list) else loop):
                ends = [_seg_ends(segments, c) for c in l]
                assert all(
                    ends[i][1] == ends[(i + 1) % len(ends)][0] for i in range(len(ends))
                ), f"loop {l} is not closed"


def test_compute_segments_deduplicates(grains_synth):
    segments, outer, inner, sp, phases = compute_segments(grains_synth)
    seqs = [tuple(np.asarray(s).tolist()) for s in segments]
    assert len(seqs) == len(set(seqs))
    # each segment id used in loops exists
    all_ids = np.concatenate([np.abs(np.asarray(l)) for l in outer] + [np.abs(np.asarray(x)) for li in inner for x in li])
    assert all_ids.max() <= len(segments)


def test_vtx_on_bounds(grains_synth):
    gB = grains_synth.boundary
    vb = vtx_on_bounds(gB)
    outer = np.any(np.asarray(gB.grainId) == 0, axis=1)
    assert set(vb.tolist()) == set(np.unique(np.asarray(gB.F)[outer]).tolist())
