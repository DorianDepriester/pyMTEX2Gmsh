"""Loop handling and Gmsh ``.geo`` sequence writing.

Ports of the Matlab private functions ``uniqueLoops.m``, ``borderLoop.m``,
``writeSequence.m`` and ``printMin.m``.
"""

from __future__ import annotations

import numpy as np

from .graph import euler_path


def unique_loops(grains) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Remove duplicated line loops.

    ``grains`` is the grain table of a :class:`~pymtex2gmsh.GmshGeo` (a sequence
    of objects with ``outer_loop`` and ``inner_loops`` attributes of signed
    segment ids). Returns ``(line_loops, plane_surfaces)`` where
    ``plane_surfaces[i]`` gives, for grain ``i``, the indices (1-based) of its
    outer loop then its inner loops in ``line_loops``.
    """
    n_grains = len(grains)
    plane_surface: list[np.ndarray] = []
    line_loops: list[np.ndarray] = []
    for i in range(n_grains):
        outer_loop = np.asarray(grains[i].outer_loop, dtype=np.int64)
        inner_loops = grains[i].inner_loops
        seq = np.zeros(1 + len(inner_loops), dtype=np.int64)
        j = _find_loop(line_loops, outer_loop)
        if j is None:
            line_loops.append(outer_loop)
            j = len(line_loops)
        seq[0] = j
        for k, inner in enumerate(inner_loops):
            inner = np.asarray(inner, dtype=np.int64)
            jj = _find_loop(line_loops, inner)
            if jj is None:
                line_loops.append(inner)
                jj = len(line_loops)
            seq[1 + k] = jj
        plane_surface.append(seq)
    return line_loops, plane_surface


def _find_loop(line_loops: list[np.ndarray], loop: np.ndarray):
    key = np.sort(np.abs(loop))
    for j, existing in enumerate(line_loops, start=1):
        if np.array_equal(key, np.sort(np.abs(existing))):
            return j
    return None


def border_loop(G) -> np.ndarray:
    """Signed segment ids of the outer boundary of the ROI, as a closed loop."""
    border = np.asarray(G.interfaces["ROI Border"], dtype=np.int64).ravel()
    flat = np.concatenate([np.asarray(G.segments[i - 1], dtype=np.int64).ravel() for i in border])
    F = np.column_stack([flat[0::2], flat[1::2]])
    p = euler_path(F - 1)[0] + 1
    n_seg = len(p) - 1
    seg_list = np.zeros(n_seg, dtype=np.int64)
    for i in range(n_seg):
        hit = np.flatnonzero((F[:, 0] == p[i]) & (F[:, 1] == p[i + 1]))
        if len(hit):
            seg_list[i] = border[hit[0]]
        else:
            hit = np.flatnonzero((F[:, 1] == p[i]) & (F[:, 0] == p[i + 1]))
            seg_list[i] = -border[hit[0]]
    return seg_list


def write_sequence(lines: list[str], title: str, idx, seq) -> None:
    """Append a Gmsh sequence definition, e.g. ``Line(3)={1:5,7};``.

    Consecutive (increasing or decreasing) ids are compressed into ranges.
    ``idx`` may be ``None`` (anonymous entity), an int, or a string.
    """
    seq = np.asarray(seq, dtype=np.int64).ravel()
    if len(seq) == 0:
        return
    if idx is None or idx == "":
        head = f"{title}{{{int(seq[0])}"
    elif isinstance(idx, str):
        head = f"{title}({idx})={{{int(seq[0])}"
    else:
        head = f"{title}({int(idx)})={{{int(seq[0])}"
    n = len(seq)
    if n == 1:
        lines.append(head + "};")
        return
    body = ""
    pending = None
    n_pend = 0
    for k in range(1, n):
        last = k == n - 1
        if seq[k] == seq[k - 1] + 1 or seq[k] == seq[k - 1] - 1:
            if last:
                body += f":{int(seq[k])}}};"
            else:
                pending = int(seq[k])
                n_pend += 1
        else:
            if pending is None:
                body += f",{int(seq[k])}"
            elif n_pend > 1:
                body += f":{pending},{int(seq[k])}"
            else:
                body += f",{pending},{int(seq[k])}"
            if last:
                body += "};"
            pending = None
            n_pend = 0
    lines.append(head + body)


def print_min(names) -> str:
    """Gmsh expression taking the minimum of the given field names, e.g. ``Min(a,b)``."""
    if isinstance(names, str):
        return names
    names = list(names)
    if len(names) == 1:
        return names[0]
    return f"Min({names[0]}," + print_min(names[1:]) + ")"
