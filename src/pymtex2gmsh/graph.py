"""Graph utilities: Eulerian paths of the grain boundary graph.

Port of the Matlab ``EulerPath.m`` (which relies on MTEX's ``EulerCycles``,
not available in the Python port of MTEX, hence reimplemented here with
Hierholzer's algorithm).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _euler_cycles(F: np.ndarray) -> list[np.ndarray]:
    """Decompose the undirected graph given by the edge list ``F`` into Eulerian cycles.

    ``F`` is an ``(n, 2)`` array of 0-based vertex indices. Returns a list of
    closed paths (first vertex == last vertex), each edge used exactly once.
    """
    F = np.asarray(F, dtype=np.int64).reshape(-1, 2)
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for e, (a, b) in enumerate(F):
        adj[int(a)].append((int(b), e))
        adj[int(b)].append((int(a), e))
    used = np.zeros(len(F), dtype=bool)
    cycles: list[np.ndarray] = []
    for v0 in sorted(adj):
        if all(used[e] for _, e in adj[v0]):
            continue
        stack = [v0]
        cyc: list[int] = []
        while stack:
            v = stack[-1]
            nxt = None
            for w, e in adj[v]:
                if not used[e]:
                    nxt = (w, e)
                    break
            if nxt is None:
                cyc.append(stack.pop())
            else:
                used[nxt[1]] = True
                stack.append(nxt[0])
        cyc = cyc[::-1]
        if len(cyc) > 1 and cyc[0] != cyc[-1]:
            cyc.append(cyc[0])
        cycles.append(np.asarray(cyc, dtype=np.int64))
    return cycles


def euler_path(F: np.ndarray) -> list[np.ndarray]:
    """Compute the Euler paths of a graph.

    ``F`` is an ``(n, 2)`` array defining the connectivity between nodes
    (0-based indices). Returns a list of closed loops (last node == first
    node), the largest one first (considered as the outer loop).

    Ambiguous paths (self-touching boundaries) are split so that the shortest
    loops are returned, whereas a plain Eulerian circuit would give the longest
    one.
    """
    F = np.asarray(F, dtype=np.int64).reshape(-1, 2)
    paths = _euler_cycles(F)

    new_paths: list[np.ndarray] = []
    for long_path in paths:
        short_path = long_path[:-1]
        u_path = np.unique(short_path)
        counts = np.array([np.count_nonzero(short_path == u) for u in u_path])
        dupl_id = np.flatnonzero(counts > 1)
        n_dupl = len(dupl_id)
        if n_dupl:
            new_loops: list[np.ndarray] = []
            dupl_dist = np.zeros(n_dupl)
            for i in range(n_dupl):
                dupl_dist[i] = np.diff(np.flatnonzero(short_path == u_path[dupl_id[i]])).sum()
            order = np.argsort(dupl_dist, kind="stable")
            for i in range(n_dupl):
                dupl_val = u_path[dupl_id[order[i]]]
                n_vtx = len(short_path)
                pos = np.flatnonzero(short_path == dupl_val)
                shortcut = (pos[0], pos[-1])
                new_loops.append(short_path[shortcut[0] : shortcut[1] + 1].copy())
                short_path = np.concatenate([short_path[: shortcut[0] + 1], short_path[shortcut[1] + 1 : n_vtx]])
            new_paths.append(np.concatenate([short_path, long_path[-1:]]))
            new_paths.extend(new_loops)
        else:
            new_paths.append(long_path.copy())

    loop_len = np.array([len(p) for p in new_paths])
    order = np.argsort(-loop_len, kind="stable")
    return [new_paths[i] for i in order]
