"""The :class:`GmshGeo` class: a Gmsh-readable description of grains.

Port of the Matlab ``@gmshGeo`` class. Construct it from an ``mtex.grain2d``
object (e.g. ``grains = mtex.calcGrains(ebsd)``)::

    import mtex
    from pymtex2gmsh import GmshGeo

    ebsd = mtex.mtexdata('small')
    grains = mtex.calcGrains(ebsd[ebsd.isIndexed])
    G = GmshGeo(grains)
    G.mesh('small.msh')
    G.savegeo('geometry.geo')
    G.export_grain_props('grains.csv')
"""

from __future__ import annotations

import copy
import os
import tempfile
from typing import NamedTuple

import numpy as np

from .geo_writer import MeshOptions, write_geo
from .gmsh_runner import run_gmsh
from .local_size import LocalSize
from .measure import segment_lengths
from .mtex_compat import grain_vertices
from .model import Grain
from .regularize import regularize as _regularize
from .roi import add_symmetry as _add_symmetry
from .roi import fix_rectangular_roi as _fix_rectangular_roi
from .segments import compute_segments
from .singular_points import SingularPoints


class GeoSize(NamedTuple):
    """Result of :meth:`GmshGeo.size`: number of grains and ROI dimensions."""

    number_of_grains: int
    roi: np.ndarray


def _check_grains(grains) -> None:
    from mtex import grain2d

    if not isinstance(grains, grain2d):
        raise TypeError("Input argument must be an mtex.grain2d object (e.g. from mtex.calcGrains(ebsd)).")


class GmshGeo:
    """Gmsh-readable description of a grain2d object.

    Single indexing selects grains: ``G[5]`` keeps only the 6th grain (Python
    0-based indexing), ``G[[0, 2, 5]]`` a list of grains, and ``G['Forsterite']``
    all the grains of the named phase.

    Attributes:
        V: ``(n, 2)`` array of the vertex coordinates (BSpline knots).
        segments: list of vertex-index sequences (1-based) defining the BSplines.
        grains: list of :class:`Grain` (id, phase, loops, Euler angles).
        singular_points: :class:`SingularPoints` (triple points, corners, ...).
        interfaces: dict mapping an interface name (``'ROI Border'`` or
            ``'Phase1 - Phase2'``) to the array of its segment ids.
        resolution: approximate spatial resolution (mean vertex-to-vertex distance).
    """

    def __init__(self, grains, rectangular_roi: bool = True):
        _check_grains(grains)
        import mtex

        segment_seqs, outer_loops, inner_loops, sp, phase_of_segment = compute_segments(grains)
        self.V = grain_vertices(grains)
        self.segments = segment_seqs
        mineral_list = list(grains.mineralList)
        phase_ids = np.asarray(grains.phaseId, dtype=np.int64)
        phi1, Phi, phi2 = mtex.Euler(grains.meanRotation, "Bunge")
        self.grains: list[Grain] = [
            Grain(
                id=int(grains.id[i]),
                phase=mineral_list[int(phase_ids[i])],
                outer_loop=np.asarray(outer_loops[i], dtype=np.int64),
                inner_loops=[np.asarray(l, dtype=np.int64) for l in inner_loops[i]],
                phi1=float(np.asarray(phi1).ravel()[i]),
                Phi=float(np.asarray(Phi).ravel()[i]),
                phi2=float(np.asarray(phi2).ravel()[i]),
            )
            for i in range(len(grains))
        ]
        interfaces: dict[str, list[int]] = {}
        for seg_id, pid in enumerate(phase_of_segment, start=1):
            i, j = int(min(pid)), int(max(pid))
            if i <= 0:
                name = "ROI Border"
            else:
                name = f"{mineral_list[i]} - {mineral_list[j]}"
            interfaces.setdefault(name, []).append(seg_id)
        self.interfaces: dict[str, np.ndarray] = {
            name: np.asarray(ids, dtype=np.int64) for name, ids in interfaces.items()
        }
        self.singular_points: SingularPoints = sp
        self.resolution = float(
            np.mean(np.asarray(grains.perimeter, dtype=float) / np.array([len(p) for p in grains.poly]))
        )
        if rectangular_roi:
            self.fix_rectangular_roi()

    # ------------------------------------------------------------------ basics

    def __len__(self) -> int:
        return len(self.grains)

    def __getitem__(self, key) -> "GmshGeo":
        if isinstance(key, str):
            rows = [i for i, g in enumerate(self.grains) if g.phase == key]
        elif isinstance(key, (int, np.integer)):
            rows = [int(key)]
        elif isinstance(key, slice):
            rows = list(range(len(self.grains)))[key]
        else:
            rows = [int(k) for k in np.atleast_1d(key)]
        for r in rows:
            if r < 0 or r >= len(self.grains):
                raise IndexError(f"Index {r} out of range for {len(self.grains)} grains.")
        return self._subset(rows)

    def _subset(self, rows: list[int]) -> "GmshGeo":
        out = copy.deepcopy(self)
        out.grains = [self.grains[r] for r in rows]
        if not out.grains:
            out.interfaces = {}
            return out
        seg_ids = np.unique(
            np.concatenate(
                [np.abs(np.asarray(g.outer_loop)) for g in out.grains]
                + [np.abs(np.asarray(l)) for g in out.grains for l in g.inner_loops]
            )
        )
        remap = np.zeros(len(self.segments) + 1, dtype=np.int64)
        remap[seg_ids] = np.arange(1, len(seg_ids) + 1)
        out.segments = [np.asarray(self.segments[s - 1], dtype=np.int64) for s in seg_ids]
        for g in out.grains:
            g.outer_loop = np.sign(g.outer_loop) * remap[np.abs(g.outer_loop)]
            g.inner_loops = [np.sign(l) * remap[np.abs(l)] for l in g.inner_loops]
        new_interfaces = {}
        for name, ids in self.interfaces.items():
            new_ids = remap[np.asarray(ids, dtype=np.int64)]
            new_ids = new_ids[new_ids > 0]
            if len(new_ids):
                new_interfaces[name] = new_ids
        out.interfaces = new_interfaces
        return out

    def copy(self) -> "GmshGeo":
        return copy.deepcopy(self)

    def size(self) -> GeoSize:
        """Number of grains and size of the ROI as ``[dx, dy]``."""
        vtx = self.V
        roi = np.array([vtx[:, 0].max() - vtx[:, 0].min(), vtx[:, 1].max() - vtx[:, 1].min()])
        return GeoSize(number_of_grains=len(self.grains), roi=roi)

    def segment_lengths(self) -> np.ndarray:
        """Curvilinear length of each segment (B-spline approximation)."""
        return segment_lengths(self)

    # ------------------------------------------------------------------ geometry edits

    def fix_rectangular_roi(self) -> "GmshGeo":
        """Make the border of the ROI perfectly rectangular (in place)."""
        _fix_rectangular_roi(self)
        return self

    def add_symmetry(self, direction: str):
        """Add symmetry conditions for periodic meshing.

        Returns ``(G_sym, A_seg)`` with ``G_sym`` a new :class:`GmshGeo` and
        ``A_seg`` the pairs of symmetric segment ids.
        """
        return _add_symmetry(self, direction)

    def regularize(self, epsilon: float | None = None) -> "GmshGeo":
        """Collapse segments shorter than ``epsilon`` (default: resolution)."""
        return _regularize(self, epsilon)

    # ------------------------------------------------------------------ export

    def savegeo(self, output_file_path: str, **kwargs) -> int:
        """Dump the geometry into an ASCII ``.geo`` file (Gmsh syntax).

        Accepts the same options as :meth:`mesh`.
        """
        root, ext = os.path.splitext(str(output_file_path))
        filepath = str(output_file_path) if ext.lower() == ".geo" else root + ".geo"
        return self.mesh(filepath, **kwargs)

    def mesh(self, filepath: str, **kwargs) -> int:
        """Mesh the geometry with Gmsh and export the mesh to ``filepath``.

        If ``filepath`` ends with ``.geo``, only the geometry file is written
        (no meshing). Otherwise a temporary ``.geo`` is generated and executed
        with the Gmsh Python SDK, and the mesh is saved at ``filepath`` (the
        format is inferred from the extension, e.g. ``.msh``, ``.inp``).

        Options (keyword arguments):
            element_size: element size (default: the EBSD resolution).
            thickness: extrusion thickness for 3D meshes (default: element size).
            gradient: size gradient slope k, size = s + k*d from the boundaries.
            element_type: 'Wedge' (default), 'Hex', 'Tet', 'Tri', 'Quad',
                'HexOnly', 'QuadOnly' (aliases 'Tetrahedron', 'Hexahedron',
                'Brick', 'Triangular', 'Quadrangular', ... accepted).
            element_order: 1 (default) or 2 for quadratic elements.
            curvature: np nodes per 2*pi for curvature-based sizing (0=off).
            medium: (dx, dy[, dz]) embeds the ROI in a surrounding medium.
            medium_element_size: element size at the corners of the medium.
            grain_prefix: prefix for the physical group names ('Grain_' default;
                empty string for plain numbering).
            verbosity: Gmsh verbosity 0-10 (default 4).
            partition: create p partitions for parallel computing (0=off).
            periodic: 'x', 'y', 'both' or 'none' (default).
            local_size: a :class:`LocalSize` for per-grain element sizes.
            layers: number of element layers through the thickness (default 1).
            grain_boundaries_only: mesh only the grain boundaries (surfacic).
            externalize_gbs: path of an additional surfacic GB-only mesh file.

        Returns 0 on success.
        """
        o = MeshOptions.from_kwargs(kwargs)
        if o.externalize_gbs:
            rest = {k: v for k, v in kwargs.items() if k != "externalize_gbs"}
            self.mesh(o.externalize_gbs, grain_boundaries_only=True, **rest)

        filepath = str(filepath)
        export_geo = os.path.splitext(filepath)[1].lower() == ".geo"
        periodic = str(o.periodic).lower()
        if periodic in ("x", "y", "both"):
            G, a_seg = self.add_symmetry(periodic)
        elif periodic == "none":
            G, a_seg = self, None
        else:
            raise ValueError("Wrong argument for the periodic option. It can be 'x', 'y', 'both' or 'none'.")

        if export_geo:
            path_to_geo = filepath
        else:
            fd, path_to_geo = tempfile.mkstemp(suffix=".geo", prefix="pymtex2gmsh_")
            os.close(fd)
        try:
            write_geo(G, path_to_geo, o, a_seg=a_seg)
            if export_geo:
                return 0
            status = run_gmsh(path_to_geo, filepath, verbosity=o.verbosity)
        finally:
            if not export_geo and os.path.exists(path_to_geo):
                os.remove(path_to_geo)
        return status

    def export_grain_props(self, filename: str, *props) -> None:
        """Export grain properties as tab-separated ASCII (CSV-like) data.

        By default exports GrainID, Phase and the Euler angles (radians).
        ``props`` can be any of 'GrainID', 'Phase', 'PhaseID', 'Euler',
        'Rodrigues', or 'PRISMS' alone (alias for GrainID, Rodrigues, PhaseID,
        suitable for PRISMS-Plasticity).
        """
        import mtex

        if not props:
            flags = ["GrainID", "Phase", "Euler"]
        elif len(props) == 1 and str(props[0]).upper() == "PRISMS":
            flags = ["GrainID", "Rodrigues", "PhaseID"]
        else:
            flags = [str(p) for p in props]
            valid = {"GrainID", "Phase", "PhaseID", "Euler", "Rodrigues"}
            if "PRISMS" in [f.upper() for f in flags]:
                raise ValueError("'PRISMS' cannot be used together with other arguments.")
            if not set(flags) <= valid:
                raise ValueError(
                    "Wrong property name. It can be any combination of: "
                    + ", ".join(sorted(valid))
                    + ", or 'PRISMS' alone."
                )

        columns: list[tuple[str, np.ndarray]] = []

        def add(flag: str) -> None:
            if flag == "GrainID":
                columns.append(("GrainID", np.array([g.id for g in self.grains], dtype=np.int64)))
            elif flag == "Phase":
                columns.append(("Phase", np.array([g.phase for g in self.grains], dtype=object)))
            elif flag == "Euler":
                add("phi1")
                add("Phi")
                add("phi2")
            elif flag in ("phi1", "Phi", "phi2"):
                columns.append((flag, np.array([getattr(g, flag) for g in self.grains], dtype=float)))
            elif flag == "Rodrigues":
                rot = mtex.rotation.byEuler(
                    np.array([g.phi1 for g in self.grains]),
                    np.array([g.Phi for g in self.grains]),
                    np.array([g.phi2 for g in self.grains]),
                    convention="Bunge",
                )
                rodr = mtex.Rodrigues(rot)
                columns.append(("r_x", np.asarray(rodr.x, dtype=float)))
                columns.append(("r_y", np.asarray(rodr.y, dtype=float)))
                columns.append(("r_z", np.asarray(rodr.z, dtype=float)))
            elif flag == "PhaseID":
                phases = np.array([g.phase for g in self.grains], dtype=object)
                phase_list = np.unique(phases)
                phase_id = np.array([int(np.flatnonzero(phase_list == p)[0]) + 1 for p in phases])
                columns.append(("PhaseID", phase_id.astype(np.int64)))
                print(f"In {filename}, the phases are numbered as follows:")
                for i, p in enumerate(phase_list, start=1):
                    print(f"  {i}: {p}")

        for f in flags:
            add(f)

        with open(filename, "w") as fh:
            fh.write("\t".join(name for name, _ in columns) + "\n")
            n = len(self.grains)
            for i in range(n):
                row = []
                for _, col in columns:
                    v = col[i]
                    row.append(str(v) if isinstance(v, str) else (f"{v:g}" if isinstance(v, (int, np.integer)) else f"{float(v):.17g}"))
                fh.write("\t".join(row) + "\n")

    # ------------------------------------------------------------------ plotting

    def plot(self, ax=None, **kwargs):
        """Plot the geometry (segments) with matplotlib."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        for seg in self.segments:
            pts = self.V[np.asarray(seg, dtype=np.int64) - 1]
            ax.plot(pts[:, 0], pts[:, 1], color="k", linewidth=0.8, **kwargs)
        ax.set_aspect("equal")
        return ax

    def plot_singular_points(self, ax=None):
        """Plot the singular points over the geometry."""
        import matplotlib.pyplot as plt

        if ax is None:
            ax = self.plot()
        sp = self.singular_points
        styles = {
            "triple_points": ("o", "tab:red", "triple points"),
            "quadruple_points": ("s", "tab:green", "quadruple points"),
            "corners": ("*", "tab:blue", "ROI corners"),
            "double_points_on_border": ("^", "tab:orange", "double points on border"),
            "symmetric": ("x", "tab:purple", "symmetric points"),
        }
        for attr, (marker, color, label) in styles.items():
            ids = np.asarray(getattr(sp, attr), dtype=np.int64)
            if len(ids):
                pts = self.V[ids - 1]
                ax.scatter(pts[:, 0], pts[:, 1], marker=marker, c=color, label=label, zorder=5)
        return ax

    def __repr__(self) -> str:
        s = self.size()
        return (
            f"GmshGeo with {s.number_of_grains} grains, "
            f"{len(self.segments)} segments, {len(self.V)} vertices, "
            f"ROI {s.roi[0]:g} x {s.roi[1]:g}, resolution {self.resolution:g}"
        )
