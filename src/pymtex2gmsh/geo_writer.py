"""Generation of Gmsh ``.geo`` files from a :class:`GmshGeo`.

Port of the geometry-dumping part of the Matlab ``@gmshGeo/mesh.m``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .local_size import LocalSize
from .loops import border_loop, print_min, unique_loops, write_sequence

VALID_ELEMENT_TYPES = ("Wedge", "Hex", "Tet", "Tri", "Quad", "HexOnly", "QuadOnly")

_ELEMENT_TYPE_ALIASES = {
    "tetrahedron": "Tet",
    "hexahedron": "Hex",
    "brick": "Hex",
    "triangular": "Tri",
    "quadrangular": "Quad",
    "hexahedrononly": "HexOnly",
    "brickonly": "HexOnly",
    "quadrangularonly": "QuadOnly",
}


def normalize_element_type(element_type: str) -> str:
    et = str(element_type)
    low = et.lower()
    if low in _ELEMENT_TYPE_ALIASES:
        return _ELEMENT_TYPE_ALIASES[low]
    for valid in VALID_ELEMENT_TYPES:
        if et.lower() == valid.lower():
            return valid
    raise ValueError(
        f"Unrecognized element type {element_type!r}. It can be: {', '.join(repr(v) for v in VALID_ELEMENT_TYPES)}."
    )


@dataclass
class MeshOptions:
    """Meshing options, mirroring the name-value arguments of the Matlab ``mesh``."""

    element_size: float = 0.0
    thickness: float = 0.0
    gradient: float = 0.0
    element_type: str = "Wedge"
    element_order: int = 1
    curvature: int = 0
    medium: tuple = (0.0, 0.0, 0.0)
    medium_element_size: float = 0.0
    grain_prefix: str = "Grain_"
    verbosity: int = 4
    partition: int = 0
    periodic: str = "none"
    local_size: LocalSize | None = None
    layers: int = 1
    grain_boundaries_only: bool = False
    externalize_gbs: str = ""

    @classmethod
    def from_kwargs(cls, kwargs: dict) -> "MeshOptions":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(kwargs) - known
        if unknown:
            raise TypeError(f"Unknown mesh option(s): {', '.join(sorted(unknown))}")
        return cls(**kwargs)


def _fmt(x) -> str:
    return f"{float(x):g}"


def write_geo(G, path: str, o: MeshOptions, a_seg: np.ndarray | None = None) -> None:
    """Write the Gmsh ``.geo`` file describing the geometry of ``G``."""
    elem_type = normalize_element_type(o.element_type)
    mesh3d = elem_type in ("Hex", "Wedge", "Tet", "HexOnly")

    default_element_size = float(o.element_size) if o.element_size else G.resolution
    medium = tuple(float(v) for v in o.medium) != (0.0, 0.0, 0.0)
    medium_element_size = float(o.medium_element_size)
    if medium_element_size and not medium:
        raise ValueError("Specify the size of the embedding medium first with option 'medium'.")
    slope = float(o.gradient)
    thickness = float(np.atleast_1d(o.thickness)[0]) if o.thickness else default_element_size
    if o.thickness and not mesh3d:
        raise ValueError(f"{elem_type} elements are not compatible with a 2D mesh (thickness given).")
    curv = int(o.curvature)
    if curv and slope:
        import warnings

        warnings.warn(
            "Non constant element sizes at boundaries are inconsistent with the 'gradient' option. "
            "You might get unexpected results."
        )

    segments = [np.asarray(s, dtype=np.int64) for s in G.segments]
    vtx = np.asarray(G.V, dtype=float)

    # The microstructure is embedded in a medium
    if medium:
        roi = G.size().roi
        dx, dy = float(o.medium[0]), float(o.medium[1])
        if mesh3d:
            dz = float(o.medium[2])
            dmin = min((dx - roi[0]) / 2, (dy - roi[1]) / 2, dz - thickness)
            if dmin < 0:
                raise ValueError(
                    f"The dimensions of the medium must be larger than that of the ROI ({roi[0]:g} {roi[1]:g} {thickness:g})."
                )
        else:
            dz = 0.0
            dmin = min((dx - roi[0]) / 2, (dy - roi[1]) / 2)
            if dmin < 0:
                raise ValueError(f"The dimensions of the medium must be larger than that of the ROI ({roi[0]:g} {roi[1]:g}).")
        xmin, xmax = vtx[:, 0].min(), vtx[:, 0].max()
        ymin, ymax = vtx[:, 1].min(), vtx[:, 1].max()
        center = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2])
        corners = np.array(
            [
                center + np.array([-dx, -dy]) / 2,
                center + np.array([-dx, dy]) / 2,
                center + np.array([dx, dy]) / 2,
                center + np.array([dx, -dy]) / 2,
            ]
        )
        n_vtx = len(vtx)
        vtx = np.vstack([vtx, corners])
        new_segments = np.array(
            [[n_vtx + 1, n_vtx + 2], [n_vtx + 2, n_vtx + 3], [n_vtx + 3, n_vtx + 4], [n_vtx + 4, n_vtx + 1]],
            dtype=np.int64,
        )
        for row in new_segments:
            segments.append(row.copy())
        if medium_element_size == 0:
            q = 1.5
            n = np.log(1 + dmin / default_element_size * (q - 1)) / np.log(q)
            medium_element_size = default_element_size * q**n

    # Numbering the line loops
    line_loops, plane_surface = unique_loops(G.grains)

    n_vtx = len(vtx)
    vtx_used = np.zeros(n_vtx + 1, dtype=bool)
    for s in segments:
        vtx_used[s] = True
    vtx_used = vtx_used[1:]

    n_segments = len(segments)
    n_loops = len(line_loops)
    n_surfaces = len(G.grains)

    thickness_name = "th"
    default_element_size_name = "e_def"
    medium_thickness_name = "th_med"
    medium_element_size_name = "e_med"

    lines: list[str] = []
    lines.append(f"// File generated with pyMTEX2Gmsh on {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    lines.append("// Mesh parameters")
    lines.append(f"{thickness_name}={_fmt(thickness)};")
    lines.append(f"{default_element_size_name}={_fmt(default_element_size)};")
    if medium:
        lines.append(f"{medium_thickness_name}={_fmt(dz)};")
        lines.append(f"{medium_element_size_name}={_fmt(medium_element_size)};")

    # Check if local size is requested
    local_size = o.local_size
    local_size_names: list[str] = []
    A = np.zeros((n_vtx, 0), dtype=bool)
    if local_size is not None and len(local_size):
        cols = []
        for i, entry in enumerate(local_size):
            name = f"e_Grain_{entry.grain_id}"
            local_size_names.append(name)
            lines.append(f"{name}={_fmt(entry.size_at_boundaries)};")
            g = next((gr for gr in G.grains if gr.id == entry.grain_id), None)
            if g is None:
                raise ValueError(f"No grain with id {entry.grain_id} in the geometry.")
            out_seg_ids = np.abs(np.asarray(g.outer_loop, dtype=np.int64))
            in_seg_ids = (
                np.concatenate([np.abs(np.asarray(l, dtype=np.int64)) for l in g.inner_loops])
                if g.inner_loops
                else np.zeros(0, dtype=np.int64)
            )
            seg_ids = np.concatenate([out_seg_ids, in_seg_ids])
            vtx_ids = np.unique(np.concatenate([np.asarray(segments[s - 1]) for s in seg_ids]))
            col = np.zeros(n_vtx, dtype=bool)
            col[vtx_ids - 1] = True
            cols.append(col)
        A = np.column_stack(cols)

    lines.append('\nSetFactory("OpenCASCADE");')

    # Vertices
    lines.append("\n// Vertices")
    for i in range(1, n_vtx + 1):
        if not vtx_used[i - 1]:
            continue
        x, y = vtx[i - 1]
        if medium:
            size_expr = medium_element_size_name
        elif A.size and A[i - 1].any():
            size_expr = print_min([local_size_names[j] for j in np.flatnonzero(A[i - 1])])
        else:
            size_expr = default_element_size_name
        lines.append(f"Point({i})={{{_fmt(x)},{_fmt(y)},0,{size_expr}}};")

    # (B-)Splines
    lines.append("\n// Grain boundaries")
    for i, seg in enumerate(segments, start=1):
        if len(seg) == 2:
            write_sequence(lines, "Line", i, seg)
        elif len(seg) == 3 and curv != 0:
            write_sequence(lines, "Spline", i, seg)
        else:
            write_sequence(lines, "BSpline", i, seg)

    if o.grain_boundaries_only:
        # The user only requests the grain boundaries
        k = 1
        for name, seg_ids in G.interfaces.items():
            lines.append("\n// 2D Interfaces")
            surfaces = np.asarray(seg_ids, dtype=np.int64)
            n_surf = len(surfaces)
            line_seq: list[str] = []
            write_sequence(line_seq, "Line", None, surfaces)
            lines.append(f"Extrude {{0,0,{thickness_name}}}{{\n\t{line_seq[0].rstrip()}\n\tLayers{{{int(o.layers)}}};}}")
            lines.append(f'Physical Surface("{name}")={{{k}:{k + n_surf}}};')
            k = k + n_surf + 1
    else:
        # Closed loops
        lines.append("\n// Closed Loops")
        for i, loop in enumerate(line_loops, start=1):
            write_sequence(lines, "Line Loop", i, loop)

        # Surfaces
        lines.append("\n// Grains")
        for i, ps in enumerate(plane_surface, start=1):
            write_sequence(lines, "Plane Surface", i, ps)

        if elem_type in ("Hex", "Quad", "HexOnly", "QuadOnly"):
            lines.append("\n// Quadrangular elements")
            lines.append(f"Recombine Surface{{1:{n_surfaces}}};")

        # Extrusions
        if mesh3d:
            lines.append("\n// 3D geometry")
            lines.append(f"Extrude {{0,0,{thickness_name}}}{{\n\tSurface{{1:{n_surfaces}}};\n\tLayers{{{int(o.layers)}}};" + ("\tRecombine;" if elem_type != "Tet" else "") + "\n}")

        # Add surrounding medium (if requested)
        if medium:
            n_surfaces_tot = n_surfaces + 1
            lines.append("\n// Add surrounding medium")
            idx_shift = 10000

            bl = border_loop(G)
            if dz > thickness:  # Add medium below the ROI
                lines.append(f"L[]=Extrude{{0,0,{thickness_name}-{medium_thickness_name}}}{{\n\t")
                write_sequence(lines, "Line", None, np.abs(bl))
                lines.append("};")
                n_loops += 1
                seq = []
                for i in range(len(bl)):
                    j = (i) * 4
                    seq.append(f"L[{j}]" if bl[i] > 0 else f"-L[{j}]")
                lines.append(f"Line Loop({n_loops + idx_shift})={{{','.join(seq)}}};")
                id_surface_loop = n_surfaces + 1
                write_sequence(lines, "Plane Surface", id_surface_loop + idx_shift + 1, [n_loops + idx_shift])
                side = [f"L[{i * 4 - 3}]," for i in range(1, len(bl) + 1)]
                lines.append(
                    f"Surface Loop({id_surface_loop + idx_shift + 1})={{\n\t"
                    + "".join(side)
                    + f"\n\t1:{n_surfaces},{id_surface_loop + idx_shift + 1}\n}};"
                )
                write_sequence(lines, "Volume", n_surfaces + 1, [id_surface_loop + idx_shift + 1])
                n_surfaces_tot += 2

            # Create a volume surrounding the ROI
            n_loops += 1
            write_sequence(lines, "Line Loop", n_loops + 2 * idx_shift + 1, np.arange(n_segments - 3, n_segments + 1))
            n_loops += 1
            write_sequence(lines, "Line Loop", n_loops + 2 * idx_shift + 1, bl)
            if mesh3d:
                write_sequence(lines, "Plane Surface", n_surfaces + 2 + 2 * idx_shift, [n_loops - 1 + 2 * idx_shift + 1, n_loops + 2 * idx_shift + 1])
            else:
                write_sequence(lines, "Plane Surface", n_surfaces + 1 + 2 * idx_shift, [n_loops - 1 + 2 * idx_shift + 1, n_loops + 2 * idx_shift + 1])
            if dz > thickness:
                lines.append(f"Extrude {{0,0,{thickness_name}-{medium_thickness_name}}}{{\n\tSurface{{{n_surfaces + 2 + 2 * idx_shift}}};\n}}")
            if mesh3d:
                lines.append(
                    f"Extrude {{0,0,{thickness_name}}}{{\n\tSurface{{{n_surfaces + 2 + 2 * idx_shift}}};\n\tLayers{{1}}; Recombine;\n}}"
                )

            lines.append(f"Characteristic Length {{1:{n_vtx - 4}}} = {default_element_size_name};")

        # Physical groups
        groupname = "Volume" if mesh3d else "Surface"
        ids = np.array([g.id for g in G.grains], dtype=np.int64)
        lines.append("\n// Sets")
        if np.array_equal(ids, np.arange(1, n_surfaces + 1)):
            lines.append(f"For k In {{1:{n_surfaces}}}")
            if not o.grain_prefix:
                lines.append(f"\tPhysical {groupname}(k)={{k}};")
            else:
                lines.append(f'\tPhysical {groupname}(Sprintf("{o.grain_prefix}%g",k))={{k}};')
            lines.append("EndFor")
        else:
            for i, gid in enumerate(ids, start=1):
                if not o.grain_prefix:
                    lines.append(f"Physical {groupname}({int(gid)})={{{i}}};")
                else:
                    lines.append(f'Physical {groupname}("{o.grain_prefix}_{int(gid)}")={{{i}}};')
        if medium:
            if mesh3d:
                write_sequence(lines, f"Physical {groupname}", '"Medium"', np.arange(n_surfaces + 1, n_surfaces_tot + 1))
            else:
                write_sequence(lines, f"Physical {groupname}", '"Medium"', [n_surfaces + 1 + 2 * idx_shift])

    if a_seg is not None and len(a_seg):
        lines.append("\n// Periodicity conditions")
        for row in np.atleast_2d(a_seg):
            lines.append(f"Periodic Line {{{int(row[0])}}} = {{{int(row[1])}}};")

    # Mesh
    lines.append("\n// Mesh")
    if not o.grain_boundaries_only:
        lines.append("Mesh.SubdivisionAlgorithm=0;")
        if elem_type in ("HexOnly", "QuadOnly"):
            mesh_algo = 8
        else:
            mesh_algo = 7 if (slope != 0 and not mesh3d) else 6
        lines.append(f"Mesh.Algorithm={mesh_algo};")
        lines.append("Mesh.CharacteristicLengthExtendFromBoundary=1;")
        lines.append(f"Mesh.ElementOrder={int(o.element_order)};")
        if curv != 0:
            lines.append("Mesh.CharacteristicLengthFromCurvature = 1;")
            lines.append(f"Mesh.MinimumCirclePoints = {curv}; // points per 2*pi")
        if slope != 0 or (local_size is not None and np.any(local_size.slopes != 0)):
            id_field = _add_attractor_field(lines, 1, n_segments, default_element_size, slope)
            if local_size is not None and len(local_size):
                restrict_fields = []
                surface_list = np.arange(1, n_surfaces + 1)
                ls_ids = local_size.grain_ids
                surfaces_with_def_size = surface_list[~np.isin(ids, ls_ids)]
                id_field = _add_restrict_field(lines, id_field + 1, surfaces_with_def_size)
                restrict_fields.append(id_field)
                for entry in local_size:
                    id_field = _add_attractor_field(lines, id_field + 1, n_segments, entry.size_at_boundaries, entry.slope)
                    surf_ls = surface_list[np.isin(ids, [entry.grain_id])]
                    id_field = _add_restrict_field(lines, id_field + 1, surf_ls)
                    restrict_fields.append(id_field)
                lines.append(f"Field[{id_field + 1}] = Min;")
                lines.append(f"Field[{id_field + 1}].FieldsList = {{{','.join(str(int(f)) for f in restrict_fields)}}};")
                id_field += 1
            lines.append(f"Background Field={id_field};")
            lines.append("Mesh.CharacteristicLengthExtendFromBoundary=0;")
            lines.append("Mesh 2;")
            lines.append("Mesh.CharacteristicLengthExtendFromBoundary=1;")
        if elem_type in ("HexOnly", "QuadOnly"):
            lines.append("Mesh.RecombineAll = 0;")
            lines.append("Mesh.SaveParametric = 0;")
            lines.append("Mesh.RecombinationAlgorithm = 0;")
            lines.append("Mesh.SecondOrderLinear = 1;")
            lines.append("Mesh.SubdivisionAlgorithm=1;")
            lines.append(f"Mesh.MeshSizeFactor={_fmt(1)};")
            lines.append("RefineMesh;")
            if slope == 0.0:
                lines.append("Mesh 2;")
        if medium:
            lines.append("Mesh.Algorithm3D=4;")

    if mesh3d and not o.grain_boundaries_only:
        lines.append("Mesh 3;")
    else:
        lines.append("Mesh 2;")

    if o.partition:
        lines.append("\n// Partition mesh")
        lines.append(f"PartitionMesh {int(o.partition)};")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _add_attractor_field(lines: list[str], id_field: int, n_segments: int, def_size: float, slope: float) -> int:
    lines.append(f"Field[{id_field}] = Distance;")
    lines.append(f"Field[{id_field}].EdgesList ={{1:{n_segments}}};")
    id_field += 1
    lines.append(f"Field[{id_field}] = MathEval;")
    lines.append(f'Field[{id_field}].F = "F{id_field - 1}*{_fmt(slope)}+{_fmt(def_size)}";')
    return id_field


def _add_restrict_field(lines: list[str], id_field: int, surface_list) -> int:
    lines.append(f"Field[{id_field}] = Restrict;")
    lines.append(f"Field[{id_field}].InField = {id_field - 1};")
    sl = ",".join(str(int(s)) for s in np.atleast_1d(surface_list))
    lines.append(f"Field[{id_field}].SurfacesList = {{{sl}}};")
    return id_field
