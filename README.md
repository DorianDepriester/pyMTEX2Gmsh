# pyMTEX2Gmsh

Generate a conforming mesh from EBSD data processed with Python and MTEX.

**pyMTEX2Gmsh** is a 100% Python port of the Matlab toolbox
[MTEX2Gmsh](https://github.com/DorianDepriester/MTEX2Gmsh) (Depriester & Kubler,
JOSS 2020, [doi:10.21105/joss.02094](https://doi.org/10.21105/joss.02094)). It
converts grains reconstructed with the [Python port of
MTEX](https://pypi.org/project/mtex/) (`pip install mtex`) into a Gmsh-readable
geometry, then meshes it with [Gmsh](https://gmsh.info) through its official
Python SDK (`pip install gmsh`) — no Matlab, no external binary required.

It is intended to perform Finite Element Analysis (FEA) at grain scale on
polycrystals imaged by EBSD.

## Install

```bash
pip install pymtex2gmsh[mesh,plot]   # gmsh SDK for meshing, matplotlib for plotting
```

Requirements: Python >= 3.10, `mtex` (the Python port of MTEX), `numpy`, `scipy`.
Meshing additionally needs the `gmsh` Python SDK; plotting needs `matplotlib`.

## Quickstart

```python
import mtex
from pymtex2gmsh import GmshGeo, LocalSize

ebsd = mtex.mtexdata('small')              # example dataset shipped with MTEX
grains = mtex.calcGrains(ebsd[ebsd.isIndexed])

G = GmshGeo(grains)                        # Gmsh-readable description of the grains
print(G.size())                            # number of grains and ROI size

G.mesh('small.msh')                        # 3D mesh of linear wedge elements (default)
G.mesh('small_tri.msh', element_type='Tri', element_size=100)   # 2D triangular mesh
G.savegeo('geometry.geo')                  # dump only the Gmsh geometry
G.export_grain_props('grains.csv')         # grain ids, phases, Euler orientations
```

A complete runnable script is provided in [`examples/example.py`](examples/example.py).

## Meshing options

`GmshGeo.mesh(filepath, **options)` writes the geometry to a temporary `.geo`
file, executes it with the Gmsh SDK and saves the mesh at `filepath` (format
inferred from the extension: `.msh`, `.inp` for Abaqus, ...). If `filepath` ends
with `.geo`, only the geometry file is written (`savegeo` does exactly that).

| Option | Default | Description |
| --- | --- | --- |
| `element_size` | EBSD resolution | Element size |
| `thickness` | `element_size` | Extrusion thickness (3D meshes) |
| `gradient` | 0 | Size gradient: `size = element_size + gradient * d`, `d` = distance to the nearest grain boundary |
| `element_type` | `'Wedge'` | `'Wedge'`, `'Hex'`, `'Tet'`, `'Tri'`, `'Quad'`, `'HexOnly'`, `'QuadOnly'` (aliases `'Tetrahedron'`, `'Brick'`, `'Triangular'`, `'Quadrangular'`, ... accepted) |
| `element_order` | 1 | 1 = linear, 2 = quadratic elements |
| `curvature` | 0 | If > 0, size from local curvature (`curvature` nodes per 2π) |
| `medium` | `(0,0,0)` | `(dx, dy[, dz])` embeds the ROI in a surrounding medium (tetrahedra) |
| `medium_element_size` | auto | Element size at the corners of the medium |
| `grain_prefix` | `'Grain_'` | Prefix of the physical group names; empty string for plain numbering |
| `verbosity` | 4 | Gmsh verbosity (0–10) |
| `partition` | 0 | Create this many mesh partitions (parallel computing) |
| `periodic` | `'none'` | `'x'`, `'y'`, `'both'` for periodic boundary conditions |
| `local_size` | None | A `LocalSize` object for per-grain element sizes/gradients |
| `layers` | 1 | Number of element layers through the thickness |
| `grain_boundaries_only` | False | Mesh only the grain boundaries (surfacic mesh) |
| `externalize_gbs` | `''` | Also write a surfacic grain-boundary-only mesh at this path |

`GmshGeo` supports single indexing to select grains: `G[5]` (6th grain,
0-based), `G[[0, 2, 5]]`, `G['Forsterite']` (all grains of a phase).

Other methods: `G.size()`, `G.segment_lengths()`, `G.regularize(epsilon=None)`
(collapse too-short segments), `G.add_symmetry('x'|'y'|'both')`,
`G.plot()`, `G.plot_singular_points()`, `G.export_grain_props(...)`.

## How it works

1. `mtex.calcGrains(ebsd)` reconstructs the grains (`grain2d`, Voronoi-based).
2. `GmshGeo(grains)` splits every grain boundary at its singular points (triple
   junctions, ROI corners, ...) into B-spline segments (`compute_segments`,
   Euler paths of the boundary graph), shared segments being stored once.
3. `mesh()` writes a `.geo` file (OpenCASCADE kernel: `Point`/`BSpline`/`Line
   Loop`/`Plane Surface`, optional `Extrude`, physical groups, size fields) and
   runs Gmsh on it.

## Citation

If you use this toolbox, please cite the original paper:

```bibtex
@article{MTEX2Gmsh,
  doi = {10.21105/joss.02094},
  url = {https://doi.org/10.21105/joss.02094},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {52},
  pages = {2094},
  author = {Dorian Depriester and Régis Kubler},
  title = {MTEX2Gmsh: a tool for generating 2D meshes from EBSD data},
  journal = {Journal of Open Source Software}
}
```

## License

MIT (see [LICENSE](LICENSE)).
