"""Example: generate a conforming mesh from EBSD data with pyMTEX2Gmsh.

Port of the Matlab documentation examples (example.m, meshing.m,
AdvancedMeshing.m). Run with:  python examples/example.py
Results are written to ./output/.
"""

import os

import mtex

from pymtex2gmsh import GmshGeo, LocalSize

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- MTEX part
# Load the 'small' example dataset shipped with MTEX and reconstruct the grains
ebsd = mtex.mtexdata("small")
ebsd = ebsd[ebsd.isIndexed]  # remove unindexed points
grains = mtex.calcGrains(ebsd)
print(f"{len(grains)} grains reconstructed")

# ------------------------------------------------------- meshable geometry
G = GmshGeo(grains)
print(G)

# ------------------------------------------------------------------- meshing
# Default: 3D mesh of linear wedge elements, element size = EBSD resolution
status = G.mesh(os.path.join(OUT, "default.msh"))
print("default (wedge) mesh status:", status)

# 2D triangular mesh with a prescribed element size
G.mesh(os.path.join(OUT, "tri.msh"), element_type="Tri", element_size=100)

# Size gradient: element size = s + k * d, d = distance to the nearest grain boundary
G.mesh(os.path.join(OUT, "sizeGradient.msh"), element_type="Tri", element_size=50, gradient=0.5)

# Curvature-based element size (5 nodes per 2*pi)
G.mesh(os.path.join(OUT, "curvature.msh"), element_type="Tri", element_size=50, curvature=5)

# Hexahedron-dominated 3D mesh, quadratic elements
G.mesh(os.path.join(OUT, "brick-quadratic.msh"), element_type="Hex", element_order=2, element_size=100)

# Periodic mesh along X
G.mesh(os.path.join(OUT, "Periodic-x.msh"), element_type="Tri", element_size=100, periodic="x")

# Embed the ROI in a surrounding medium (tetrahedra there)
roi = G.size().roi
G.mesh(
    os.path.join(OUT, "medium.msh"),
    element_type="Tet",
    element_size=100,
    medium=(roi[0] * 1.4, roi[1] * 1.4, 100),
    medium_element_size=200,
)

# Local element size in specific grains
ls = LocalSize([5, 7], 30, 0.5)  # size 30 at boundaries, slope 0.5, in grains 5 and 7
G.mesh(os.path.join(OUT, "localSize.msh"), element_type="Tri", element_size=100, local_size=ls)

# Dump the Gmsh geometry only (no meshing)
G.savegeo(os.path.join(OUT, "geometry.geo"))

# Export grain properties (ids, phases, orientations) for the FEM code
G.export_grain_props(os.path.join(OUT, "grains.csv"))
G.export_grain_props(os.path.join(OUT, "grains_prisms.csv"), "PRISMS")  # PRISMS-Plasticity format

# Plot the geometry and its singular points
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
G.plot(ax)
G.plot_singular_points(ax)
ax.legend()
fig.savefig(os.path.join(OUT, "geometry.png"), dpi=150)

print("All results written to", OUT)
