"""pyMTEX2Gmsh: generate conforming meshes from EBSD data processed with the Python port of MTEX.

Python port of the Matlab toolbox MTEX2Gmsh (Depriester & Kubler, JOSS 2020,
https://doi.org/10.21105/joss.02094), based on the ``mtex`` Python package and
the Gmsh Python SDK.

Example::

    import mtex
    from pymtex2gmsh import GmshGeo, LocalSize

    ebsd = mtex.mtexdata('small')
    grains = mtex.calcGrains(ebsd[ebsd.isIndexed])
    G = GmshGeo(grains)
    G.mesh('small.msh')                      # 3D wedge mesh, size = EBSD resolution
    G.mesh('small_tri.msh', element_type='Tri')   # 2D triangular mesh
    G.savegeo('geometry.geo')                # dump the Gmsh geometry
    G.export_grain_props('grains.csv')       # grain ids, phases, orientations
"""

from .bspline import bspline, bspline_auto
from .gmsh_geo import GeoSize, GmshGeo
from .graph import euler_path
from .local_size import LocalSize, LocalSizeEntry
from .model import Grain
from .segments import compute_segments
from .singular_points import SingularPoints, find_singular_points
from .mtex_compat import grain_num_pixel, grain_vertices, vtx_on_bounds

__version__ = "0.1.0"

__all__ = [
    "GmshGeo",
    "GeoSize",
    "Grain",
    "LocalSize",
    "LocalSizeEntry",
    "SingularPoints",
    "bspline",
    "bspline_auto",
    "compute_segments",
    "euler_path",
    "find_singular_points",
    "grain_num_pixel",
    "grain_vertices",
    "vtx_on_bounds",
    "__version__",
]
