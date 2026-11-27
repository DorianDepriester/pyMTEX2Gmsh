"""Run the Gmsh meshing through the official Gmsh Python SDK (``pip install gmsh``).

This replaces the shell call to the ``gmsh`` binary of the original Matlab
toolbox: the generated ``.geo`` file is parsed and executed in-process
(``gmsh.merge``), which also runs the ``Mesh 2`` / ``Mesh 3`` commands it
contains, then the mesh is written to the requested file.
"""

from __future__ import annotations


class GmshRunError(RuntimeError):
    """Raised when the Gmsh SDK fails to mesh a geometry."""


def run_gmsh(geo_path: str, out_path: str, verbosity: int = 4) -> int:
    """Execute ``geo_path`` with Gmsh and write the mesh to ``out_path``.

    Returns 0 on success; raises :class:`GmshRunError` otherwise.
    """
    try:
        import gmsh
    except ImportError as exc:
        raise GmshRunError(
            "The Gmsh Python SDK is required to run the meshing. Install it with 'pip install gmsh' "
            "(or 'pip install pymtex2gmsh[mesh]')."
        ) from exc
    initialized = False
    try:
        gmsh.initialize()
        initialized = True
        gmsh.option.setNumber("General.Verbosity", int(verbosity))
        gmsh.merge(geo_path)
        gmsh.write(out_path)
        return 0
    except Exception as exc:  # noqa: BLE001 - report any Gmsh failure uniformly
        raise GmshRunError(f"Gmsh failed to mesh {geo_path!r}: {exc}") from exc
    finally:
        if initialized:
            gmsh.finalize()
