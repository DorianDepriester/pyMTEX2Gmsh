import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _gmsh_usable() -> bool:
    try:
        import gmsh

        gmsh.initialize()
        gmsh.finalize()
        return True
    except Exception:
        return False


requires_gmsh = pytest.mark.skipif(not _gmsh_usable(), reason="Gmsh Python SDK not usable in this environment")


@pytest.fixture(scope="session")
def grains_synth():
    """A small synthetic single-phase EBSD map (12x12 pixels), no download needed."""
    import mtex

    cs = mtex.loadCIF("Al-Aluminum.cif")
    rng = mtex.rotation.rand(12, 12)
    ebsd = mtex.EBSDsquare(rng, CSList=cs, dxy=(2.0, 2.0))
    return mtex.calcGrains(ebsd)


@pytest.fixture(scope="session")
def grains_small():
    """The 'small' EBSD dataset shipped with MTEX (requires a one-time download)."""
    import mtex

    try:
        ebsd = mtex.mtexdata("small")
    except Exception as exc:
        pytest.skip(f"cannot download the 'small' dataset: {exc}")
    return mtex.calcGrains(ebsd[ebsd.isIndexed])
