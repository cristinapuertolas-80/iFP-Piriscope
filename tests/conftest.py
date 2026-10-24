import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


@pytest.fixture
def bronze(tmp_path):
    """Carpeta Bronze temporal para no tocar data/ del repositorio."""
    d = tmp_path / "bronze"
    d.mkdir()
    return d


#ML
from tests.toy_data import make_toy_gold


@pytest.fixture(scope="session")
def toy_gold():
    return make_toy_gold()
