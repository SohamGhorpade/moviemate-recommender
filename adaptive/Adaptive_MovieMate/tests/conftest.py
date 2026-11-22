import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_movielens, make_synthetic  # noqa: E402
from models import Catalog, HybridRecommender  # noqa: E402


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    return load_movielens(make_synthetic(tmp_path_factory.mktemp("synthetic")))


@pytest.fixture(scope="session")
def model(data):
    movies, ratings = data
    return HybridRecommender().fit(Catalog(movies), ratings)
