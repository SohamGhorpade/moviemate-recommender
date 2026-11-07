import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moviemate.chat import ChatAgent
from moviemate.data import load_movielens, make_synthetic
from moviemate.models import Catalog, HybridRecommender


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    folder = make_synthetic(tmp_path_factory.mktemp("synthetic"))
    movies, ratings = load_movielens(folder)
    return movies, ratings


@pytest.fixture(scope="session")
def model(data):
    movies, ratings = data
    return HybridRecommender().fit(Catalog(movies), ratings)


@pytest.fixture()
def agent(model, data):
    return ChatAgent(model, data[1])
