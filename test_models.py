import numpy as np

from moviemate.models import ItemCF, minmax, center_value
from moviemate.evaluate import split_ratings, user_profiles


def test_minmax_scales_only_masked_entries():
    x = np.array([1.0, 5.0, 3.0, 100.0])
    mask = np.array([True, True, True, False])
    out = minmax(x, mask)
    assert out[0] == 0.0 and out[1] == 1.0 and out[2] == 0.5 and out[3] == 0.0


def test_center_value_falls_back_to_global_mean_for_small_profiles():
    assert center_value({1: 5.0, 2: 4.0}, 3.5) == 3.5


def test_similarity_matrix_is_nonnegative_with_zero_diagonal(model):
    S = model.cf.S.toarray()
    assert (S >= 0).all()
    assert np.allclose(np.diag(S), 0)


def test_shrinkage_reduces_similarity(data):
    from moviemate.models import Catalog
    movies, ratings = data
    cat = Catalog(movies)
    weak = ItemCF(shrinkage=0.0, k=0).fit(cat, ratings).S.sum()
    strong = ItemCF(shrinkage=50.0, k=0).fit(cat, ratings).S.sum()
    assert strong < weak


def test_recommendations_exclude_rated_items(model):
    profile = {0: 5.0, 1: 4.0, 2: 5.0}
    recs = model.recommend(profile, n=10)
    assert len(recs) == 10 and not set(recs) & set(profile)


def test_cold_start_returns_popular_items(model):
    recs = model.recommend({}, n=5)
    assert len(recs) == 5
    pop = model.pop.scores()
    assert np.mean([pop[i] for i in recs]) > np.mean(pop)


def test_hybrid_recovers_genre_taste(model):
    """A user who loves one genre should be recommended mostly that genre."""
    cat = model.catalog
    genre = "Horror" if "Horror" in cat.genres else cat.genres[0]
    in_genre = [i for i, gl in enumerate(cat.movies["genre_list"]) if genre in gl]
    profile = {i: 5.0 for i in in_genre[:6]}
    recs = model.recommend(profile, n=10)
    share = np.mean([genre in cat.movies.at[i, "genre_list"] for i in recs])
    base = len(in_genre) / len(cat)
    assert share > base + 0.2


def test_genre_and_year_filters(model):
    recs = model.recommend({}, n=5, genre="Comedy", year_min=1990, year_max=1999)
    for i in recs:
        assert "Comedy" in model.catalog.movies.at[i, "genre_list"]
        assert 1990 <= model.catalog.movies.at[i, "year"] <= 1999


def test_split_removes_test_item_from_training(data):
    _, ratings = data
    train, val, test = split_ratings(ratings)
    for uid, mid in list(test.items())[:20]:
        assert not ((train["userId"] == uid) & (train["movieId"] == mid)).any()
