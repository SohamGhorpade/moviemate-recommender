"""Loading, downloading, and (for offline testing) synthesizing MovieLens-format data.

MovieLens data is NOT included in this repository. The GroupLens license does not
allow redistribution, so `download_dataset` fetches it from GroupLens directly.
"""
from __future__ import annotations

import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DEFAULT_DIR = Path(__file__).resolve().parent / "data"
DATASET_NAME = "ml-latest-small"


def download_dataset(dest: Path = DEFAULT_DIR) -> Path:
    """Download and unzip ml-latest-small into `dest`. Returns the dataset folder."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / DATASET_NAME
    if (target / "ratings.csv").exists():
        return target
    zip_path = dest / f"{DATASET_NAME}.zip"
    print(f"Downloading {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    zip_path.unlink()
    return target


def load_movielens(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (movies, ratings) DataFrames from a MovieLens-format folder."""
    folder = Path(folder)
    movies = pd.read_csv(folder / "movies.csv")
    ratings = pd.read_csv(folder / "ratings.csv")
    movies = movies.copy()
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    movies["genre_list"] = movies["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    return movies, ratings


GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Children", "Thriller"]


def make_synthetic(folder: Path, n_users: int = 300, n_movies: int = 400, seed: int = 7) -> Path:
    """Write a small synthetic dataset in MovieLens format (for tests and offline demos).

    Users have latent genre tastes, so a working recommender should recover them.
    Results on this data say nothing about real-world accuracy.
    """
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    movie_genres = []
    for _ in range(n_movies):
        k = rng.integers(1, 3)
        movie_genres.append(list(rng.choice(GENRES, size=k, replace=False)))
    years = rng.integers(1970, 2020, size=n_movies)
    movies = pd.DataFrame(
        {
            "movieId": np.arange(1, n_movies + 1),
            "title": [f"Movie {i} ({y})" for i, y in zip(range(1, n_movies + 1), years)],
            "genres": ["|".join(g) for g in movie_genres],
        }
    )
    popularity = rng.pareto(1.5, size=n_movies) + 1
    popularity /= popularity.sum()
    rows = []
    for u in range(1, n_users + 1):
        taste = {g: rng.uniform(-1, 1) for g in GENRES}
        n_r = int(rng.integers(20, 80))
        picks = rng.choice(n_movies, size=n_r, replace=False, p=popularity)
        for t, m in enumerate(picks):
            affinity = np.mean([taste[g] for g in movie_genres[m]])
            r = np.clip(round(3.3 + 1.6 * affinity + rng.normal(0, 0.5)), 1, 5)
            rows.append((u, m + 1, float(r), 1_000_000 + u * 1000 + t))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    movies.to_csv(folder / "movies.csv", index=False)
    ratings.to_csv(folder / "ratings.csv", index=False)
    return folder
