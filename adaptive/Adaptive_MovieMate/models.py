"""Recommendation models: popularity, item-based CF, content-based, and a weighted hybrid.

Reused from the author's earlier MovieMate project (Ghorpade, 2026) to power the adaptive
interface in this repository (model code unchanged); the adaptation logic is in adaptive.py.

Foundation: the public item-to-item collaborative-filtering notebook by vishwajeet-hogale (n.d.),
which builds a user x movie matrix with missing ratings filled as 0, computes Pearson correlations
between movie columns, and scores candidates by summing similarity x the user's rating
(item-based CF idea: Sarwar et al., 2001). That notebook has no evaluation. Modifications here:
  1. Similarity is adjusted cosine on user-mean-centered ratings over OBSERVED ratings only;
     missing ratings are not treated as zeros.
  2. Shrinkage of similarities by the number of co-raters (damps noisy pairs).
  3. A minimum-ratings filter and top-k neighbors per item, stored as a sparse matrix.
  4. A genre-based content model and a Bayesian popularity prior.
  5. A weighted hybrid (Burke, 2002) with per-user min-max normalization and cold-start fallback.
  6. Explanations, a chat interface, and an offline evaluation protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class Catalog:
    """Maps MovieLens movieIds to dense item indices and holds item metadata."""

    movies: pd.DataFrame
    ids: np.ndarray = field(init=False)
    index: dict = field(init=False)
    genres: list = field(init=False)
    genre_matrix: np.ndarray = field(init=False)

    def __post_init__(self):
        self.movies = self.movies.reset_index(drop=True)
        self.ids = self.movies["movieId"].to_numpy()
        self.index = {mid: i for i, mid in enumerate(self.ids)}
        self.genres = sorted({g for gl in self.movies["genre_list"] for g in gl})
        gi = {g: j for j, g in enumerate(self.genres)}
        G = np.zeros((len(self.movies), len(self.genres)), dtype=np.float32)
        for i, gl in enumerate(self.movies["genre_list"]):
            for g in gl:
                G[i, gi[g]] = 1.0
        self.genre_matrix = G

    def __len__(self):
        return len(self.movies)

    def title(self, i: int) -> str:
        return str(self.movies.at[i, "title"])


def center_value(profile: dict, global_mean: float) -> float:
    """Mean used to center a profile: the user's own mean if reliable, else the global mean."""
    vals = np.array(list(profile.values()), dtype=float)
    if len(vals) >= 5 and vals.std() > 0:
        return float(vals.mean())
    return global_mean


def minmax(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scale x to [0, 1] over the entries where mask is True; others become 0."""
    out = np.zeros_like(x, dtype=float)
    if mask.sum() == 0:
        return out
    v = x[mask].astype(float)
    lo, hi = v.min(), v.max()
    out[mask] = 0.0 if hi - lo < 1e-12 else (v - lo) / (hi - lo)
    return out


class PopularityModel:
    """Bayesian-average rating: shrinks items with few ratings toward the global mean."""

    def __init__(self, prior_count: float = 10.0):
        self.prior_count = prior_count

    def fit(self, catalog: Catalog, ratings: pd.DataFrame):
        self.catalog = catalog
        idx = ratings["movieId"].map(catalog.index).dropna().astype(int)
        r = ratings.loc[idx.index, "rating"].to_numpy()
        n = len(catalog)
        sums = np.bincount(idx.to_numpy(), weights=r, minlength=n)
        counts = np.bincount(idx.to_numpy(), minlength=n).astype(float)
        self.global_mean = float(r.mean())
        self.counts = counts
        self.score_ = (self.prior_count * self.global_mean + sums) / (self.prior_count + counts)
        return self

    def scores(self) -> np.ndarray:
        return self.score_


class ItemCF:
    """Item-based collaborative filtering with adjusted cosine and shrinkage."""

    def __init__(self, min_ratings: int = 5, shrinkage: float = 10.0, k: int = 40):
        self.min_ratings = min_ratings
        self.shrinkage = shrinkage
        self.k = k

    def fit(self, catalog: Catalog, ratings: pd.DataFrame):
        self.catalog = catalog
        n_items = len(catalog)
        df = ratings[ratings["movieId"].isin(catalog.ids)].copy()
        df["i"] = df["movieId"].map(catalog.index).astype(int)
        users = df["userId"].astype("category")
        df["u"] = users.cat.codes
        self.global_mean = float(df["rating"].mean())
        user_means = df.groupby("u")["rating"].transform("mean")
        df["c"] = df["rating"] - user_means
        counts = np.bincount(df["i"], minlength=n_items)
        self.counts = counts
        self.eligible = counts >= self.min_ratings
        df = df[self.eligible[df["i"].to_numpy()]]
        # Work only on eligible items: a dense E x E matrix instead of N x N keeps memory small.
        elig_idx = np.flatnonzero(self.eligible)
        pos = np.full(n_items, -1, dtype=np.int64)
        pos[elig_idx] = np.arange(len(elig_idx))
        cols = pos[df["i"].to_numpy()]
        n_users = int(users.cat.codes.max()) + 1
        E = len(elig_idx)
        X = sparse.csr_matrix((df["c"].to_numpy(), (df["u"], cols)), shape=(n_users, E))
        B = sparse.csr_matrix((np.ones(len(df)), (df["u"], cols)), shape=(n_users, E))
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        S = (X.T @ X).toarray() / np.outer(norms, norms)
        co = (B.T @ B).toarray()
        denom = co + self.shrinkage
        weight = np.divide(co, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
        S = S * weight  # modification 2: shrinkage by co-rater count
        np.fill_diagonal(S, 0.0)
        # modification 3: keep only the k strongest positive neighbors per item
        S = np.where(S > 0, S, 0.0)
        if self.k and self.k < E:
            kth = np.partition(S, -self.k, axis=1)[:, -self.k][:, None]
            S = np.where(S >= kth, S, 0.0)
        S = sparse.csr_matrix(S.astype(np.float32))
        # Expand back to catalog indices (sparse, so cost is proportional to nonzeros).
        P = sparse.csr_matrix((np.ones(E), (elig_idx, np.arange(E))), shape=(n_items, E))
        self.S = (P @ S @ P.T).tocsr()
        return self

    def _vectors(self, profile: dict):
        n = len(self.catalog)
        c = np.zeros(n, dtype=np.float32)
        rated = np.zeros(n, dtype=np.float32)
        mean = center_value(profile, self.global_mean)
        for idx, r in profile.items():
            c[idx] = r - mean
            rated[idx] = 1.0
        return c, rated

    def scores(self, profile: dict) -> np.ndarray:
        c, rated = self._vectors(profile)
        num = self.S @ c
        den = abs(self.S) @ rated + 1.0  # +1 keeps weakly supported items modest
        return num / den

    def explain(self, profile: dict, item: int, top: int = 2) -> list[tuple[int, float]]:
        """Liked items in the profile that contributed most to `item`'s score."""
        c, _ = self._vectors(profile)
        contrib = self.S[item].toarray().ravel() * c
        order = np.argsort(-contrib)[:top]
        return [(int(j), float(contrib[j])) for j in order if contrib[j] > 0]

    def similar_items(self, item: int, top: int = 10) -> list[tuple[int, float]]:
        row = self.S[item].toarray().ravel()
        order = np.argsort(-row)[:top]
        return [(int(j), float(row[j])) for j in order if row[j] > 0]


class ContentModel:
    """Genre-profile model: cosine between item genre vectors and the user's liked-genre profile."""

    def fit(self, catalog: Catalog, global_mean: float = 3.5):
        self.catalog = catalog
        self.global_mean = global_mean
        G = catalog.genre_matrix
        norms = np.linalg.norm(G, axis=1)
        norms[norms == 0] = 1.0
        self.G = G / norms[:, None]
        return self

    def scores(self, profile: dict) -> np.ndarray:
        mean = center_value(profile, self.global_mean)
        w = np.zeros(self.G.shape[1], dtype=np.float32)
        for idx, r in profile.items():
            w += (r - mean) * self.G[idx]
        n = np.linalg.norm(w)
        if n == 0:
            return np.zeros(len(self.catalog), dtype=np.float32)
        return self.G @ (w / n)


@dataclass
class HybridWeights:
    cf: float = 0.6
    content: float = 0.25
    popularity: float = 0.15


class HybridRecommender:
    """Weighted hybrid (Burke, 2002) of ItemCF, ContentModel, and PopularityModel."""

    def __init__(self, weights: HybridWeights | None = None, **cf_kwargs):
        self.weights = weights or HybridWeights()
        self.cf = ItemCF(**cf_kwargs)
        self.content = ContentModel()
        self.pop = PopularityModel()

    def fit(self, catalog: Catalog, ratings: pd.DataFrame):
        self.catalog = catalog
        self.cf.fit(catalog, ratings)
        self.pop.fit(catalog, ratings)
        self.content.fit(catalog, self.pop.global_mean)
        return self

    def component_scores(self, profile: dict, mask: np.ndarray):
        return (
            minmax(self.cf.scores(profile), mask),
            minmax(self.content.scores(profile), mask),
            minmax(self.pop.scores(), mask),
        )

    def candidate_mask(self, profile: dict, genre: str | None = None,
                       year_min: float | None = None, year_max: float | None = None) -> np.ndarray:
        mask = self.cf.eligible.copy()
        for idx in profile:
            mask[idx] = False
        movies = self.catalog.movies
        if genre:
            mask &= movies["genre_list"].apply(lambda gl: genre in gl).to_numpy()
        if year_min is not None:
            mask &= (movies["year"] >= year_min).to_numpy()
        if year_max is not None:
            mask &= (movies["year"] <= year_max).to_numpy()
        return mask

    def scores(self, profile: dict, mask: np.ndarray | None = None) -> np.ndarray:
        if mask is None:
            mask = self.candidate_mask(profile)
        cf, ct, pp = self.component_scores(profile, mask)
        w = self.weights
        if not profile:  # cold start: only popularity is meaningful
            total = pp
        else:
            total = w.cf * cf + w.content * ct + w.popularity * pp
        return np.where(mask, total, -np.inf)

    def recommend(self, profile: dict, n: int = 10, **filters) -> list[int]:
        mask = self.candidate_mask(profile, **filters)
        s = self.scores(profile, mask)
        order = np.argsort(-s)[:n]
        return [int(i) for i in order if np.isfinite(s[i])]
