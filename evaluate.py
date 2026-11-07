"""Offline evaluation: leave-last-liked-out ranking, with a validation step for weight tuning.

For each user with at least `min_liked` liked movies (rating >= 4), the most recent liked
movie is the TEST item and the second most recent is the VALIDATION item. Both are removed
from the training ratings. Models rank every movie the user has not rated in training; we
report HitRate@K, NDCG@K, MRR, catalog coverage, and mean popularity of the recommendations.
Hybrid weights are chosen on the validation item, then reported on the test item.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .data import download_dataset, load_movielens, make_synthetic
from .models import Catalog, HybridRecommender, HybridWeights, PopularityModel, minmax


def split_ratings(ratings: pd.DataFrame, liked: float = 4.0, min_liked: int = 3):
    """Return (train, validation_items, test_items) where items are {userId: movieId}."""
    ratings = ratings.sort_values(["userId", "timestamp"])
    val, test, drop_rows = {}, {}, []
    for uid, g in ratings.groupby("userId"):
        lk = g[g["rating"] >= liked]
        if len(lk) < min_liked:
            continue
        test_row, val_row = lk.index[-1], lk.index[-2]
        test[uid], val[uid] = int(lk.at[test_row, "movieId"]), int(lk.at[val_row, "movieId"])
        drop_rows += [test_row, val_row]
    train = ratings.drop(index=drop_rows)
    return train, val, test


def user_profiles(train: pd.DataFrame, catalog: Catalog) -> dict:
    profiles = {}
    for uid, g in train.groupby("userId"):
        profiles[uid] = {
            catalog.index[m]: float(r) for m, r in zip(g["movieId"], g["rating"]) if m in catalog.index
        }
    return profiles


def rank_of(scores: np.ndarray, target: int) -> int:
    """1-based rank of `target` among finite scores (ties resolved pessimistically)."""
    t = scores[target]
    if not np.isfinite(t):
        return len(scores) + 1
    return int((scores > t).sum()) + 1


def evaluate(score_fn, profiles: dict, targets: dict, catalog: Catalog, pop_counts: np.ndarray,
             eligible: np.ndarray, k: int = 10) -> dict:
    hits, ndcgs, rrs, recommended, pops = 0, [], [], set(), []
    n = 0
    for uid, movie_id in targets.items():
        prof = profiles[uid]
        # Hide the target from the user's profile mask, but keep it as a candidate.
        target = catalog.index[movie_id]
        if not eligible[target]:
            continue
        s = score_fn(prof, target)
        r = rank_of(s, target)
        n += 1
        if r <= k:
            hits += 1
            ndcgs.append(1.0 / math.log2(r + 1))
        else:
            ndcgs.append(0.0)
        rrs.append(1.0 / r)
        top = np.argsort(-s)[:k]
        recommended.update(int(i) for i in top)
        pops.extend(pop_counts[top])
    return {
        "users": n,
        f"HitRate@{k}": hits / n,
        f"NDCG@{k}": float(np.mean(ndcgs)),
        "MRR": float(np.mean(rrs)),
        "Coverage": len(recommended) / int(eligible.sum()),
        "MeanPopularity": float(np.mean(pops)),
    }


def run(folder: Path, k: int = 10, seed: int = 0) -> dict:
    movies, ratings = load_movielens(folder)
    catalog = Catalog(movies)
    train, val, test = split_ratings(ratings)
    profiles = user_profiles(train, catalog)

    hybrid = HybridRecommender().fit(catalog, train)
    eligible = hybrid.cf.eligible
    pop = hybrid.pop
    pop_counts = pop.counts

    def candidates(prof, target):
        m = eligible.copy()
        for i in prof:
            m[i] = False
        m[target] = True
        return m

    def score_pop(prof, target):
        m = candidates(prof, target)
        return np.where(m, pop.scores(), -np.inf)

    def score_cf(prof, target):
        m = candidates(prof, target)
        return np.where(m, minmax(hybrid.cf.scores(prof), m), -np.inf)

    def score_content(prof, target):
        m = candidates(prof, target)
        return np.where(m, minmax(hybrid.content.scores(prof), m), -np.inf)

    def make_hybrid(w):
        def f(prof, target):
            m = candidates(prof, target)
            cf, ct, pp = hybrid.component_scores(prof, m)
            return np.where(m, w.cf * cf + w.content * ct + w.popularity * pp, -np.inf)
        return f

    # Tune hybrid weights on the validation item (profiles contain neither held-out item).
    grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    best, best_ndcg, tuning = HybridWeights(), -1.0, []
    for a, b in itertools.product(grid, grid):
        c = round(1.0 - a - b, 2)
        if c < -1e-9:
            continue
        w = HybridWeights(cf=a, content=b, popularity=max(c, 0.0))
        res = evaluate(make_hybrid(w), profiles, val, catalog, pop_counts, eligible, k)
        tuning.append({"cf": a, "content": b, "popularity": max(c, 0.0), f"NDCG@{k}": res[f"NDCG@{k}"]})
        if res[f"NDCG@{k}"] > best_ndcg:
            best, best_ndcg = w, res[f"NDCG@{k}"]

    # Test: the same profiles (validation item removed), test item as target.
    results = {
        "Popularity": evaluate(score_pop, profiles, test, catalog, pop_counts, eligible, k),
        "Item CF": evaluate(score_cf, profiles, test, catalog, pop_counts, eligible, k),
        "Content (genre)": evaluate(score_content, profiles, test, catalog, pop_counts, eligible, k),
        "Hybrid (default 0.6/0.25/0.15)": evaluate(
            make_hybrid(HybridWeights()), profiles, test, catalog, pop_counts, eligible, k),
        f"Hybrid (tuned {best.cf}/{best.content}/{best.popularity})": evaluate(
            make_hybrid(best), profiles, test, catalog, pop_counts, eligible, k),
    }
    return {"results": results, "best_weights": best.__dict__, "tuning": tuning,
            "n_movies": len(catalog), "n_eligible": int(eligible.sum()),
            "n_ratings": int(len(ratings)), "n_users": int(ratings["userId"].nunique())}


def format_table(results: dict, k: int = 10) -> str:
    cols = ["users", f"HitRate@{k}", f"NDCG@{k}", "MRR", "Coverage", "MeanPopularity"]
    lines = ["Model".ljust(40) + "".join(c.rjust(16) for c in cols)]
    for name, r in results.items():
        row = name.ljust(40)
        for c in cols:
            v = r[c]
            row += (f"{v:>16d}" if isinstance(v, int) else f"{v:>16.3f}")
        lines.append(row)
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Offline evaluation of MovieMate models.")
    p.add_argument("--data", type=Path, default=None, help="Folder containing movies.csv and ratings.csv")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data (offline testing only)")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--out", type=Path, default=Path("results.json"))
    a = p.parse_args(argv)
    if a.synthetic:
        folder = make_synthetic(Path("data") / "synthetic")
        print("NOTE: synthetic data -- numbers say nothing about real accuracy.")
    elif a.data:
        folder = a.data
    else:
        folder = download_dataset()
    out = run(folder, a.k)
    print(f"\nUsers: {out['n_users']}  Movies: {out['n_movies']} (eligible {out['n_eligible']})  "
          f"Ratings: {out['n_ratings']}\n")
    print(format_table(out["results"], a.k))
    print("\nBest hybrid weights on validation:", out["best_weights"])
    a.out.write_text(json.dumps(out, indent=2))
    print(f"\nSaved details to {a.out}")


if __name__ == "__main__":
    main()
