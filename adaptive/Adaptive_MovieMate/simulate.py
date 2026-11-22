"""Offline simulation: does adaptation reduce effort compared with a static layout?

SIMULATED users only. Each has two favorite genres and clicks by a simple probabilistic model, so the
results show that the mechanism works as designed, not that real people would benefit. Two checks:
  1. Layout: users scan the genre rows from the top, paying less attention further down. The measure
     is the row position of the favorite-genre movie they open (lower is better) and the open rate.
     "Static" keeps the default row order; "Adaptive" re-ranks rows from the user's clicks.
  2. Toolbar: users repeat their own favorite actions; an action in the visible strip costs 1 click,
     an action in the "More" menu costs 2.

Run:  python simulate.py                 (synthetic catalog, 200 users, 30 rounds)
      python simulate.py --movielens     (uses the MovieLens catalog after server.py downloaded it)
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from adaptive import ACTIONS, AdaptationEngine, DEFAULT_PRIMARY, ToolbarAdapter
from data import DEFAULT_DIR, download_dataset, load_movielens, make_synthetic
from models import Catalog, HybridRecommender

VIEW_CARDS = 5          # cards visible per row without scrolling
P_FAVORITE = 0.5        # chance of opening a visible favorite-genre card (before position decay)
POSITION_DECAY = 0.9    # attention falls with card position within a row
ROW_ATTENTION = 0.9     # chance the user looks at the first genre row
ROW_DECAY = 0.85        # attention falls with row position


def run_user(model, user_seed: int, rounds: int, adaptive: bool):
    """One simulated user over `rounds` visits. Each visit the user scans the genre rows top to
    bottom (the 'Top picks' row is excluded because it mixes all genres). Attention to a row falls
    with its position, and within a row the user opens a favorite-genre card with a fixed chance.
    Returns, per round: 1 if a favorite-genre card was opened, and the row position (1-based) of that
    open (NaN if none). Adaptive users see rows re-ranked by the engine; static users see the default
    order (the engine is frozen but still personalizes the movies inside each row)."""
    rng = np.random.default_rng(user_seed)
    engine = AdaptationEngine(model, seed=user_seed)
    if not adaptive:
        engine.set_frozen(True)
    fav = set(rng.choice(engine.catalog.genres, size=2, replace=False))
    hits, positions = [], []
    for _ in range(rounds):
        genre_rows = engine.rows()[1:]  # skip "Top picks"
        opened = None
        for pos, row in enumerate(genre_rows, start=1):
            if rng.random() >= ROW_ATTENTION * ROW_DECAY ** (pos - 1):
                continue
            for c_idx, it in enumerate(row["items"][:VIEW_CARDS]):
                is_fav = bool(fav & set(it["genres"]))
                if is_fav and rng.random() < P_FAVORITE * POSITION_DECAY ** c_idx:
                    opened = (pos, row["genre"], it, is_fav)
                    break
            if opened:
                break
        if opened is None:
            hits.append(0.0)
            positions.append(np.nan)
            engine.n += 1
            engine._adapt()
            continue
        pos, genre, it, is_fav = opened
        hits.append(1.0)
        positions.append(float(pos))
        engine.event("open", item=it["id"], row=genre)
        if rng.random() < 0.6:
            engine.event("like", item=it["id"], row=genre)
    return np.array(hits, float), np.array(positions, float)


def phase_stats(per_user, lo, hi):
    """Per-user mean over rounds [lo, hi); returns (mean, standard error) across users."""
    vals = np.array([np.nanmean(u[lo:hi]) if np.isfinite(u[lo:hi]).any() else np.nan for u in per_user])
    vals = vals[np.isfinite(vals)]
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))


def simulate_layout(model, n_users: int, rounds: int):
    out = {}
    for name, adaptive in (("Static", False), ("Adaptive", True)):
        hits, rows = zip(*[run_user(model, 10_000 + u, rounds, adaptive) for u in range(n_users)])
        out[name] = (list(hits), list(rows))
    return out


def simulate_toolbar(n_users: int, n_actions: int = 60, seed: int = 5):
    rng = np.random.default_rng(seed)
    static_cost, adaptive_cost = [], []
    for _ in range(n_users):
        probs = rng.dirichlet(np.full(len(ACTIONS), 0.5))
        seq = rng.choice(ACTIONS, size=n_actions, p=probs)
        tb = ToolbarAdapter()
        s = a = 0
        for act in seq:
            s += 1 if act in DEFAULT_PRIMARY else 2
            a += 1 if act in tb.split()[0] else 2
            tb.record(act)
        static_cost.append(s / n_actions)
        adaptive_cost.append(a / n_actions)
    return np.array(static_cost), np.array(adaptive_cost)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--movielens", action="store_true")
    args = ap.parse_args()
    if args.movielens:
        folder = download_dataset(DEFAULT_DIR)
    else:
        folder = make_synthetic(Path(tempfile.mkdtemp()))
    movies, ratings = load_movielens(folder)
    model = HybridRecommender().fit(Catalog(movies), ratings)
    print(f"Catalog: {len(movies)} movies, {len(model.catalog.genres)} genres; "
          f"{args.users} simulated users x {args.rounds} rounds\n")
    res = simulate_layout(model, args.users, args.rounds)
    phases = [(0, 5), (5, 15), (15, args.rounds)]
    print(f"{'Rounds':<10}{'Condition':<10}{'Open rate':>16}{'Row of open (lower=better)':>30}")
    for lo, hi in phases:
        for name in ("Static", "Adaptive"):
            hits, rows = res[name]
            hm, hs = phase_stats(hits, lo, hi)
            rm, rs = phase_stats(rows, lo, hi)
            print(f"{lo + 1}-{hi:<7}{name:<10}{hm:>9.3f} ±{hs:.3f}{rm:>18.2f} ±{rs:.2f}")
    s, a = simulate_toolbar(args.users)
    print(f"\nToolbar clicks per action (60 actions per user): static {s.mean():.2f} ±{s.std(ddof=1)/np.sqrt(len(s)):.2f}, "
          f"adaptive {a.mean():.2f} ±{a.std(ddof=1)/np.sqrt(len(a)):.2f}")
    print("\n(± is one standard error across simulated users. Simulated users, not real ones.)")


if __name__ == "__main__":
    main()
