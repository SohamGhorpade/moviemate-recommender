"""Runtime adaptation logic for the adaptive MovieMate interface.

Three things adapt while the user interacts:
  1. The ORDER OF GENRE ROWS, using Thompson sampling (a multi-armed bandit) over genres.
  2. The TOOLBAR: the most-used actions are promoted to the visible strip (usage frequency with decay).
  3. The DEFAULT EXPLANATION MODE: repeated "Why?" requests switch explanations to inline display.
Within each row, movies are ranked by the hybrid recommender from models.py using the user's profile.

Design rules taken from the human-AI interaction literature (Amershi et al., 2019): learn from user
behavior, adapt cautiously (a minimum gap between layout changes), explain what changed, and give the
user global control (freeze and reset). All state lives in memory only; nothing is written to disk.

The bandit follows the Beta-Bernoulli Thompson sampling described by Russo et al. (2018); the code
is written from scratch for this project.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

ACTIONS = ["search", "decade", "surprise", "explain", "likes"]
ACTION_LABELS = {
    "search": "Search",
    "decade": "Decade",
    "surprise": "Surprise me",
    "explain": "Explain all",
    "likes": "My likes",
}
DEFAULT_PRIMARY = ["search", "decade", "surprise"]


class GenreBandit:
    """Beta-Bernoulli Thompson sampling with one arm per genre.

    Each arm keeps Beta(alpha, beta). A reward of 1 means the user showed interest in that genre;
    a reward of 0 means a displayed genre was ignored or rejected. Old evidence fades through a
    decay toward the prior, so interests that change are followed.
    """

    def __init__(self, genres, prior_alpha: float = 1.0, prior_beta: float = 4.0,
                 decay: float = 0.97, seed: int = 0):
        self.genres = list(genres)  # also the tie-break order for arms with equal scores
        self.pa, self.pb = prior_alpha, prior_beta
        self.decay = decay
        self.seed = seed
        self.alpha = {g: prior_alpha for g in self.genres}
        self.beta = {g: prior_beta for g in self.genres}
        self.evidence = Counter()

    def update(self, genre: str, reward: float, weight: float = 1.0) -> None:
        if genre not in self.alpha:
            return
        self.alpha[genre] = self.pa + self.decay * (self.alpha[genre] - self.pa) + weight * reward
        self.beta[genre] = self.pb + self.decay * (self.beta[genre] - self.pb) + weight * (1 - reward)
        if reward > 0:
            self.evidence[genre] += 1

    def mean(self, genre: str) -> float:
        a, b = self.alpha[genre], self.beta[genre]
        return a / (a + b)

    def rank(self, version: int) -> list[str]:
        """Genres ordered by score. Arms with positive evidence use one posterior sample each
        (Thompson sampling, which keeps some exploration); arms without it use their posterior mean,
        so untouched genres keep their default order instead of reshuffling by chance. The seed
        depends on `version`, so one layout version always gives the same order."""
        rng = np.random.default_rng(self.seed + 1000 * version)
        score = {}
        for g in self.genres:
            draw = rng.beta(self.alpha[g], self.beta[g])  # drawn for every arm to keep the stream stable
            score[g] = draw if self.evidence[g] > 0 else self.mean(g)
        default_pos = {g: i for i, g in enumerate(self.genres)}
        return sorted(self.genres, key=lambda g: (-score[g], default_pos[g]))


class ToolbarAdapter:
    """Keeps a decayed usage score per action and shows the top `n_primary` in the visible strip."""

    def __init__(self, n_primary: int = 3, decay: float = 0.9):
        self.n_primary = n_primary
        self.decay = decay
        self.score = {a: 0.0 for a in ACTIONS}

    def record(self, action: str) -> None:
        if action not in self.score:
            return
        for a in self.score:
            self.score[a] *= self.decay
        self.score[action] += 1.0

    def order(self) -> list[str]:
        """Cold start follows DEFAULT_PRIMARY; once there is usage, frequency decides."""
        default_rank = {a: i for i, a in enumerate(DEFAULT_PRIMARY + [a for a in ACTIONS if a not in DEFAULT_PRIMARY])}
        return sorted(ACTIONS, key=lambda a: (-self.score[a], default_rank[a]))

    def split(self) -> tuple[list[str], list[str]]:
        o = self.order()
        return o[: self.n_primary], o[self.n_primary:]


class AdaptationEngine:
    """Holds one user's in-memory state and applies the three adaptations."""

    def __init__(self, model, seed: int = 0, n_rows: int = 5, per_row: int = 8, min_gap: int = 3):
        self.model = model
        self.catalog = model.catalog
        self.seed, self.n_rows, self.per_row, self.min_gap = seed, n_rows, per_row, min_gap
        movies = self.catalog.movies
        counts = self.catalog.genre_matrix.sum(axis=0)
        self.default_order = [g for _, g in sorted(zip(-counts, self.catalog.genres))]
        self._titles = movies["clean_title"].tolist()
        self.reset()

    # ----- lifecycle -------------------------------------------------------------------------
    def reset(self) -> None:
        """Forget everything the interface has learned and return to the static default layout."""
        self.bandit = GenreBandit(self.default_order, seed=self.seed)
        self.toolbar = ToolbarAdapter()
        self.profile: dict[int, float] = {}
        self.n = 0
        self.version = 0
        self.row_order = list(self.default_order)
        self.shown_primary = list(DEFAULT_PRIMARY)
        self.explain_inline = False
        self.frozen = False
        self.recent: list[str] = []
        self.log: list[dict] = []
        self._last_change = {"rows": 0, "toolbar": 0, "explain": 0}

    def set_frozen(self, frozen: bool) -> None:
        self.frozen = bool(frozen)
        self._note("Adaptation frozen: the layout will not change." if self.frozen
                   else "Adaptation resumed.")

    def _note(self, text: str) -> None:
        self.log.append({"n": self.n, "text": text})
        self.log = self.log[-30:]

    # ----- events ----------------------------------------------------------------------------
    def event(self, kind: str, item: int | None = None, row: str | None = None,
              action: str | None = None) -> None:
        """Record one interaction. kind is open, like, dislike, why, or action."""
        self.n += 1
        shown = self.row_order[: self.n_rows]
        if kind in ("open", "like", "dislike") and item is not None:
            genres = list(self.catalog.movies.at[item, "genre_list"])
            if kind == "open":
                self.profile.setdefault(item, 4.0)
                for g in genres:
                    self.bandit.update(g, 1.0, weight=1.0 if g == row else 0.5)
                for g in shown:
                    if g != row and g not in genres:
                        self.bandit.update(g, 0.0, weight=0.1)
            elif kind == "like":
                self.profile[item] = 5.0
                for g in genres:
                    self.bandit.update(g, 1.0, weight=2.0)
            else:
                self.profile[item] = 1.0
                for g in genres:
                    self.bandit.update(g, 0.0, weight=2.0 if g == row else 1.0)
        elif kind == "action" and action:
            self.toolbar.record(action)
        self.recent.append(action if kind == "action" else kind)
        self.recent = self.recent[-20:]
        self._adapt()

    # ----- adaptation rules ------------------------------------------------------------------
    def _can_change(self, part: str) -> bool:
        return (not self.frozen) and (self.n - self._last_change[part] >= self.min_gap)

    def _adapt(self) -> None:
        if self._can_change("rows"):
            self.version += 1
            ranked = self.bandit.rank(self.version)
            if ranked[: self.n_rows] != self.row_order[: self.n_rows]:
                old_top = self.row_order[: self.n_rows]
                moved = [g for g in ranked[: self.n_rows] if g not in old_top]
                promoted = [g for g in ranked[: self.n_rows]
                            if g in old_top and ranked.index(g) < self.row_order.index(g)]
                self.row_order = ranked
                self._last_change["rows"] = self.n
                for g in [x for x in (moved + promoted) if self.bandit.evidence[x] > 0][:2]:
                    self._note(f"Moved the {g} row up: your interest estimate is now "
                               f"{self.bandit.mean(g):.0%} ({self.bandit.evidence[g]} positive signals).")
        if self._can_change("toolbar"):
            primary, _ = self.toolbar.split()
            if set(primary) != set(self.shown_primary):
                added = [a for a in primary if a not in self.shown_primary]
                self.shown_primary = primary
                self._last_change["toolbar"] = self.n
                for a in added:
                    self._note(f"Promoted \"{ACTION_LABELS[a]}\" to the toolbar because you use it often.")
        if self._can_change("explain"):
            whys = self.recent[-12:].count("why") + self.recent[-12:].count("explain")
            if not self.explain_inline and whys >= 3:
                self.explain_inline = True
                self._last_change["explain"] = self.n
                self._note("Showing explanations on every card because you asked \"Why?\" three times.")
            elif self.explain_inline and self.recent[-20:].count("why") + self.recent[-20:].count("explain") == 0:
                self.explain_inline = False
                self._last_change["explain"] = self.n
                self._note("Hid inline explanations: you have not asked for them recently.")

    # ----- views -----------------------------------------------------------------------------
    def _item(self, i: int) -> dict:
        m = self.catalog.movies
        year = m.at[i, "year"]
        rating = self.profile.get(i)
        return {
            "id": int(i),
            "title": self._titles[i],
            "year": None if year != year else int(year),
            "genres": list(m.at[i, "genre_list"]),
            "rated": "like" if rating == 5.0 else "dislike" if rating == 1.0 else None,
        }

    def rows(self, decade: int | None = None) -> list[dict]:
        """Top picks first, then the genre rows in adapted order. A movie appears in at most one
        row, so a card never repeats across rows (the first row it appears in keeps it)."""
        kw = {}
        if decade:
            kw = {"year_min": float(decade), "year_max": float(decade + 9)}
        used: set[int] = set()
        out = []

        def take(ids):
            fresh = [i for i in ids if i not in used][: self.per_row]
            used.update(fresh)
            return [self._item(i) for i in fresh]

        top = self.model.recommend(self.profile, self.per_row + 8, **kw)
        out.append({"genre": "Top picks for you", "adapted": False, "items": take(top)})
        for pos, g in enumerate(self.row_order[: self.n_rows]):
            moved_up = pos < self.default_order.index(g) if self.n else False
            ids = self.model.recommend(self.profile, self.per_row + 8, genre=g, **kw)
            out.append({"genre": g, "adapted": bool(moved_up), "items": take(ids)})
        return out

    def state(self, decade: int | None = None) -> dict:
        primary = list(self.shown_primary)
        more = [a for a in self.toolbar.order() if a not in primary]
        interest = sorted(((g, self.bandit.mean(g)) for g in self.default_order), key=lambda t: -t[1])[:6]
        return {
            "rows": self.rows(decade),
            "toolbar": {"primary": [{"id": a, "label": ACTION_LABELS[a]} for a in primary],
                        "more": [{"id": a, "label": ACTION_LABELS[a]} for a in more]},
            "explain_inline": self.explain_inline,
            "frozen": self.frozen,
            "interest": [{"genre": g, "p": round(p, 3)} for g, p in interest],
            "log": list(reversed(self.log))[:12],
            "interactions": self.n,
            "version": self.version,
        }

    # ----- helpers used by the server --------------------------------------------------------
    def explain(self, item: int) -> str:
        titles = lambda idxs: " and ".join(f"“{self._titles[j]}”" for j in idxs)
        if not self.profile:
            return "Popular with many viewers. Like or open a few movies and this will become personal."
        contrib = self.model.cf.explain(self.profile, item, top=2)
        if contrib:
            return f"Because you showed interest in {titles([j for j, _ in contrib])}."
        mine = Counter(g for i, r in self.profile.items() if r >= 4 for g in self.catalog.movies.at[i, "genre_list"])
        shared = [g for g in self.catalog.movies.at[item, "genre_list"] if g in mine]
        if shared:
            return f"Matches genres you engaged with: {', '.join(shared)}."
        return "Rated highly by many viewers."

    def search(self, q: str, limit: int = 8) -> list[dict]:
        q = q.strip().lower()
        if not q:
            return []
        hits = [i for i, t in enumerate(self._titles) if q in t.lower()][:limit]
        return [self._item(i) for i in hits]

    def surprise(self, seed: int | None = None) -> dict | None:
        """A popular title from a genre outside the visible rows (exploration for the user)."""
        rng = np.random.default_rng(self.n if seed is None else seed)
        hidden = set(self.row_order[self.n_rows:])
        mask = self.model.candidate_mask(self.profile)
        cand = [i for i in np.flatnonzero(mask)
                if hidden & set(self.catalog.movies.at[i, "genre_list"])]
        if not cand:
            cand = list(np.flatnonzero(mask))
        if not cand:
            return None
        weights = np.array([self.model.pop.counts[i] + 1.0 for i in cand])
        return self._item(int(rng.choice(cand, p=weights / weights.sum())))

    def liked(self) -> list[dict]:
        return [self._item(i) for i, r in self.profile.items() if r >= 4]
