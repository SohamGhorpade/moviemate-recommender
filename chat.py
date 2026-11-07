"""A small rule-based chat agent on top of the hybrid recommender.

Examples of what you can type:
    i like Toy Story            i hated Titanic            rate Heat 4
    recommend                   recommend 5 comedy         something scary from the 90s
    more like The Matrix        why 2                      profile
    user 12  (load an existing MovieLens user)             reset      help      quit
"""
from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path

import numpy as np

from .data import download_dataset, load_movielens, make_synthetic
from .models import Catalog, HybridRecommender

MOODS = {
    "funny": "Comedy", "comedy": "Comedy", "laugh": "Comedy",
    "scary": "Horror", "horror": "Horror", "spooky": "Horror",
    "romantic": "Romance", "romance": "Romance", "love story": "Romance",
    "action": "Action", "thriller": "Thriller", "suspense": "Thriller",
    "sci-fi": "Sci-Fi", "scifi": "Sci-Fi", "space": "Sci-Fi",
    "kids": "Children", "family": "Children", "children": "Children",
    "drama": "Drama", "animated": "Animation", "animation": "Animation",
    "documentary": "Documentary", "western": "Western", "musical": "Musical",
    "crime": "Crime", "fantasy": "Fantasy", "adventure": "Adventure", "war": "War",
}
LIKE = r"(?:i\s+)?(?:really\s+)?(?:like|liked|love|loved|enjoyed|enjoy)"
DISLIKE = r"(?:i\s+)?(?:really\s+)?(?:hate|hated|dislike|disliked|didn'?t like)"


class ChatAgent:
    def __init__(self, model: HybridRecommender, ratings=None):
        self.model = model
        self.catalog: Catalog = model.catalog
        self.ratings = ratings
        self.profile: dict[int, float] = {}
        self.last_recs: list[int] = []
        self.pending: list[int] | None = None  # disambiguation choices
        self.pending_rating: float | None = None
        movies = self.catalog.movies
        self._titles = movies["clean_title"].str.lower().tolist()

    # ---- title matching -------------------------------------------------------------
    def find_title(self, query: str, limit: int = 3) -> list[int]:
        q = re.sub(r"\(\d{4}\)", "", query).strip().lower().strip("\"'.!?")
        if not q:
            return []
        exact = [i for i, t in enumerate(self._titles) if t == q]
        if exact:
            return exact[:limit]
        # MovieLens writes "Matrix, The"; try the article-last form too.
        for art in ("the", "a", "an"):
            if q.startswith(art + " "):
                alt = f"{q[len(art) + 1:]}, {art}"
                hit = [i for i, t in enumerate(self._titles) if t == alt]
                if hit:
                    return hit[:limit]
        sub = [i for i, t in enumerate(self._titles) if q in t]
        if sub:
            pop = self.model.pop.counts
            return sorted(sub, key=lambda i: -pop[i])[:limit]
        close = difflib.get_close_matches(q, self._titles, n=limit, cutoff=0.7)
        return [self._titles.index(c) for c in close]

    def _label(self, i: int) -> str:
        return self.catalog.title(i)

    # ---- filters ----------------------------------------------------------------------
    @staticmethod
    def parse_filters(text: str) -> dict:
        t = text.lower()
        f: dict = {}
        for word, genre in MOODS.items():
            if word in t:
                f["genre"] = genre
                break
        m = re.search(r"(?:from the |in the )?(\d{2})s\b", t)
        if m:
            d = int(m.group(1))
            start = 1900 + d if d >= 20 else 2000 + d
            f["year_min"], f["year_max"] = start, start + 9
        m = re.search(r"after (\d{4})", t)
        if m:
            f["year_min"] = int(m.group(1))
        m = re.search(r"before (\d{4})", t)
        if m:
            f["year_max"] = int(m.group(1))
        return f

    # ---- main entry --------------------------------------------------------------------
    def handle(self, text: str) -> str:
        raw = text.strip()
        t = raw.lower()
        if not t:
            return "Say something like 'I like Toy Story' or 'recommend a comedy'."
        if self.pending is not None:
            return self._resolve_pending(t)
        if t in {"help", "?"}:
            return __doc__.strip()
        if t in {"quit", "exit", "bye"}:
            return "Goodbye! Enjoy your movies."
        if t == "reset":
            self.profile.clear(); self.last_recs = []
            return "Cleared your profile. Tell me a few movies you like to get started."
        if t == "profile":
            return self._profile_text()
        m = re.fullmatch(r"user\s+(\d+)", t)
        if m:
            return self._load_user(int(m.group(1)))
        m = re.match(r"why\s+(.+)", t)
        if m:
            return self._why(m.group(1))
        m = re.match(r"(?:more like|similar to|movies like)\s+(.+)", t)
        if m:
            return self._more_like(m.group(1))
        m = re.match(r"rate\s+(.+?)\s+([1-5](?:\.5)?)$", t)
        if m:
            return self._rate(m.group(1), float(m.group(2)))
        m = re.match(rf"{DISLIKE}\s+(.+)", t)
        if m:
            return self._rate(m.group(1), 1.0)
        m = re.match(rf"{LIKE}\s+(.+)", t)
        if m:
            return self._rate(m.group(1), 5.0)
        if re.search(r"recommend|suggest|what should i watch|something|watch", t):
            return self._recommend(t)
        return "I didn't catch that. Type 'help' to see what I can do."

    # ---- actions --------------------------------------------------------------------------
    def _rate(self, title: str, rating: float) -> str:
        hits = self.find_title(title)
        if not hits:
            return f"I couldn't find a movie like '{title}'. Try a different spelling."
        if len(hits) > 1:
            self.pending, self.pending_rating = hits, rating
            opts = "\n".join(f"  {n}. {self._label(i)}" for n, i in enumerate(hits, 1))
            return f"Which one did you mean? Reply with a number.\n{opts}"
        return self._store(hits[0], rating)

    def _store(self, i: int, rating: float) -> str:
        self.profile[i] = rating
        word = "liked" if rating >= 4 else "disliked" if rating <= 2 else f"rated {rating:g}"
        return (f"Noted: you {word} {self._label(i)}. "
                f"({len(self.profile)} movie{'s' if len(self.profile) != 1 else ''} in your profile.) "
                "Say 'recommend' when you're ready.")

    def _resolve_pending(self, t: str) -> str:
        choices, rating = self.pending, self.pending_rating
        self.pending, self.pending_rating = None, None
        if t.isdigit() and 1 <= int(t) <= len(choices):
            return self._store(choices[int(t) - 1], rating)
        return "Okay, I'll skip that one."

    def _recommend(self, t: str) -> str:
        n = 5
        m = re.search(r"\b(\d{1,2})\b(?!\s*s\b)", t)
        if m and 1 <= int(m.group(1)) <= 20 and not re.search(r"after|before", t):
            n = int(m.group(1))
        filters = self.parse_filters(t)
        recs = self.model.recommend(self.profile, n=n, **filters)
        if not recs:
            return "I couldn't find anything matching that. Try fewer filters."
        self.last_recs = recs
        lines = []
        cold = not self.profile
        if cold:
            lines.append("I don't know your taste yet, so these are widely loved picks. "
                         "Tell me what you like to personalize them.")
        for k, i in enumerate(recs, 1):
            lines.append(f"  {k}. {self._label(i)}  [{self.catalog.movies.at[i, 'genres']}]")
        lines.append("Ask 'why 1' for an explanation, or 'more like <title>'.")
        return "\n".join(lines)

    def _why(self, arg: str) -> str:
        if not self.last_recs:
            return "Ask me for recommendations first."
        if arg.isdigit() and 1 <= int(arg) <= len(self.last_recs):
            item = self.last_recs[int(arg) - 1]
        else:
            hits = [i for i in self.find_title(arg) if i in self.last_recs]
            if not hits:
                return "I can only explain items from my latest list. Try 'why 1'."
            item = hits[0]
        reasons = self.model.cf.explain(self.profile, item)
        genres = self.catalog.movies.at[item, "genre_list"]
        liked_genres = set()
        for j, r in self.profile.items():
            if r >= 4:
                liked_genres.update(self.catalog.movies.at[j, "genre_list"])
        shared = [g for g in genres if g in liked_genres]
        parts = []
        if reasons:
            parts.append("people who liked " + " and ".join(self._label(j) for j, _ in reasons)
                         + " also tended to like it")
        if shared:
            parts.append("it matches genres you enjoy (" + ", ".join(shared) + ")")
        if not parts:
            parts.append("it is highly rated overall")
        return f"I suggested {self._label(item)} because " + " and ".join(parts) + "."

    def _more_like(self, title: str) -> str:
        hits = self.find_title(title)
        if not hits:
            return f"I couldn't find '{title}'."
        sims = self.model.cf.similar_items(hits[0], 5)
        if not sims:
            return f"I don't have enough ratings to find movies like {self._label(hits[0])}."
        body = "\n".join(f"  {n}. {self._label(j)}" for n, (j, _) in enumerate(sims, 1))
        return f"Movies similar to {self._label(hits[0])}:\n{body}"

    def _profile_text(self) -> str:
        if not self.profile:
            return "Your profile is empty."
        return "\n".join(f"  {self._label(i)}: {r:g}" for i, r in self.profile.items())

    def _load_user(self, uid: int) -> str:
        if self.ratings is None:
            return "No ratings data loaded."
        g = self.ratings[self.ratings["userId"] == uid]
        if g.empty:
            return f"No MovieLens user {uid}."
        self.profile = {self.catalog.index[m]: float(r) for m, r in zip(g["movieId"], g["rating"])
                        if m in self.catalog.index}
        return f"Loaded MovieLens user {uid} with {len(self.profile)} ratings."


def build_agent(folder: Path) -> ChatAgent:
    movies, ratings = load_movielens(folder)
    catalog = Catalog(movies)
    model = HybridRecommender().fit(catalog, ratings)
    return ChatAgent(model, ratings)


def main(argv=None):
    p = argparse.ArgumentParser(description="MovieMate: chat with a movie recommender.")
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data (offline test only)")
    a = p.parse_args(argv)
    folder = make_synthetic(Path("data") / "synthetic") if a.synthetic else (a.data or download_dataset())
    print("Loading data and training models ...")
    agent = build_agent(folder)
    print("MovieMate: Hi! Tell me movies you like (e.g. 'I like Toy Story'), then say 'recommend'. "
          "Type 'help' for more.")
    while True:
        try:
            text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nMovieMate: Goodbye!")
            break
        reply = agent.handle(text)
        print("MovieMate:", reply)
        if text.strip().lower() in {"quit", "exit", "bye"}:
            break


if __name__ == "__main__":
    main()
