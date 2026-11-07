# MovieMate: a hybrid movie recommender with a chat interface

MSAI-631 (Artificial Intelligence for Human-Computer Interaction) - recommendation system project.

MovieMate recommends movies from the MovieLens "ml-latest-small" dataset (about 100,000 ratings by 600+ users).
You talk to it in the terminal ("I like Toy Story", "recommend 5 comedy", "why 1").

## What it does

1. **Item-based collaborative filtering** (starting point) with several modifications:
   adjusted cosine similarity, shrinkage by co-rater count, minimum-ratings filter, top-k neighbors.
2. **Content model** using genre vectors and **Bayesian popularity** prior.
3. **Weighted hybrid** of the three scores (per-user min-max normalized); weights tuned on a validation split.
4. **Cold start**: new users with no ratings get popularity-based picks.
5. **Explanations**: `why 1` names the liked movies and genres behind a suggestion.
6. **Offline evaluation**: leave-last-liked-out protocol with HitRate@K, NDCG@K, MRR, coverage, mean popularity.

See `CREDITS.md` for the project and papers this builds on.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python -m moviemate.chat                 # downloads MovieLens on first run, then starts the chat
python -m moviemate.evaluate -k 10 --out results.json   # offline evaluation on real data
pytest                                   # unit tests (use synthetic data; no download needed)
python examples/demo_synthetic.py        # scripted chat demo on synthetic data (offline)
```

If the automatic download fails, download `ml-latest-small.zip` from https://grouplens.org/datasets/movielens/latest/,
unzip it, and pass the folder: `python -m moviemate.chat --data path/to/ml-latest-small`.

### Example chat commands

| You type | What happens |
|---|---|
| `i like Toy Story` / `i hated Titanic` / `rate Heat 4` | Adds to your profile (asks which one if the title is ambiguous) |
| `recommend` / `recommend 5 comedy` | Top-N list; optional genre/mood word |
| `something scary from the 90s` / `recommend after 2010` | Genre, decade and year filters |
| `why 2` | Explains recommendation #2 |
| `more like The Matrix` | Most similar movies |
| `user 12` | Loads an existing MovieLens user's ratings |
| `profile`, `reset`, `help`, `quit` | Utilities |

## Layout

```
moviemate/data.py      download/load MovieLens; synthetic data generator (tests only)
moviemate/models.py    popularity, item-CF, content model, hybrid
moviemate/evaluate.py  split, metrics, weight tuning, CLI
moviemate/chat.py      rule-based chat agent and terminal loop
tests/                 pytest suite
examples/              offline demo script
```

## Data license

MovieLens data is provided by GroupLens for research use and may not be redistributed, so it is not in this repository.
Results on the synthetic data are only for testing the code, not measures of recommendation quality.
