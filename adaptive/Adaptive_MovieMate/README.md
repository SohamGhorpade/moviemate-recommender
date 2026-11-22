# Adaptive MovieMate: an adaptive, AI-based movie interface

A browser interface that adapts to the user's interactions at runtime. Genre rows re-rank as the user
opens, likes, or skips movies (Thompson sampling over genres); the toolbar promotes the actions a user
uses most; explanations appear inline after repeated "Why?" requests; and the user can freeze the layout
or reset everything learned. Recommendations come from a hybrid model (item-based collaborative
filtering, genre content, popularity) reused from the author's earlier MovieMate project.

## Run

    pip install -r requirements.txt
    python server.py --synthetic    # offline demo, no download
    python server.py                # real MovieLens ml-latest-small (downloaded on first run)

Open http://127.0.0.1:8000 in a browser. The state lives in memory only and is lost when the server stops.

## Test and simulate

    python -m pytest -q             # 19 tests: adaptation rules, freeze/reset, HTTP endpoints
    python simulate.py --users 100  # simulated users: adaptive vs static layout, toolbar cost

The simulation uses simulated users with fixed favorite genres. It shows how the mechanism behaves,
not how real people respond.

## Files

- `adaptive.py`: adaptation logic (genre bandit, toolbar, explanation mode, freeze and reset)
- `server.py`: standard-library HTTP server and JSON API
- `index.html`: the single-page interface
- `models.py`, `data.py`: recommender and data loading, reused from the MovieMate project
- `simulate.py`: offline simulation
- `tests/`: pytest suite

MovieLens data is not included, because its license does not allow redistribution. `data.py`
downloads it from GroupLens on first run.
