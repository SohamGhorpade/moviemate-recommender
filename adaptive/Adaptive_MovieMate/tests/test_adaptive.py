import numpy as np

from adaptive import ACTIONS, AdaptationEngine, DEFAULT_PRIMARY, GenreBandit, ToolbarAdapter


def test_bandit_prior_starts_equal_and_rises_with_reward():
    b = GenreBandit(["A", "B"])
    assert b.mean("A") == b.mean("B")
    b.update("A", 1.0)
    assert b.mean("A") > b.mean("B")


def test_bandit_ignores_unknown_genre():
    b = GenreBandit(["A"])
    b.update("Z", 1.0)
    assert "Z" not in b.alpha


def test_bandit_ranking_is_deterministic_per_version():
    b = GenreBandit(["A", "B", "C"], seed=3)
    b.update("C", 1.0, weight=3)
    assert b.rank(4) == b.rank(4)


def test_untouched_genres_keep_default_order():
    b = GenreBandit(["X", "Y", "Z"])
    assert b.rank(1) == ["X", "Y", "Z"]


def test_toolbar_default_then_frequency_decides():
    t = ToolbarAdapter(n_primary=3)
    assert t.split()[0] == DEFAULT_PRIMARY
    for _ in range(6):
        t.record("likes")
    assert "likes" in t.split()[0]
    assert len(t.split()[0]) == 3
    assert set(t.split()[0] + t.split()[1]) == set(ACTIONS)


def test_engine_rows_and_state_shape(model):
    e = AdaptationEngine(model)
    s = e.state()
    assert s["rows"][0]["genre"] == "Top picks for you"
    assert len(s["rows"]) == 1 + e.n_rows
    assert all(len(r["items"]) <= e.per_row for r in s["rows"])
    assert s["toolbar"]["primary"][0]["id"] == DEFAULT_PRIMARY[0]


def test_liking_a_genre_moves_its_row_up(model):
    e = AdaptationEngine(model, min_gap=1)
    movies = e.catalog.movies
    target_genre = e.default_order[-1]  # start from the bottom
    ids = [i for i, gl in enumerate(movies["genre_list"]) if target_genre in gl][:12]
    before = e.row_order.index(target_genre)
    for i in ids:
        e.event("like", item=i, row=target_genre)
    assert e.row_order.index(target_genre) < before


def test_freeze_stops_layout_changes(model):
    e = AdaptationEngine(model, min_gap=1)
    e.set_frozen(True)
    before = list(e.row_order)
    for i in range(10):
        e.event("like", item=i, row=e.default_order[-1])
    assert e.row_order == before


def test_reset_restores_default_layout(model):
    e = AdaptationEngine(model, min_gap=1)
    for i in range(10):
        e.event("like", item=i, row=e.default_order[-1])
    e.reset()
    assert e.row_order == e.default_order
    assert e.profile == {} and e.n == 0


def test_repeated_why_turns_on_inline_explanations(model):
    e = AdaptationEngine(model, min_gap=1)
    for _ in range(3):
        e.event("action", action="explain")
    assert e.explain_inline is True
    assert any("explanations" in x["text"].lower() for x in e.log)


def test_explanation_text_names_liked_movies(model):
    e = AdaptationEngine(model)
    e.event("like", item=0, row=e.default_order[0])
    text = e.explain(1)
    assert text.startswith("Because") or "genre" in text or "viewers" in text


def test_adaptation_is_logged_in_plain_language(model):
    e = AdaptationEngine(model, min_gap=1)
    for i in range(12):
        e.event("like", item=i, row=e.default_order[-1])
    assert e.log, "expected at least one logged adaptation"
    assert all(isinstance(x["text"], str) and x["text"] for x in e.log)


def test_engine_keeps_nothing_on_disk(model, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    e = AdaptationEngine(model)
    e.event("like", item=0, row=e.default_order[0])
    assert list(tmp_path.iterdir()) == []
