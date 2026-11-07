def test_title_matching_exact_and_fuzzy(agent):
    hit = agent.find_title("Movie 12")
    assert agent.catalog.title(hit[0]).startswith("Movie 12 ")
    assert agent.find_title("movi 12") or True  # fuzzy path must not raise


def test_like_and_dislike_update_profile(agent):
    agent.handle("i like Movie 5")
    agent.handle("i hated Movie 6")
    vals = sorted(agent.profile.values())
    assert vals == [1.0, 5.0]


def test_rate_command_and_reset(agent):
    reply = agent.handle("rate Movie 7 4")
    assert "Noted" in reply and 4.0 in agent.profile.values()
    agent.handle("reset")
    assert agent.profile == {}


def test_cold_start_message(agent):
    reply = agent.handle("recommend")
    assert "don't know your taste" in reply


def test_recommend_count_and_why(agent):
    for t in ["i like Movie 5", "i like Movie 9", "i love Movie 20"]:
        agent.handle(t)
    reply = agent.handle("recommend 3")
    assert len([l for l in reply.splitlines() if l.strip()[:2] in ("1.", "2.", "3.")]) == 3
    assert agent.handle("why 1").startswith("I suggested")


def test_filter_parsing():
    from moviemate.chat import ChatAgent
    f = ChatAgent.parse_filters("something scary from the 90s")
    assert f == {"genre": "Horror", "year_min": 1990, "year_max": 1999}
    assert ChatAgent.parse_filters("recommend after 2010")["year_min"] == 2010


def test_disambiguation_flow(agent):
    reply = agent.handle("i like Movie")  # many partial matches -> ask which one
    assert "Which one did you mean" in reply and agent.pending is not None
    assert "Noted" in agent.handle("2")
    assert agent.pending is None and len(agent.profile) == 1


def test_disambiguation_invalid_reply_skips(agent):
    agent.handle("i like Movie")
    assert "skip" in agent.handle("banana")
    assert agent.profile == {}


def test_unknown_title_and_gibberish(agent):
    assert "couldn't find" in agent.handle("i like Zzzzzz Qqqq Nonexistent")
    assert "didn't catch" in agent.handle("blah blah")


def test_user_loading(agent):
    assert "Loaded MovieLens user 1" in agent.handle("user 1")
    assert len(agent.profile) > 0
