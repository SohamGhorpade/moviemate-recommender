"""Scripted chat demo on SYNTHETIC data (offline smoke test; not real MovieLens results).

Run from the project root:  python examples/demo_synthetic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moviemate.chat import build_agent
from moviemate.data import make_synthetic

agent = build_agent(make_synthetic(Path("data/synthetic")))
SCRIPT = ["help", "recommend", "I like Movie 12", "i like movie 5", "1", "i hate Movie 40",
          "recommend 5 comedy", "why 1", "more like Movie 12", "profile",
          "something scary from the 90s", "after 2010 recommend", "rate Movie 7 4", "xyz"]
for line in SCRIPT:
    print("You:", line)
    print("MovieMate:", agent.handle(line), "\n")
