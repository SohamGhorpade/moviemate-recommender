"""Local web server for the adaptive MovieMate interface (Python standard library only).

Run:  python server.py              (downloads MovieLens ml-latest-small on first run)
      python server.py --synthetic  (offline demo with synthetic data)
Then open http://127.0.0.1:8000 in a browser. The state lives in memory for one user and is lost
when the server stops.
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from adaptive import AdaptationEngine
from data import DEFAULT_DIR, download_dataset, load_movielens, make_synthetic
from models import Catalog, HybridRecommender

HERE = Path(__file__).resolve().parent


def build_engine(synthetic: bool = False, data_dir: Path | None = None, seed: int = 0) -> AdaptationEngine:
    if synthetic:
        folder = make_synthetic(Path(data_dir or DEFAULT_DIR) / "synthetic")
        print("Demo mode: synthetic data (movie titles are placeholders).")
    else:
        folder = download_dataset(Path(data_dir or DEFAULT_DIR))
    movies, ratings = load_movielens(folder)
    print(f"Fitting the recommender on {len(ratings):,} ratings ...")
    model = HybridRecommender().fit(Catalog(movies), ratings)
    return AdaptationEngine(model, seed=seed)


def make_handler(engine: AdaptationEngine):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, payload, status=200, ctype="application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _decade(self, query):
            v = query.get("decade", [""])[0]
            return int(v) if v.isdigit() else None

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            with lock:
                if url.path in ("/", "/index.html"):
                    return self._send((HERE / "index.html").read_bytes(), ctype="text/html; charset=utf-8")
                if url.path == "/api/state":
                    return self._send(engine.state(self._decade(q)))
                if url.path == "/api/why":
                    item = int(q.get("item", ["-1"])[0])
                    return self._send({"text": engine.explain(item)})
                if url.path == "/api/search":
                    return self._send({"items": engine.search(q.get("q", [""])[0])})
            self._send({"error": "not found"}, 404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send({"error": "bad json"}, 400)
            path = urlparse(self.path).path
            with lock:
                if path == "/api/event":
                    engine.event(body.get("kind", ""), body.get("item"), body.get("row"), body.get("action"))
                    result = None
                    if body.get("action") == "surprise":
                        result = {"items": [x for x in [engine.surprise()] if x]}
                    elif body.get("action") == "likes":
                        result = {"items": engine.liked()}
                    return self._send({"state": engine.state(body.get("decade")), "result": result})
                if path == "/api/freeze":
                    engine.set_frozen(bool(body.get("frozen")))
                    return self._send({"state": engine.state()})
                if path == "/api/reset":
                    engine.reset()
                    return self._send({"state": engine.state()})
            self._send({"error": "not found"}, 404)

    return Handler


def make_server(engine: AdaptationEngine, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(engine))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="use generated demo data (no download)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    eng = build_engine(synthetic=args.synthetic)
    srv = make_server(eng, port=args.port)
    print(f"Open http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
