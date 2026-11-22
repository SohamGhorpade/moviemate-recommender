import json
import threading
import urllib.request

import pytest

from adaptive import AdaptationEngine
from server import make_server


@pytest.fixture()
def base(model):
    engine = AdaptationEngine(model, min_gap=1)
    srv = make_server(engine, port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_index_page_served(base):
    status, ctype, body = _get(base + "/")
    assert status == 200 and "text/html" in ctype and b"MovieMate Adaptive" in body


def test_state_endpoint(base):
    _, _, body = _get(base + "/api/state")
    s = json.loads(body)
    assert s["rows"][0]["genre"] == "Top picks for you"


def test_event_changes_state_over_http(base):
    before = json.loads(_get(base + "/api/state")[2])
    genre = before["rows"][-1]["genre"]
    item = before["rows"][-1]["items"][0]["id"]
    out = _post(base + "/api/event", {"kind": "like", "item": item, "row": genre})
    assert out["state"]["interactions"] == 1


def test_freeze_and_reset_endpoints(base):
    s = _post(base + "/api/freeze", {"frozen": True})["state"]
    assert s["frozen"] is True
    s = _post(base + "/api/reset", {})["state"]
    assert s["frozen"] is False and s["interactions"] == 0


def test_search_and_why(base):
    found = json.loads(_get(base + "/api/search?q=movie")[2])["items"]
    assert found
    why = json.loads(_get(base + f"/api/why?item={found[0]['id']}")[2])
    assert "text" in why


def test_unknown_route_is_404(base):
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(base + "/nope")
    assert err.value.code == 404
