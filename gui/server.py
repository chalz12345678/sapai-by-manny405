"""
Local web server for the sapai GUI. Standard library only, no torch needed.

    python -m gui.server                       # then open http://127.0.0.1:8405
    python -m gui.server --agent my_agent.py   # add your own agent

Play against the greedy bot or a trained model, or pick two agents and watch
them play each other. Trained models are the exports in gui/models/ (refresh
them with ``python -m gui.export_models`` after training).

A custom agent is a Python file defining ``shop(seat, rng)``; see
``gui/README.md``. It is loaded once at startup from your own disk.
"""

import argparse
import glob
import json
import os
import random
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gui import game

HERE = os.path.dirname(os.path.abspath(__file__))
MATCHES = {}
LOCK = threading.Lock()
MAX_MATCHES = 200


def load_models():
    for path in sorted(glob.glob(os.path.join(HERE, "models", "*.npz"))):
        game.load_model(os.path.basename(path)[:-4], path)


def meta():
    E = game.E
    agents = game.agent_list()
    order = {"model": 0, "bot": 1, "custom": 2}
    agents.sort(key=lambda a: (order.get(a["group"], 3), -(a.get("rating") or 0)))
    return {"agents": agents, "actionStart": E.ACTION_START, "maxTeam": E.MAX_TEAM,
            "pairs": E.PAIRS, "maxWins": game.MAX_WINS, "lives": game.START_LIVES}


def series(a, b, n):
    """Play n full games between agents a and b and count the results."""
    n = max(1, min(int(n), 200))
    out = {"games": n, "a": 0, "b": 0, "unfinished": 0, "rounds": 0,
           "battles": [0, 0, 0]}
    t = time.time()
    for _ in range(n):
        m = game.Match(a, b, seed=random.randrange(1 << 30))
        while not m.over and len(m.history) < 40:
            m.play_round()
        if m.result == "seat0":
            out["a"] += 1
        elif m.result == "seat1":
            out["b"] += 1
        else:
            out["unfinished"] += 1
        for h in m.history:
            out["battles"][["win", "loss", "draw"].index(h)] += 1
        out["rounds"] += len(m.history)
    out["seconds"] = round(time.time() - t, 2)
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if self.path == "/api/meta":
            return self._send(200, json.dumps(meta()))
        return self._send(404, json.dumps({"error": "Not found"}))

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            with LOCK:
                out = self.route(self.path, req)
            return self._send(200, out if isinstance(out, str) else json.dumps(out))
        except Exception as e:
            return self._send(400, json.dumps({"ok": False, "error": str(e)}))

    def route(self, path, req):
        if path == "/api/new":
            a, b = req.get("seat0", "human"), req.get("seat1", "greedy")
            for k in (a, b):
                if k != "human" and k not in game.AGENTS:
                    raise Exception(f"Unknown agent {k}")
            if len(MATCHES) >= MAX_MATCHES:
                MATCHES.pop(next(iter(MATCHES)))
            mid = uuid.uuid4().hex
            MATCHES[mid] = game.Match(a, b, seed=req.get("seed"))
            return {"ok": True, "id": mid, "state": MATCHES[mid].state()}
        if path == "/api/series":
            for k in (req.get("a"), req.get("b")):
                if k not in game.AGENTS:
                    raise Exception(f"Unknown agent {k}")
            return {"ok": True, "series": series(req["a"], req["b"], req.get("n", 20))}
        m = MATCHES.get(req.get("id", ""))
        if m is None:
            raise Exception("That game has expired. Start a new one.")
        if path == "/api/act":
            return m.do(req["name"], *req.get("args", []))
        if path == "/api/step":
            return m.step_json()
        raise Exception("Unknown request")


def main(argv=None):
    ap = argparse.ArgumentParser(description="sapai GUI server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8405)
    ap.add_argument("--agent", action="append", default=[],
                    help="Python file defining shop(seat, rng); can be repeated")
    args = ap.parse_args(argv)
    load_models()
    for path in args.agent:
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            game.load_agent_source(name, f.read())
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"sapai arena on http://{args.host}:{args.port}  ({len(game.AGENTS)} agents)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
