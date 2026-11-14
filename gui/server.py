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

from gui import chat, game, replays

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
            "pairs": E.PAIRS, "maxWins": game.MAX_WINS, "lives": game.START_LIVES,
            "perks": [game._status(sid) for sid in game.data["statuses"]]}


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
        if self.path.startswith("/static/"):
            name = self.path[len("/static/"):].split("?")[0]
            types = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
            ext = os.path.splitext(name)[1]
            ok = name and all(c.isalnum() or c in "-_." for c in name) and ".." not in name
            path = os.path.join(HERE, "static", name)
            if ok and ext in types and os.path.isfile(path):
                with open(path, "rb") as f:
                    return self._send(200, f.read(), types[ext])
            return self._send(404, json.dumps({"error": "Not found"}))
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
            save(mid)
            return {"ok": True, "id": mid, "state": MATCHES[mid].state()}
        if path == "/api/replays":
            return {"ok": True, "replays": replays.listing()}
        if path == "/api/replay":
            return {"ok": True, "replay": replays.get(req.get("id", ""))}
        if path == "/api/replay/rename":
            return {"ok": True, "meta": replays.rename(req.get("id", ""), req.get("title"), req.get("note"))}
        if path == "/api/replay/delete":
            replays.delete(req.get("id", ""))
            return {"ok": True}
        if path == "/api/series":
            for k in (req.get("a"), req.get("b")):
                if k not in game.AGENTS:
                    raise Exception(f"Unknown agent {k}")
            return {"ok": True, "series": series(req["a"], req["b"], req.get("n", 20))}
        mid = req.get("id", "")
        m = find(mid)
        if m is None:
            raise Exception("That game is no longer saved. Start a new one.")
        if path == "/api/state":
            return {"ok": True, "id": mid, "state": m.state()}
        if path == "/api/log":
            return {"ok": True, "events": m.events_since(req.get("since", 0))}
        if path == "/api/chat":
            if req.get("message") is None:
                return {"ok": True, "chat": m.chat, "claude": chat.claude_available()}
            mine = {"from": "you", "text": str(req["message"])[:500], "seat": None}
            out = chat.reply(m, req["message"], req.get("seat"))
            mine["seat"] = out["seat"]
            m.chat += [mine, out]
            m.chat[:] = m.chat[-200:]
            save(mid)
            return {"ok": True, "reply": out, "claude": chat.claude_available()}
        rounds = len(m.history)
        if path == "/api/act":
            out = m.do(req["name"], *req.get("args", []))
        elif path == "/api/step":
            out = m.step_json()
        else:
            raise Exception("Unknown request")
        save(mid)
        if len(m.history) != rounds:
            try:
                replays.record(mid, m)
            except OSError as e:
                print(f"Could not save replay {mid}: {e}")
        return out


SAVE_DIR = os.path.join(HERE, "saves")
MAX_SAVES = 100


def _save_path(mid):
    if not (len(mid) == 32 and all(c in "0123456789abcdef" for c in mid)):
        return None
    return os.path.join(SAVE_DIR, mid + ".json")


def save(mid):
    """Write the match to gui/saves/<id>.json so it survives a refresh or restart."""
    path = _save_path(mid)
    if path is None or mid not in MATCHES:
        return
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(MATCHES[mid].to_save(), f, default=str)
        os.replace(tmp, path)
        saves = sorted(glob.glob(os.path.join(SAVE_DIR, "*.json")), key=os.path.getmtime)
        for old in saves[:-MAX_SAVES]:
            os.remove(old)
    except OSError as e:
        print(f"Could not save game {mid}: {e}")


def find(mid):
    """The match with this id, loading it from gui/saves/ if needed."""
    if mid in MATCHES:
        return MATCHES[mid]
    path = _save_path(mid)
    if path is None or not os.path.exists(path):
        return None
    with open(path) as f:
        m = game.Match.from_save(json.load(f))
    m.rounds = replays.load_rounds(mid)
    for s in m.seats:
        if s.kind != "human" and s.kind not in game.AGENTS:
            raise Exception(f"This saved game uses agent {s.kind}, which is not loaded.")
    if len(MATCHES) >= MAX_MATCHES:
        MATCHES.pop(next(iter(MATCHES)))
    MATCHES[mid] = m
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description="sapai GUI server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8405)
    ap.add_argument("--agent", action="append", default=[],
                    help="Python file defining shop(seat, rng); can be repeated")
    args = ap.parse_args(argv)
    load_models()
    added = replays.import_saves(SAVE_DIR, game.Match.from_save)
    if added:
        print(f"Added {added} earlier saved games to the replay catalog")
    for path in args.agent:
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            game.load_agent_source(name, f.read())
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"sapai arena on http://{args.host}:{args.port}  ({len(game.AGENTS)} agents)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
