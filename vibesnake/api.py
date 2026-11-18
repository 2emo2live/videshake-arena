"""Battlesnake HTTP service: minimal raw-ASGI app (no framework overhead). Original code.

Concurrency: the search is CPU-bound, so when several /move requests arrive at
once (the arena runs parallel games), the per-move budget is divided by the
number of in-flight searches and overflow requests get an instant safe move
instead of a full search. This keeps every response well under the 500 ms game
budget. The search itself runs in a worker thread so the event loop is never
blocked.
"""

import asyncio
import hashlib
import itertools
import json
import os
import sys
import threading
import time
from pathlib import Path

from .board import tables_for, DIR_NAMES
from .state import parse_api_state, step, legal_moves
from .search import choose_move, choose_move_multi
from .eval import DEFAULT_WEIGHTS, Weights


def _load_weights() -> Weights:
    path = os.environ.get(
        "SNAKE_WEIGHTS_PATH",
        str(Path(__file__).resolve().parent.parent / "weights.json"),
    )
    try:
        return Weights(json.loads(Path(path).read_text()))
    except Exception as e:
        print(f"WARNING: could not load weights from {path}: {e}; falling back to defaults", file=sys.stderr)
        return Weights(DEFAULT_WEIGHTS)


def _weights_version(w: Weights) -> str:
    """Build id shown in the hello response: weights plus engine sources.

    Hashing the sources too lets us verify from outside which build is live
    (weight-only hashes made code changes invisible).
    """
    h = hashlib.sha1(json.dumps(dict(w), sort_keys=True).encode())
    for name in ("api.py", "search.py", "eval.py", "state.py", "board.py"):
        try:
            h.update(Path(__file__).with_name(name).read_bytes())
        except OSError:
            pass
    return h.hexdigest()[:8]


SNAKE_INFO = {
    "apiversion": "1",
    "author": "ivsem",
    "color": "#4C6EF5",
    "head": "trans-rights-scarf",
    "tail": "hook",
}

# time budget for /move (seconds); total response must stay under 500ms.
# Render free tier runs on a very slow 0.1 vCPU: keep the default low there.
MOVE_BUDGET = float(os.environ.get("SNAKE_MOVE_BUDGET", "0.33"))
MAX_DEPTH = int(os.environ.get("SNAKE_MAX_DEPTH", "6"))
# in-flight searches above which requests get the instant safe move
MAX_INFLIGHT = int(os.environ.get("SNAKE_MAX_INFLIGHT", "4"))

WEIGHTS = _load_weights()
SNAKE_INFO["version"] = _weights_version(WEIGHTS)

_HEADERS = [(b"content-type", b"application/json")]

_inflight_lock = threading.Lock()
_inflight = 0


def _instant_safe_move(t, state, me: int) -> int:
    """No-search fallback: pick the neighbor move that survives every opponent
    reply and is closest to food. Runs in microseconds."""
    n = len(state.bodies)
    opp_moves = [legal_moves(t, state.bodies[i]) for i in range(n) if i != me]
    combos = list(itertools.product(*opp_moves)) if opp_moves else [()]
    foods = [f for f in range(t.ncells) if (state.food >> f) & 1]
    best = None
    for d, c in legal_moves(t, state.bodies[me]):
        risk = 0
        for combo in combos:
            targets = [0] * n
            targets[me] = c
            k = 0
            for i in range(n):
                if i != me:
                    targets[i] = combo[k][1]
                    k += 1
            _, alive = step(t, state, targets)
            if not alive[me]:
                risk += 1
        fx, fy = c % t.w, c // t.w
        fd = min((abs(fx - f % t.w) + abs(fy - f // t.w)) for f in foods) if foods else 0
        key = (risk, fd)
        if best is None or key < best[0]:
            best = (key, d)
    return best[1] if best is not None else 0


def _handle_move_sync(data: dict) -> dict:
    board = data["board"]
    t = tables_for(int(board.get("width", 11)), int(board.get("height", 11)))
    state, me, n = parse_api_state(data, t)
    if n < 2:
        d, c, _ = choose_move_multi(t, state, me, min(MOVE_BUDGET, 0.35), WEIGHTS)
        return {"move": DIR_NAMES[d]}
    global _inflight
    with _inflight_lock:
        active = _inflight
        _inflight += 1
    try:
        if active >= MAX_INFLIGHT:
            # overloaded: skip the search entirely, answer instantly
            d = _instant_safe_move(t, state, me)
            return {"move": DIR_NAMES[d], "shout": "instant"}
        budget = MOVE_BUDGET / (active + 1)
        if n == 2:
            opp = 1 - me
            d, c, depth = choose_move(t, state, me, opp, budget, WEIGHTS, MAX_DEPTH)
        else:
            d, c, depth = choose_move_multi(t, state, me, budget, WEIGHTS)
        return {"move": DIR_NAMES[d], "shout": f"d{depth}"}
    finally:
        with _inflight_lock:
            _inflight -= 1


async def _handle_move(data: dict) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _handle_move_sync, data)


async def app(scope, receive, send):
    """Raw ASGI application: GET / -> info, POST /start /move /end."""
    if scope["type"] != "http":
        return
    path = scope["path"].rstrip("/") or "/"
    method = scope["method"]

    body = b""
    if method == "POST":
        while True:
            msg = await receive()
            body += msg.get("body", b"")  # ASGI: request body chunks live under "body"
            if not msg.get("more_body"):
                break

    if method == "GET" and path == "/":
        payload = SNAKE_INFO
    elif method == "POST" and (path == "/start" or path == "/end"):
        payload = {}
    elif method == "POST" and path == "/move":
        try:
            data = json.loads(body)
        except Exception:
            payload = {"move": "up"}
        else:
            try:
                payload = await _handle_move(data)
            except Exception:
                # never fail a move request: return any legal-ish direction
                payload = {"move": "up"}
    else:
        await send({
            "type": "http.response.start",
            "status": 404,
            "headers": _HEADERS,
        })
        await send({"type": "http.response.body", "body": b""})
        return

    resp = json.dumps(payload).encode()
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": _HEADERS,
    })
    await send({"type": "http.response.body", "body": resp})


def main():
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning", workers=1)


if __name__ == "__main__":
    main()
