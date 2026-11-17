"""Battlesnake HTTP service: minimal raw-ASGI app (no framework overhead). Original code."""

import json
import os
from pathlib import Path

from .board import tables_for, DIR_NAMES
from .state import parse_api_state
from .search import choose_move, choose_move_multi
from .eval import DEFAULT_WEIGHTS, Weights


def _load_weights() -> Weights:
    path = os.environ.get(
        "SNAKE_WEIGHTS_PATH",
        str(Path(__file__).resolve().parent.parent / "weights.json"),
    )
    try:
        return Weights(json.loads(Path(path).read_text()))
    except Exception:
        return Weights(DEFAULT_WEIGHTS)


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

WEIGHTS = _load_weights()

_HEADERS = [(b"content-type", b"application/json")]


def _handle_move(data: dict) -> dict:
    board = data["board"]
    t = tables_for(int(board.get("width", 11)), int(board.get("height", 11)))
    state, me, n = parse_api_state(data, t)
    if n < 2:
        # alone (solo mode): pick a safe greedy direction
        d, c, _ = choose_move_multi(t, state, me, min(MOVE_BUDGET, 0.35), WEIGHTS)
        return {"move": DIR_NAMES[d]}
    if n == 2:
        opp = 1 - me
        d, c, depth = choose_move(t, state, me, opp, MOVE_BUDGET, WEIGHTS, MAX_DEPTH)
    else:
        d, c, depth = choose_move_multi(t, state, me, MOVE_BUDGET, WEIGHTS)
    return {"move": DIR_NAMES[d], "shout": f"d{depth}"}


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
                payload = _handle_move(data)
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
