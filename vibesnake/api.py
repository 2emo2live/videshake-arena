"""Battlesnake HTTP service (FastAPI). Original implementation."""

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, Request

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

# time budget for /move (seconds); total response must stay under 500ms
MOVE_BUDGET = float(os.environ.get("SNAKE_MOVE_BUDGET", "0.33"))
MAX_DEPTH = int(os.environ.get("SNAKE_MAX_DEPTH", "6"))

app = FastAPI()
WEIGHTS = _load_weights()


@app.get("/")
async def index():
    return SNAKE_INFO


@app.post("/start")
async def start():
    return {}


@app.post("/end")
async def end():
    return {}


@app.post("/move")
async def move(request: Request):
    data = await request.json()
    board = data["board"]
    t = tables_for(int(board.get("width", 11)), int(board.get("height", 11)))
    state, me, n = parse_api_state(data, t)
    if n < 2:
        # alone: just move somewhere safe-ish via multi path with no opponents
        d, c, _ = choose_move_multi(t, state, me, min(MOVE_BUDGET, 0.35), DEFAULT_WEIGHTS)
        return {"move": DIR_NAMES[d]}
    if n == 2:
        opp = 1 - me
        d, c, depth = choose_move(t, state, me, opp, MOVE_BUDGET, WEIGHTS, MAX_DEPTH)
    else:
        d, c, depth = choose_move_multi(t, state, me, MOVE_BUDGET, DEFAULT_WEIGHTS)
    return {"move": DIR_NAMES[d], "shout": f"d{depth}"}


def main():
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
