"""Search correctness: alpha-beta must equal a naive full-width minimax.

Catches the classic fail-low bug: a move that scores <= alpha must not abort
the remaining sibling moves.

Run from the repo root:  uv run python tests/test_search.py
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vibesnake.board import tables_for
from vibesnake.eval import MATE, Weights, evaluate
from vibesnake.search import _Clock, _search, choose_move
from vibesnake.state import Sim, legal_moves, step
from run_local import rand_state

T = tables_for(11, 11)
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights.json"
INF = float("inf")


def naive(t, state, me, opp, depth, w):
    if depth <= 0:
        return evaluate(t, state, me, opp, w)
    best = -INF
    for _, mt in legal_moves(t, state.bodies[me]):
        worst = INF
        for _, ot in legal_moves(t, state.bodies[opp]):
            targets = [0, 0]
            targets[me] = mt
            targets[opp] = ot
            ns, alive = step(t, state, targets)
            if not alive[me]:
                v = w["draw_value"] if not alive[opp] else -(MATE - (10 - depth))
            elif not alive[opp]:
                v = MATE - (10 - depth)
            else:
                v = naive(t, ns, me, opp, depth - 1, w)
            worst = min(worst, v)
        best = max(best, worst)
    return best


def sim_root(t, state, me, opp, depth, w):
    sim = Sim.from_state(state)
    opp_moves = legal_moves(t, state.bodies[opp])
    out = {}
    for _, mt in legal_moves(t, state.bodies[me]):
        worst = INF
        for _, ot in opp_moves:
            targets = [0, 0]
            targets[me] = mt
            targets[opp] = ot
            rec, alive = sim.apply(t, targets)
            if not alive[me]:
                v = w["draw_value"] if not alive[opp] else -(MATE - (10 - depth))
                sim.undo(rec)
            elif not alive[opp]:
                v = MATE - (10 - depth)
                sim.undo(rec)
            else:
                try:
                    v = _search(t, sim, me, opp, depth - 1, -INF, INF,
                                _Clock(time.perf_counter() + 30), w)
                finally:
                    sim.undo(rec)
            worst = min(worst, v)
        out[mt] = worst
    return out


def main():
    w = Weights(json.loads(WEIGHTS_PATH.read_text()))
    # root food pull is a deliberate tie-breaker, not part of plain minimax
    w["root_food"] = 0.0
    w["root_food_dist"] = 0.0
    rng = random.Random(3)
    checked = 0
    for trial in range(8):
        s = rand_state(rng)
        for _ in range(rng.randrange(0, 10)):
            moves = [rng.choice(legal_moves(T, s.bodies[j]))[1] for j in range(2)]
            s, alive = step(T, s, moves)
            if not all(alive):
                s = rand_state(rng)
                break
        if len(legal_moves(T, s.bodies[0])) < 2:
            continue
        depth = 3
        ref = naive(T, s, 0, 1, depth, w)
        got = sim_root(T, s, 0, 1, depth, w)
        best = max(got, key=got.get)
        assert abs(got[best] - ref) <= 1.0, (
            f"trial {trial}: alpha-beta {got[best]:.1f} != minimax {ref:.1f}"
        )
        d, c, _ = choose_move(T, s, 0, 1, 30.0, w, depth)
        assert abs(got[c] - got[best]) <= 1.0, (
            f"trial {trial}: choose_move picked a suboptimal move"
        )
        checked += 1
    assert checked >= 5, f"too few positions checked: {checked}"
    print(f"search tests passed ({checked} positions)")


if __name__ == "__main__":
    main()
