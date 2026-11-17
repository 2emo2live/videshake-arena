"""Local duel runner + hill-climbing harness (original code).

Usage:
  uv run run_local.py bench                 # move-time benchmark on random positions
  uv run run_local.py duel --n 20           # current weights vs baseline opponents
  uv run run_local.py climb --iters 30      # hill-climbing over weights
"""

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

from vibesnake.board import tables_for
from vibesnake.state import State, step, legal_moves, prune_dead
from vibesnake.search import choose_move
from vibesnake.eval import Weights, DEFAULT_WEIGHTS

T = tables_for(11, 11)
WEIGHTS_PATH = Path(__file__).parent / "weights.json"
HISTORY_PATH = Path(__file__).parent / "weights_history.jsonl"


def rand_state(rng: random.Random):
    """Random symmetric-ish start: both snakes length 3, 5 foods."""
    while True:
        ya = rng.randrange(1, 10)
        xa = rng.randrange(0, 8)
        yb = rng.randrange(1, 10)
        xb = rng.randrange(0, 8)
        # horizontal bodies, heads at right end (about to move right/up/down randomly later)
        a = tuple(ya * 11 + x for x in range(xa, xa + 3))[::-1]  # head at xa+2
        b = tuple(yb * 11 + x for x in range(xb, xb + 3))[::-1]
        if set(a) & set(b):
            continue
        ha, hb = a[0], b[0]
        dx = abs(ha % 11 - hb % 11)
        dy = abs(ha // 11 - hb // 11)
        if dx + dy < 6:
            continue
        bits_a = sum(1 << c for c in a)
        bits_b = sum(1 << c for c in b)
        occ = bits_a | bits_b
        foods = 0
        placed = 0
        while placed < 5:
            c = rng.randrange(121)
            if (occ >> c) & 1 or (foods >> c) & 1:
                continue
            foods |= 1 << c
            placed += 1
        health = rng.choice([100, 100, 100, 60, 40])
        return State((a, b), (bits_a, bits_b), (health, health), foods)


def play_game(wa: Weights, wb: Weights, rng: random.Random, budget=0.04,
              max_depth=6, max_turns=300, food_target=5):
    """Returns 'A', 'B' or None (draw)."""
    state = rand_state(rng)
    for turn in range(max_turns):
        moves = []
        for i, w in ((0, wa), (1, wb)):
            opp = 1 - i
            d, c, _ = choose_move(T, state, i, opp, budget, w, max_depth)
            moves.append(c)
        state, alive = step(T, state, moves)
        if not all(alive):
            if alive[0] and not alive[1]:
                return "A"
            if alive[1] and not alive[0]:
                return "B"
            return None
        state = prune_dead(T, state, alive)
        # respawn food (approximation of standard rules); step() already removed eaten food
        occ = state.bits[0] | state.bits[1]
        while state.food.bit_count() < food_target:
            c = rng.randrange(121)
            if not ((occ | state.food) >> c) & 1:
                state.food |= 1 << c
    la, lb = len(state.bodies[0]), len(state.bodies[1])
    if la > lb:
        return "A"
    if lb > la:
        return "B"
    return None


def series(wa, wb, n, rng, budget=0.08, max_depth=6, swap=True):
    """Play n games A(wa) vs B(wb); returns (wins_a, wins_b, draws) for wa-perspective with color swaps."""
    wins = {"A": 0, "B": 0, "D": 0}
    for g in range(n):
        if swap and g % 2 == 1:
            r = play_game(wb, wa, rng, budget, max_depth)
            if r == "A":
                r = "B"
            elif r == "B":
                r = "A"
        else:
            r = play_game(wa, wb, rng, budget, max_depth)
        if r == "A":
            wins["A"] += 1
        elif r == "B":
            wins["B"] += 1
        else:
            wins["D"] += 1
    return wins


# ---------------- opponent baselines (our own variants, not copied snakes) ---------------

def baseline_weights(kind: str) -> Weights:
    w = Weights(Weights.DEFAULTS)
    if kind == "greedy":  # food-driven, shallow, brave
        w.update({"area": 10.0, "food_race": 60.0, "food_dist": 30.0,
                  "head_danger": 100.0, "len": 10.0, "low_health_food": 20.0,
                  "mobility": 4.0})
    elif kind == "coward":  # over-cautious, stalls
        w.update({"area": 60.0, "food_race": 5.0, "food_dist": 3.0,
                  "head_danger": 2000.0, "head_kill": 5.0, "wall": 20.0})
    elif kind == "aggressive":  # hunts head-kills
        w.update({"head_kill": 600.0, "area": 14.0, "food_race": 20.0,
                  "len": 60.0})
    elif kind == "balanced":
        pass
    return w


def mutate(w: Weights, rng: random.Random, strength=0.25) -> Weights:
    nw = Weights(copy.deepcopy(dict(w)))
    keys = [k for k in nw if isinstance(nw[k], float)]
    for k in rng.sample(keys, k=max(1, len(keys) // 3)):
        v = nw[k]
        if v == 0:
            v = 1.0
        nw[k] = max(0.0, v * (1.0 + rng.uniform(-strength, strength)))
        if rng.random() < 0.1:
            nw[k] = max(0.0, v + rng.choice([-1, 1]) * abs(v) * rng.uniform(0.3, 1.0))
    return nw


def load_current() -> Weights:
    if WEIGHTS_PATH.exists():
        return Weights(json.loads(WEIGHTS_PATH.read_text()))
    return Weights(Weights.DEFAULTS)


def save_current(w: Weights):
    WEIGHTS_PATH.write_text(json.dumps(dict(w), indent=2))


def cmd_bench(args):
    rng = random.Random(0)
    w = load_current()
    total = 0.0
    depths = []
    n = args.n
    for i in range(n):
        state = rand_state(rng)
        # make it mid-game-ish: grow snakes randomly
        for _ in range(30):
            moves = [random.choice(legal_moves(T, state.bodies[j]))[1] for j in range(2)]
            state, alive = step(T, state, moves)
            if not all(alive):
                state = rand_state(rng)
        t0 = time.perf_counter()
        d, c, depth = choose_move(T, state, 0, 1, 0.33, w, 6)
        dt = time.perf_counter() - t0
        total += dt
        depths.append(depth)
    print(f"avg {total / n * 1000:.1f}ms  max_depth_reached hist: "
          f"{ {d: depths.count(d) for d in sorted(set(depths))} }")


def cmd_duel(args):
    rng = random.Random(args.seed)
    w = load_current()
    opps = args.opponents.split(",") if args.opponents else ["greedy", "coward", "aggressive", "balanced"]
    for kind in opps:
        wo = baseline_weights(kind)
        budget = args.opp_budget if kind in ("greedy",) else args.budget
        r = series(w, wo, args.n, rng, budget=budget, max_depth=args.max_depth)
        wr = r["A"] / max(1, r["A"] + r["B"])
        print(f"vs {kind:11s}: {r['A']}W {r['B']}L {r['D']}D  winrate={wr:.2f}")


def cmd_climb(args):
    rng = random.Random(args.seed)
    w = load_current()
    best = dict(w)
    save_current(Weights(best))
    history = []
    opps = [(k, baseline_weights(k)) for k in ("greedy", "coward", "aggressive", "balanced")]

    def fitness(cand):
        # candidate plays vs current best + baselines; returns mean winrate
        total_w, total_g = 0, 0
        r = series(Weights(cand), Weights(best), args.games // 2, rng, budget=args.budget)
        total_w += r["A"]
        total_g += r["A"] + r["B"]
        for _, wo in opps:
            r = series(Weights(cand), wo, args.games // 4, rng, budget=args.budget)
            total_w += r["A"]
            total_g += r["A"] + r["B"]
        return total_w / max(1, total_g)

    base_fit = fitness(best)
    print(f"start fitness={base_fit:.3f}")
    it = 0
    t_start = time.time()
    while it < args.iters and time.time() - t_start < args.time_limit:
        it += 1
        cand = mutate(Weights(best), rng, args.strength)
        f = fitness(dict(cand))
        line = f"iter {it}: fitness={f:.3f} (best {base_fit:.3f})"
        if f > base_fit + 0.02:
            best = dict(cand)
            base_fit = f
            save_current(Weights(best))
            line += "  ACCEPTED -> saved"
        print(line, flush=True)
        history.append({"iter": it, "fitness": f, "weights": dict(cand), "accepted": f >= base_fit})
        HISTORY_PATH.open("a").write(json.dumps(history[-1]) + "\n")
    print("final fitness:", base_fit)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["bench", "duel", "climb"])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--games", type=int, default=16)
    p.add_argument("--budget", type=float, default=0.04)
    p.add_argument("--opp-budget", type=float, default=0.02)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--strength", type=float, default=0.25)
    p.add_argument("--time-limit", type=float, default=3600.0)
    p.add_argument("--opponents", type=str, default="")
    args = p.parse_args()
    if args.cmd == "bench":
        cmd_bench(args)
    elif args.cmd == "duel":
        cmd_duel(args)
    else:
        cmd_climb(args)


if __name__ == "__main__":
    main()
