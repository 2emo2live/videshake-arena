"""Alpha-beta search over simultaneous (duel) moves with iterative deepening.

Original implementation; standard game-tree techniques (alpha-beta pruning,
iterative deepening, time control) applied to our own simulation and evaluation.
"""

import time

from .board import Tables
from .state import State, Sim, legal_moves, step, vacating_tail
from .eval import BIG, MATE, Weights, evaluate, _food_waves

INF = float("inf")


class Abort(Exception):
    pass


class _Clock:
    __slots__ = ("deadline", "counter")

    def __init__(self, deadline):
        self.deadline = deadline
        self.counter = 0

    def check(self):
        self.counter += 1
        if self.counter >= 16:
            self.counter = 0
            if time.perf_counter() > self.deadline:
                raise Abort


def _order_my_moves(t, state, me, opp, moves, w):
    """Cheap static ordering: avoid immediate collisions first, then food/centrality."""
    opp_bits = state.bits[opp]
    my_bits = state.bits[me]
    opp_head = state.bodies[opp][0]
    scored = []
    my_tail = vacating_tail(state.bodies[me])
    opp_tail = vacating_tail(state.bodies[opp])
    for d, c in moves:
        s = 0.0
        cellb = 1 << c
        occupied = (my_bits | opp_bits) & ~my_tail & ~opp_tail
        if (occupied >> c) & 1:
            s -= 1000.0
        if (state.food >> c) & 1:
            s += 50.0
        if state.healths[me] < 50:
            s -= bfs_dist_cell(t, c, state, me) * 2.0
        s -= abs(c % t.w - opp_head % t.w) + abs(c // t.w - opp_head // t.w)
        scored.append((s, d, c))
    scored.sort(reverse=True)
    return [(d, c) for _, d, c in scored]


def bfs_dist_cell(t, c, state, me):
    # rough: manhattan to nearest food
    best = t.ncells
    fx = c % t.w
    fy = c // t.w
    f = state.food
    while f:
        cell = (f & -f).bit_length() - 1
        f &= f - 1
        d = abs(cell % t.w - fx) + abs(cell // t.w - fy)
        if d < best:
            best = d
    return best


def _search(t, sim, me, opp, depth, alpha, beta, clock, w):
    """Alpha-beta over the mutable Sim state with apply/undo (no per-node allocs)."""
    clock.check()
    if depth <= 0:
        return evaluate(t, sim, me, opp, w)

    my_moves = legal_moves(t, sim.bodies[me])
    opp_moves = legal_moves(t, sim.bodies[opp])
    if len(my_moves) > 1:
        my_moves = _order_my_moves(t, sim, me, opp, my_moves, w)

    best = -INF
    for md, mt in my_moves:
        vals = []
        for od, ot in opp_moves:
            targets = [0, 0]
            targets[me] = mt
            targets[opp] = ot
            rec, alive = sim.apply(t, targets)
            if not alive[me]:
                sim.undo(rec)
                if not alive[opp]:
                    # mutual destruction: a draw, worth the standing position
                    vals.append(w["draw_value"])
                else:
                    vals.append(-(MATE - (10 - depth)))
                    break
            elif not alive[opp]:
                sim.undo(rec)
                vals.append(MATE - (10 - depth))
            else:
                try:
                    vals.append(_search(t, sim, me, opp, depth - 1, alpha, INF, clock, w))
                finally:
                    sim.undo(rec)
            if vals[-1] <= alpha:
                # opponent already has a reply that keeps this move at or
                # below our best so far; the min can only get worse, so stop
                # examining replies (this is the valid min-node cutoff)
                break
        worst = min(vals)
        if worst > best:
            best = worst
            if best > alpha:
                alpha = best
                if alpha >= beta:
                    return best
    return best


def choose_move(t: Tables, state: State, me: int, opp: int, budget: float, w: Weights,
                max_depth: int = 6):
    """Returns (dir_index, target_cell, depth_reached)."""
    start = time.perf_counter()
    deadline = start + budget
    my_moves = legal_moves(t, state.bodies[me])

    if len(my_moves) == 1:
        return my_moves[0][0], my_moves[0][1], 0

    ordered = _order_my_moves(t, state, me, opp, my_moves, w)
    sim = Sim.from_state(state)
    opp_root_moves = legal_moves(t, state.bodies[opp])

    # Root food pull: a constant, move-specific tie-breaker added to the
    # searched value. Leaf food terms are diluted by search depth and the
    # opponent model, which made the snake circle food without eating.
    # Safety is unaffected: a line that dies is near -MATE and no bonus can
    # lift it over a playable continuation.
    bonus = {}
    foods = state.food
    if foods and (w["root_food"] or w["root_food_dist"]):
        my_blocked = (state.bits[me] & ~(1 << state.bodies[me][-1])) | state.bits[opp]
        hp = state.healths[me]
        urg = 1.0 + w["root_hunger"] * max(0, 100 - hp) / 100.0
        for _, mt in my_moves:
            if (foods >> mt) & 1:
                b = w["root_food"]
            else:
                waves = _food_waves(t, mt, my_blocked, foods)
                b = -w["root_food_dist"] * min(waves.values()) if waves else 0.0
            bonus[mt] = b * urg

    best_d, best_c = ordered[0]
    depth_done = 0
    prev_best = (best_d, best_c)

    for depth in range(2, max_depth + 1):
        clock = _Clock(deadline)
        try:
            alpha = -INF
            beta = INF
            cur_best_d = cur_best_c = None
            moves = ordered
            # search previous best first
            if prev_best in moves:
                moves = [prev_best] + [m for m in moves if m != prev_best]
            for md, mt in moves:
                vals = []
                b = bonus.get(mt, 0.0)
                child_alpha = alpha - b
                for od, ot in opp_root_moves:
                    targets = [0, 0]
                    targets[me] = mt
                    targets[opp] = ot
                    rec, alive = sim.apply(t, targets)
                    if not alive[me]:
                        sim.undo(rec)
                        if not alive[opp]:
                            vals.append(w["draw_value"])
                        else:
                            vals.append(-(MATE - (10 - depth)))
                            break
                    elif not alive[opp]:
                        sim.undo(rec)
                        vals.append(MATE - (10 - depth))
                    else:
                        try:
                            vals.append(_search(t, sim, me, opp, depth - 1, child_alpha, INF, clock, w))
                        finally:
                            sim.undo(rec)
                    if vals[-1] + b <= alpha:
                        break
                worst = min(vals)
                val = worst + b
                if cur_best_d is None or val > alpha:
                    alpha = val
                    cur_best_d, cur_best_c = md, mt
            if cur_best_d is not None:
                best_d, best_c = cur_best_d, cur_best_c
                prev_best = (best_d, best_c)
                depth_done = depth
            else:
                break
        except Abort:
            break

    return best_d, best_c, depth_done


def choose_move_multi(t: Tables, state: State, me: int, budget: float, w: Weights):
    """Fallback for 3+ snakes: worst-case 1-ply over all opponent joint moves."""
    start = time.perf_counter()
    deadline = start + budget
    n = len(state.bodies)
    others = [i for i in range(n) if i != me]
    opp_move_lists = [legal_moves(t, state.bodies[i]) for i in others]
    my_moves = legal_moves(t, state.bodies[me])
    ordered = _order_my_moves_multi(t, state, me, my_moves)

    best_d, best_c = ordered[0]
    best_score = -INF
    for depth in (1, 2):
        clock = _Clock(deadline)
        cur_best = None
        cur_score = -INF
        try:
            for md, mt in ordered:
                worst = INF
                import itertools
                for combo in itertools.product(*opp_move_lists):
                    targets = [None] * n
                    targets[me] = mt
                    for k, i in enumerate(others):
                        targets[i] = combo[k][1]
                    new_state, alive = step(t, state, targets)
                    if not alive[me]:
                        val = -MATE
                    else:
                        val = evaluate(t, new_state, me, -1, w) if False else _eval_multi(t, new_state, me, w)
                    if val < worst:
                        worst = val
                if worst > cur_score:
                    cur_score = worst
                    cur_best = (md, mt)
                if time.perf_counter() > deadline:
                    break
            if cur_best is not None and depth == 2:
                return cur_best[0], cur_best[1], depth
            if cur_best is not None and depth == 1:
                best_d, best_c, best_score = cur_best[0], cur_best[1], cur_score
        except Abort:
            break
    return best_d, best_c, 1


def _eval_multi(t, state, me, w):
    """Evaluate against the nearest (by flood) opponent."""
    from .eval import evaluate
    others = [i for i in range(len(state.bodies)) if i != me]
    best_val = None
    for o in others:
        v = evaluate(t, state, me, o, w)
        if best_val is None or v < best_val:
            best_val = v
    return best_val if best_val is not None else 0.0


def _order_my_moves_multi(t, state, me, moves):
    from .eval import manhattan
    others_heads = [state.bodies[i][0] for i in range(len(state.bodies)) if i != me]
    scored = []
    for d, c in moves:
        s = 0.0
        if (state.food >> c) & 1:
            s += 50.0
        for oh in others_heads:
            s -= manhattan(t, c, oh)
        scored.append((s, d, c))
    scored.sort(reverse=True)
    return [(d, c) for _, d, c in scored]
