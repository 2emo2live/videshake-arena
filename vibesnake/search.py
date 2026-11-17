"""Alpha-beta search over simultaneous (duel) moves with iterative deepening.

Original implementation; standard game-tree techniques (alpha-beta pruning,
iterative deepening, time control) applied to our own simulation and evaluation.
"""

import time

from .board import Tables
from .state import State, legal_moves, step
from .eval import BIG, MATE, Weights, evaluate

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
        if self.counter >= 256:
            self.counter = 0
            if time.perf_counter() > self.deadline:
                raise Abort


def _order_my_moves(t, state, me, opp, moves, w):
    """Cheap static ordering: avoid immediate collisions first, then food/centrality."""
    opp_bits = state.bits[opp]
    my_bits = state.bits[me]
    opp_head = state.bodies[opp][0]
    scored = []
    my_tail = 1 << state.bodies[me][-1]
    opp_tail = 1 << state.bodies[opp][-1]
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


def _search(t, state, me, opp, depth, alpha, beta, clock, w):
    clock.check()
    if depth <= 0:
        return evaluate(t, state, me, opp, w)

    my_moves = legal_moves(t, state.bodies[me])
    opp_moves = legal_moves(t, state.bodies[opp])
    if len(my_moves) > 1:
        my_moves = _order_my_moves(t, state, me, opp, my_moves, w)

    best = -INF
    for md, mt in my_moves:
        worst = INF
        for od, ot in opp_moves:
            targets = [0, 0]
            targets[me] = mt
            targets[opp] = ot
            new_state, alive = step(t, state, targets)
            if not alive[me]:
                if not alive[opp]:
                    val = 0.0  # both die: draw-ish
                else:
                    val = -(MATE - (10 - depth))
            elif not alive[opp]:
                val = MATE - (10 - depth)
            else:
                val = _search(t, new_state, me, opp, depth - 1, alpha, INF, clock, w)
            if val < worst:
                worst = val
            if worst <= alpha:  # opponent can force <= alpha here
                break
        if worst > best:
            best = worst
        if best >= beta:
            return best
        if best > alpha:
            alpha = best
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
                worst = INF
                for od, ot in legal_moves(t, state.bodies[opp]):
                    targets = [0, 0]
                    targets[me] = mt
                    targets[opp] = ot
                    new_state, alive = step(t, state, targets)
                    if not alive[me]:
                        if not alive[opp]:
                            val = 0.0
                        else:
                            val = -(MATE - (10 - depth))
                    elif not alive[opp]:
                        val = MATE - (10 - depth)
                    else:
                        val = _search(t, new_state, me, opp, depth - 1, alpha, INF, clock, w)
                    if val < worst:
                        worst = val
                    if worst <= alpha:
                        break
                if cur_best_d is None or worst > alpha:
                    # worst > alpha means strictly better than current alpha
                    if cur_best_d is None or worst > alpha:
                        alpha = worst
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
