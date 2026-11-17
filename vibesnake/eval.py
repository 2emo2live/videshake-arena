"""Evaluation heuristics. All heuristics are original; weights are tunable via hill-climbing."""

from .board import Tables
from .state import State

BIG = 1_000_000
MATE = 100_000


class Weights(dict):
    """Weights with sane defaults; keys documented below."""
    DEFAULTS = {
        # area control (dual flood fill), per cell of relative space
        "area": 24.0,
        # length difference
        "len": 30.0,
        # health difference (scaled /10)
        "health": 6.0,
        # food race: bonus per food we reach strictly earlier
        "food_race": 26.0,
        # nearest-food distance penalty (mild, always on)
        "food_dist": 7.0,
        # extra nearest-food urgency when health is low (< low_health)
        "low_health_food": 60.0,
        "low_health": 30,
        # my head adjacent to enemy head of >= length: danger
        "head_danger": 500.0,
        # my head adjacent to shorter enemy head: potential kill
        "head_kill": 120.0,
        # per extra legal escape cell for me (mobility)
        "mobility": 10.0,
        # penalty for enemy mobility (same scale)
        "mobility_opp": 6.0,
        # small preference to stay away from walls early (center bias)
        "wall": 2.0,
        # dying penalty shaping handled in search
    }


DEFAULT_WEIGHTS = Weights(Weights.DEFAULTS)


def manhattan(t: Tables, a: int, b: int) -> int:
    ax, ay = a % t.w, a // t.w
    bx, by = b % t.w, b // t.w
    return abs(ax - bx) + abs(ay - by)


def dual_flood(t: Tables, h1: int, h2: int, free: int):
    """BFS both heads simultaneously over `free` cells; claim cell for first reacher.

    Returns (area1, area2).
    """
    f1 = 1 << h1
    f2 = 1 << h2
    seen1 = f1
    seen2 = f2
    spread = t.spread
    while f1 or f2:
        if f1:
            f1 = spread(f1) & free & ~(seen1 | seen2)
            seen1 |= f1
        if f2:
            f2 = spread(f2) & free & ~(seen1 | seen2)
            seen2 |= f2
    return seen1.bit_count(), seen2.bit_count()


def manhattan(t: Tables, a: int, b: int) -> int:
    ax, ay = a % t.w, a // t.w
    bx, by = b % t.w, b // t.w
    return abs(ax - bx) + abs(ay - by)


def _food_waves(t: Tables, head: int, blocked: int, food: int):
    """Wave-based bitboard BFS from head; returns {food_cell: distance}."""
    out = {}
    frontier = 1 << head
    visited = frontier
    spread = t.spread
    wave = 0
    remaining = food & ~blocked
    while frontier and remaining:
        wave += 1
        frontier = spread(frontier) & ~visited & ~blocked
        if not frontier:
            break
        visited |= frontier
        hit = frontier & remaining
        if hit:
            remaining &= ~hit
            while hit:
                c = (hit & -hit).bit_length() - 1
                out[c] = wave
                hit &= hit - 1
    return out


def evaluate(t: Tables, state: State, me: int, opp: int, w: Weights) -> float:
    bodies = state.bodies
    mb = bodies[me]
    ob = bodies[opp]
    bits = state.bits
    my_bits = bits[me]
    opp_bits = bits[opp]
    my_len = len(mb)
    opp_len = len(ob)
    mh = mb[0]
    oh = ob[0]

    # free cells: not occupied by either body, but tails vacate soon -> treat as free
    blocked = my_bits | opp_bits
    free = (~blocked) & t.full
    # add both tails as effectively-free (they vacate unless the snake eats; decent approx)
    free |= 1 << mb[-1]
    free |= 1 << ob[-1]

    my_area, opp_area = dual_flood(t, mh, oh, free)

    score = w["area"] * (my_area - opp_area)
    score += w["len"] * (my_len - opp_len)
    score += w["health"] * (state.healths[me] - state.healths[opp]) / 10.0

    # mobility: number of in-bounds neighbors not blocked
    nbr = t.nbr
    m_mob = 0
    o_mob = 0
    solid = my_bits | opp_bits
    for _, c in nbr[mh]:
        if not ((solid >> c) & 1):
            m_mob += 1
    for _, c in nbr[oh]:
        if not ((solid >> c) & 1):
            o_mob += 1
    score += w["mobility"] * m_mob - w["mobility_opp"] * o_mob

    # head-to-head statics
    if manhattan(t, mh, oh) == 1:
        if opp_len >= my_len:
            score -= w["head_danger"] * (1.0 if opp_len > my_len else 0.4)
        else:
            score += w["head_kill"]

    # food: wave-based BFS distances from both heads
    foods = state.food
    my_blocked = (my_bits & ~(1 << mb[-1])) | opp_bits
    opp_blocked = (opp_bits & ~(1 << ob[-1])) | my_bits
    fd_me = _food_waves(t, mh, my_blocked, foods)
    fd_opp = _food_waves(t, oh, opp_blocked, foods)
    f = foods
    my_hunger = state.healths[me]
    opp_hunger = state.healths[opp]
    my_urg = 1.0
    if my_hunger < w["low_health"]:
        my_urg = 3.0
    opp_urg = 1.0
    if opp_hunger < w["low_health"]:
        opp_urg = 3.0
    best_d = None
    while f:
        c = (f & -f).bit_length() - 1
        f &= f - 1
        dme = fd_me.get(c)
        if dme is None:
            continue
        if best_d is None or dme < best_d:
            best_d = dme
        dopp = fd_opp.get(c)
        if dopp is None:
            score += w["food_race"] * my_urg * 0.5
        elif dme < dopp:
            score += w["food_race"] * my_urg * (1.0 if dopp - dme >= 1 else 0.3)
        elif dopp < dme:
            score -= w["food_race"] * opp_urg * 0.7
    if best_d is not None:
        score -= w["food_dist"] * best_d
        if my_hunger < w["low_health"]:
            score -= w["low_health_food"] * best_d

    # mild wall avoidance (prefer central heads)
    mx, my = mh % t.w, mh // t.w
    ox, oy = oh % t.w, oh // t.w
    score -= w["wall"] * (min(mx, t.w - 1 - mx) + min(my, t.h - 1 - my))
    score += w["wall"] * 0.5 * (min(ox, t.w - 1 - ox) + min(oy, t.h - 1 - oy))

    return score
