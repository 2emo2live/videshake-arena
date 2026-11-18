"""Evaluation heuristics. All heuristics are original; weights are tunable via hill-climbing."""

from .board import Tables
from .state import State, vacating_tail

BIG = 1_000_000
MATE = 100_000


class Weights(dict):
    """Weights with sane defaults; keys documented below.

    Missing keys fall back to DEFAULTS, so old weight files stay loadable
    after new keys are introduced.
    """
    DEFAULTS = {
        # area control (dual flood fill), per cell of relative space
        "area": 24.0,
        # saturating area advantage: beyond area_cap cells of relative space the
        # marginal value drops to area * area_tail. Flood fill over free space
        # overvalues transient enclosures (bodies move every turn), so huge
        # "we boxed them in" advantages must not dominate the evaluation.
        "area_cap": 25.0,
        "area_tail": 0.25,
        # area control only matters once space gets tight: scale the area term
        # by min(1, (my_len + opp_len) / area_phase). Early on both snakes are
        # tiny and the dual flood fill is dominated by transient partitions
        # around the heads (it swings hundreds of points per turn), which used
        # to drown out the food terms and made the snake starve at length 3.
        "area_phase": 24.0,
        # own-space safety: relative area alone happily drives us into a
        # pocket that we seal with our own body, because inside the pocket we
        # still "out-area" the opponent. Compare our reachable space with what
        # a snake of our length needs to keep moving, and penalize the deficit.
        "space_req": 1.5,
        "space_penalty": 20.0,
        # length difference
        "len": 30.0,
        # health difference (scaled /10)
        "health": 6.0,
        # food race: value of winning a food. Decays with path distance:
        # value = food_race / (1 + food_decay * d). food_decay=0 makes it flat.
        "food_race": 100.0,
        "food_decay": 1.0,
        # food the opponent reaches first still pulls us, at this fraction
        "food_lost": 0.5,
        # nearest-food distance penalty (mild, always on)
        "food_dist": 7.0,
        # extra nearest-food urgency when health is low (< low_health)
        "low_health_food": 60.0,
        "low_health": 30,
        # hunger urgency scaling: urgency = 1 + hunger_k * (low_health - hp) / low_health
        "hunger_k": 2.0,
        # my head adjacent to enemy head of >= length: danger
        "head_danger": 500.0,
        # my head adjacent to shorter enemy head: potential kill
        "head_kill": 120.0,
        # contempt for draws: mutual destruction is almost as worthless as
        # losing (a draw does not beat the reference snakes and wastes winning
        # chances). Set just above -MATE so a forced draw still beats forced
        # death, but any playable continuation is preferred over trading.
        "draw_value": -99980.0,
        # per extra legal escape cell for me (mobility)
        "mobility": 10.0,
        # penalty for enemy mobility (same scale)
        "mobility_opp": 6.0,
        # root food pull (tie-breaker added to the searched root value):
        # bonus for landing on food, and penalty per step of path distance
        # from the moved-to cell to the nearest reachable food.
        "root_food": 60.0,
        "root_food_dist": 15.0,
        # food pull grows as health falls: urg = 1 + root_hunger*(100-hp)/100
        "root_hunger": 1.0,
        # small preference to stay away from walls early (center bias)
        "wall": 2.0,
        # dying penalty shaping handled in search
    }

    def __missing__(self, key):
        try:
            return Weights.DEFAULTS[key]
        except KeyError:
            raise KeyError(key)


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


def _reach_area(t: Tables, head: int, free: int) -> int:
    """Cells reachable from `head` through `free` (head itself counts)."""
    frontier = 1 << head
    seen = frontier
    spread = t.spread
    while frontier:
        frontier = spread(frontier) & free & ~seen
        seen |= frontier
    return seen.bit_count()


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

    # free cells: not occupied by either body, but a truly vacating tail can
    # be entered this turn (growth duplicates stay occupied -> mask = 0)
    blocked = my_bits | opp_bits
    my_tail = vacating_tail(mb)
    opp_tail = vacating_tail(ob)
    free = ((~blocked) & t.full) | my_tail | opp_tail

    my_area, opp_area = dual_flood(t, mh, oh, free)

    diff = my_area - opp_area
    cap = w["area_cap"]
    if diff > cap:
        diff = cap + (diff - cap) * w["area_tail"]
    elif diff < -cap:
        diff = -cap - (-diff - cap) * w["area_tail"]
    ap = w["area_phase"]
    phase = 1.0 if ap <= 0 else (my_len + opp_len) / ap
    if phase > 1.0:
        phase = 1.0
    elif phase < 0.0:
        phase = 0.0
    score = w["area"] * phase * diff
    score += w["len"] * (my_len - opp_len)
    score += w["health"] * (state.healths[me] - state.healths[opp]) / 10.0

    # own-space deficit: relative area can look great inside a pocket that our
    # own body is sealing; a snake needs roughly space_req cells per segment.
    # NOTE: dual-flood area is contested and collapses when the opponent is
    # near, so measure what we can actually reach instead.
    space = _reach_area(t, mh, free)
    need = my_len * w["space_req"]
    if space < need:
        score -= w["space_penalty"] * (need - space)

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

    # food: distance-discounted race values from wave-based BFS distances
    foods = state.food
    my_blocked = (my_bits & ~my_tail) | opp_bits
    opp_blocked = (opp_bits & ~opp_tail) | my_bits
    fd_me = _food_waves(t, mh, my_blocked, foods)
    fd_opp = _food_waves(t, oh, opp_blocked, foods)
    f = foods
    my_hunger = state.healths[me]
    # hunger urgency: the closer to starvation, the more each food is worth
    low = w["low_health"]
    my_urg = 1.0
    if my_hunger < low:
        my_urg = 1.0 + w["hunger_k"] * (low - my_hunger) / low
    vr = w["food_race"]
    decay = w["food_decay"]
    lost = w["food_lost"]
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
        # every reachable food pulls; losing the race only discounts it, so
        # the pull never flips on and off as the opponent moves and our snake
        # stops oscillating instead of committing to a food
        gain = 1.0 if (dopp is None or dme <= dopp) else lost
        # a tie race is worth nothing: arriving the same turn as an
        # equal-length opponent often means a mutual head-on death
        if dme == dopp:
            gain = 0.0
        score += vr * my_urg * gain / (1.0 + decay * dme)
    if best_d is not None:
        score -= w["food_dist"] * best_d
        if my_hunger < low:
            # starvation survival: distance to reachable food is critical
            score -= w["low_health_food"] * best_d

    # mild wall avoidance (prefer central heads)
    mx, my = mh % t.w, mh // t.w
    ox, oy = oh % t.w, oh // t.w
    score -= w["wall"] * (min(mx, t.w - 1 - mx) + min(my, t.h - 1 - my))
    score += w["wall"] * 0.5 * (min(ox, t.w - 1 - ox) + min(oy, t.h - 1 - oy))

    return score
