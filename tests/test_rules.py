"""Rule-level regressions: growth-duplicated tails and step/Sim equivalence.

Run from the repo root:  uv run python tests/test_rules.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vibesnake.board import tables_for
from vibesnake.state import State, Sim, legal_moves, step, vacating_tail

T = tables_for(11, 11)


def bits_of(body):
    b = 0
    for c in body:
        b |= 1 << c
    return b


def make(bodies, food=0):
    return State(
        tuple(tuple(b) for b in bodies),
        tuple(bits_of(b) for b in bodies),
        tuple(100 for _ in bodies),
        food,
    )


def test_duplicated_tail_is_occupied():
    # grew last turn: (2, 0) appears twice, so it does not vacate
    me = (1, 2, 2)
    opp = (99, 98, 97)
    assert vacating_tail(me) == 0
    assert vacating_tail((1, 2, 3)) == 1 << 3
    st = make([me, opp])
    _, alive = step(T, st, [2, 110])
    assert alive == (False, True), "moving into a growth-duplicated tail must die"


def test_unique_tail_move_keeps_bitboard_consistent():
    st = make([(1, 2, 3), (99, 98, 97)])
    # move into our own unique tail (3): legal, and the cell stays occupied
    # because the new head is there
    ns, alive = step(T, st, [3, 110])
    assert alive[0] is True
    assert ns.bits[0] == bits_of(ns.bodies[0])
    assert (ns.bits[0] >> 3) & 1
    # and the next move still keeps the bitboard in sync
    ns2, _ = step(T, ns, [4, 87])
    assert ns2.bits[0] == bits_of(ns2.bodies[0])


def test_growth_duplicate_resolves_after_one_move():
    st = make([(1, 2, 2), (99, 98, 97)])
    ns, alive = step(T, st, [0, 110])
    assert alive[0] is True
    assert ns.bodies[0] == (0, 1, 2)
    assert ns.bits[0] == bits_of(ns.bodies[0])


def test_sim_matches_step_and_undo_restores():
    rng = random.Random(7)
    for _ in range(120):
        bodies = [
            (38, 39, 40),
            (71, 72, 73),
        ]
        if rng.random() < 0.4:
            i = rng.randrange(2)
            bodies[i] = bodies[i] + (bodies[i][-1],)  # growth representation
        s = make(bodies, food=(1 << 55) | (1 << 20))
        sim = Sim.from_state(s)
        base = ([list(b) for b in sim.bodies], list(sim.bits), list(sim.healths), sim.food)
        moves = [rng.choice(legal_moves(T, s.bodies[j]))[1] for j in range(2)]
        ns, alive = step(T, s, moves)
        rec, salive = sim.apply(T, moves)
        assert alive == salive
        assert sim.to_state().bodies == ns.bodies
        assert tuple(sim.bits) == ns.bits
        assert sim.food == ns.food
        for i in range(2):
            assert sim.bits[i] == bits_of(sim.bodies[i])
        sim.undo(rec)
        assert ([list(b) for b in sim.bodies], list(sim.bits), list(sim.healths), sim.food) == base


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("rules tests passed")
