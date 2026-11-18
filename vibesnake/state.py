"""Game state and turn simulation for standard Battlesnake rules (duels-focused)."""

from .board import Tables


class State:
    __slots__ = ("bodies", "bits", "healths", "food")

    def __init__(self, bodies, bits, healths, food):
        self.bodies = bodies  # tuple of tuples of cell indices, head first
        self.bits = bits      # tuple of int bitboards per snake
        self.healths = healths  # tuple of ints
        self.food = food      # int bitboard

    def copy(self):
        return State(self.bodies, self.bits, self.healths, self.food)


def parse_api_state(data: dict, t: Tables):
    """Build a State from a Battlesnake API /move request.

    Returns (state, my_index, n_snakes).
    """
    board = data["board"]
    w = board.get("width", 11)
    h = board.get("height", 11)
    if w != t.w or h != t.h:
        t = tables_for(w, h) if False else t  # tables are fixed by caller; assume standard
    snakes = board["snakes"]
    my_id = data["you"]["id"]
    my_index = -1
    bodies = []
    bits = []
    healths = []
    for i, s in enumerate(snakes):
        body = tuple(y * t.w + x for x, y in ((seg["x"], seg["y"]) for seg in s["body"]))
        b = 0
        for c in body:
            b |= 1 << c
        bodies.append(body)
        bits.append(b)
        healths.append(int(s.get("health", 100)))
        if s["id"] == my_id:
            my_index = i
    food = 0
    for f in board.get("food", ()):
        food |= 1 << (f["y"] * t.w + f["x"])
    if my_index < 0:
        my_index = 0
    return State(tuple(bodies), tuple(bits), tuple(healths), food), my_index, len(bodies)


def step(t: Tables, state: State, targets):
    """Simultaneous move for all snakes.

    targets: sequence of target cell per snake (must be in-bounds neighbors).
    Returns (new_state, alive_flags as tuple of bool).
    Follows official standard-rules order of operations: tail removal,
    body collisions (except vacating tails), head-to-head, starvation.
    """
    bodies = state.bodies
    bits = state.bits
    healths = state.healths
    foods = state.food
    n = len(bodies)

    new_bodies = [None] * n
    new_bits = [0] * n
    new_healths = [0] * n
    eat = [False] * n
    vacate = 0
    eaten_bits = 0

    for i in range(n):
        b = bodies[i]
        nh = targets[i]
        ate = (foods >> nh) & 1
        eat[i] = ate
        if ate:
            eaten_bits |= 1 << nh
        h = healths[i] - 1
        if ate:
            h = 100
            new_bodies[i] = (nh,) + b
            new_bits[i] = bits[i] | (1 << nh)
        else:
            if len(b) > 1:
                vacate |= 1 << b[-1]
            new_bodies[i] = (nh,) + b[:-1]
            new_bits[i] = (bits[i] & ~(1 << b[-1])) | (1 << nh)
        new_healths[i] = h

    occ = (bits[0] | bits[1]) if n == 2 else _or_all(bits)

    dead = [False] * n
    for i in range(n):
        if new_healths[i] <= 0 and not eat[i]:
            dead[i] = True
            continue
        nh = targets[i]
        if ((occ & ~vacate) >> nh) & 1:
            dead[i] = True

    # head-to-head collisions
    heads = {}
    for i in range(n):
        nh = targets[i]
        g = heads.get(nh)
        if g is None:
            heads[nh] = [i]
        else:
            g.append(i)
    for group in heads.values():
        if len(group) > 1:
            ls = [len(bodies[i]) for i in group]
            m = max(ls)
            if ls.count(m) > 1:
                for i in group:
                    dead[i] = True
            else:
                for i in group:
                    if len(bodies[i]) < m:
                        dead[i] = True

    alive = tuple(not d for d in dead)
    new_food = foods & ~eaten_bits
    new_state = State(tuple(new_bodies), tuple(new_bits), tuple(new_healths), new_food)
    return new_state, alive


def _or_all(bits):
    acc = 0
    for b in bits:
        acc |= b
    return acc


def legal_moves(t: Tables, body):
    """Legal (dir, target) moves for a snake: in-bounds, not into own neck."""
    head = body[0]
    neck = body[1] if len(body) > 1 else -1
    res = []
    for d, c in t.nbr[head]:
        if c != neck:
            res.append((d, c))
    return res


def prune_dead(t: Tables, state: State, alive):
    """Remove dead snakes (keeps their bodies out of the board, like the engine does)."""
    bodies = []
    bits = []
    healths = []
    for i, a in enumerate(alive):
        if a:
            bodies.append(state.bodies[i])
            bits.append(state.bits[i])
            healths.append(state.healths[i])
    if len(bodies) == len(state.bodies):
        return state
    return State(tuple(bodies), tuple(bits), tuple(healths), state.food)


class Sim:
    """Mutable in-place simulation state for the search.

    Same rules as step() but with apply/undo pairs so the alpha-beta search
    avoids allocating a new State per node. Bodies are lists (head first),
    bits/healths are lists, food is an int bitboard.

    Divergence note: a snake that dies this turn does not consume its target
    food (step() removes it). This only affects terminal branches where no
    evaluation happens, so search results are unaffected.
    """

    __slots__ = ("bodies", "bits", "healths", "food", "n")

    def __init__(self, bodies, bits, healths, food):
        self.bodies = [list(b) for b in bodies]
        self.bits = list(bits)
        self.healths = list(healths)
        self.food = food
        self.n = len(self.bodies)

    @classmethod
    def from_state(cls, state: State):
        return cls(state.bodies, state.bits, state.healths, state.food)

    def to_state(self) -> State:
        return State(
            tuple(tuple(b) for b in self.bodies),
            tuple(self.bits),
            tuple(self.healths),
            self.food,
        )

    def apply(self, t: Tables, targets):
        """Simultaneous move; mutates in place. Returns (undo_record, alive)."""
        bodies = self.bodies
        bits = self.bits
        healths = self.healths
        n = self.n
        old_food = self.food
        occ = 0
        for i in range(n):
            occ |= bits[i]

        # phase 1: deaths (pure reads; semantics match step())
        dead = [False] * n
        vacate = 0
        for j in range(n):
            if not ((self.food >> targets[j]) & 1):
                vacate |= 1 << bodies[j][-1]
        occ_free = occ & ~vacate
        for i in range(n):
            nh = targets[i]
            ate = (self.food >> nh) & 1
            if healths[i] <= 1 and not ate:
                dead[i] = True
                continue
            if (occ_free >> nh) & 1:
                dead[i] = True
        heads = {}
        for i in range(n):
            heads.setdefault(targets[i], []).append(i)
        for group in heads.values():
            if len(group) > 1:
                ls = [len(bodies[i]) for i in group]
                m = max(ls)
                if ls.count(m) > 1:
                    for i in group:
                        dead[i] = True
                else:
                    for i in group:
                        if len(bodies[i]) < m:
                            dead[i] = True

        # phase 2: mutate survivors
        rec = []
        for i in range(n):
            if dead[i]:
                continue
            b = bodies[i]
            nh = targets[i]
            ate = (self.food >> nh) & 1
            old_bits = bits[i]
            old_health = healths[i]
            popped = None
            b.insert(0, nh)
            if ate:
                bits[i] = old_bits | (1 << nh)
                healths[i] = 100
                self.food &= ~(1 << nh)
            else:
                popped = b.pop()
                bits[i] = (old_bits & ~(1 << popped)) | (1 << nh)
                healths[i] = old_health - 1
            rec.append((i, old_bits, old_health, popped))
        rec.append(old_food)
        alive = tuple(not d for d in dead)
        return rec, alive

    def undo(self, rec):
        self.food = rec[-1]
        for i, old_bits, old_health, popped in reversed(rec[:-1]):
            b = self.bodies[i]
            b.pop(0)
            if popped is not None:
                b.append(popped)
            self.bits[i] = old_bits
            self.healths[i] = old_health
