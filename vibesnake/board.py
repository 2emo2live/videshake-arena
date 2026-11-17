"""Original Battlesnake engine written from scratch (no open-source snake code copied).

Board representation: bitboards on a W x H grid.
Cell index = y * W + x. Bit i of an int corresponds to cell i.
"""

# Battlesnake coords: y=0 is the bottom. Internal dir 0 = y-1 => "down", dir 1 = y+1 => "up".
DIR_NAMES = ("down", "up", "left", "right")
# dir index: 0=down (y-1), 1=up (y+1), 2=left (x-1), 3=right (x+1)


def cell_of(w, x, y):
    return y * w + x


class Tables:
    """Precomputed per-size tables and bitboard helpers."""

    __slots__ = ("w", "h", "ncells", "full", "col0", "collast", "nbr", "dirs_to_cell")

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        n = w * h
        self.ncells = n
        self.full = (1 << n) - 1
        col0 = 0
        collast = 0
        for y in range(h):
            col0 |= 1 << (y * w)
            collast |= 1 << (y * w + w - 1)
        self.col0 = col0
        self.collast = collast

        nbr = []
        for c in range(n):
            x = c % w
            y = c // w
            entry = []
            for d in range(4):
                nx = x + (1 if d == 3 else -1 if d == 2 else 0)
                ny = y + (1 if d == 1 else -1 if d == 0 else 0)
                if 0 <= nx < w and 0 <= ny < h:
                    entry.append((d, ny * w + nx))
            nbr.append(tuple(entry))
        self.nbr = tuple(nbr)

    def spread(self, bits: int) -> int:
        """All cells adjacent (4-dir) to any set bit."""
        w = self.w
        res = ((bits & ~self.col0) >> 1) | ((bits & ~self.collast) << 1)
        res |= bits >> w
        res |= (bits << w) & self.full
        return res & self.full


_TABLE_CACHE: dict = {}


def tables_for(w: int, h: int) -> Tables:
    key = (w, h)
    t = _TABLE_CACHE.get(key)
    if t is None:
        t = Tables(w, h)
        _TABLE_CACHE[key] = t
    return t
