"""
evaluator.py - card helpers and a fast hand evaluator for Short Deck PLO5.

Short Deck uses 36 cards: ranks 6 7 8 9 T J Q K A in four suits.

Card encoding (an int from 0 to 35):
    rank = card >> 2   (0 = six, 1 = seven, ..., 8 = ace)
    suit = card & 3    (0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades)
So card 0 is "6c", card 3 is "6s", and card 35 is "As".

Hand scores are plain ints: a bigger score is a better hand and equal
scores tie. The category sits in the high bits so any hand of a better
category always beats any hand of a worse one.

Speed trick: there are only C(36, 5) = 376,992 possible five-card hands,
so we score every one of them once, store the results in a dict keyed by
a 36-bit "card mask", and afterwards scoring is just a dict lookup.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Card constants and string helpers
# ---------------------------------------------------------------------------

NUM_RANKS = 9
NUM_SUITS = 4
NUM_CARDS = NUM_RANKS * NUM_SUITS  # 36

RANK_CHARS = "6789TJQKA"  # index = rank number
SUIT_CHARS = "cdhs"       # index = suit number

ACE = 8   # rank number of the ace
NINE = 3  # rank number of the nine (top card of the wheel A-6-7-8-9)


def card_rank(card: int) -> int:
    """Return the rank number of a card (0 = six ... 8 = ace)."""
    return card >> 2


def card_suit(card: int) -> int:
    """Return the suit number of a card (0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades)."""
    return card & 3


def card_to_str(card: int) -> str:
    """Turn a card int into a two-character string, e.g. 35 -> 'As' -> 'Ace of Spades'."""
    if not 0 <= card < NUM_CARDS:
        raise ValueError(f"card must be 0..{NUM_CARDS - 1}, got {card}")
    return RANK_CHARS[card >> 2] + SUIT_CHARS[card & 3]


def str_to_card(text: str) -> int:
    """Turn a two-character string into a card int, e.g. 'As' -> 35.

    The rank is case-insensitive ('a' or 'A'); the suit must be c, d, h or s.
    """
    if len(text) != 2:
        raise ValueError(f"card string must be 2 characters, got {text!r}")
    rank = RANK_CHARS.find(text[0].upper())
    suit = SUIT_CHARS.find(text[1].lower())
    if rank < 0 or suit < 0:
        raise ValueError(f"not a short-deck card: {text!r}")
    return (rank << 2) | suit


def cards_to_str(cards: Iterable[int]) -> str:
    """Turn several cards into a space-separated string, e.g. 'Ah Kd 6c'."""
    return " ".join(card_to_str(c) for c in cards)


def cards_from_str(text: str) -> list[int]:
    """Parse cards from a string such as 'Ah Kd 6c' or 'AhKd6c'.

    Spaces and commas are ignored, then the text is read two characters
    at a time.
    """
    cleaned = text.replace(" ", "").replace(",", "")
    if len(cleaned) % 2 != 0:
        raise ValueError(f"odd number of characters in {text!r}")
    return [str_to_card(cleaned[i:i + 2]) for i in range(0, len(cleaned), 2)]


# ---------------------------------------------------------------------------
# Hand categories, ranked (short-deck order, weakest = 0, strongest = 8)
# ---------------------------------------------------------------------------
# Note the two short-deck differences from traditional Texas Hold 'Em:
#   * FLUSH is above FULL_HOUSE (flushes are rarer, as a result of the shortened deck).
#   * THREE_OF_A_KIND is above STRAIGHT.

HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
STRAIGHT = 3
THREE_OF_A_KIND = 4
FULL_HOUSE = 5
FLUSH = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8

CATEGORY_NAMES: tuple[str, ...] = (
    "high card",
    "one pair",
    "two pair",
    "straight",
    "three of a kind",
    "full house",
    "flush",
    "four of a kind",
    "straight flush",
)

# Score layout: category in bits 20 and up, then up to five 4-bit "kicker"
# slots (most important first). Ranks are 0..8, so 4 bits per slot is enough.
_CATEGORY_SHIFT = 20


def category_of(score: int) -> int:
    """Return the category number (0..8) stored inside a score."""
    return score >> _CATEGORY_SHIFT


def category_name(score: int) -> str:
    """Return the category name of a score, e.g. 'flush'."""
    return CATEGORY_NAMES[category_of(score)]


def _pack(category: int, ranks: Sequence[int]) -> int:
    """Combine a category and an ordered list of tie-break ranks into a score.

    ranks[0] is the most important tie-breaker. Unused slots stay 0.
    """
    score = category << _CATEGORY_SHIFT
    for slot, rank in enumerate(ranks):
        score |= rank << (16 - 4 * slot)  # slot 0 -> bits 16-19, slot 4 -> bits 0-3
    return score


def _straight_top(ranks_desc: Sequence[int]) -> int | None:
    """If five DISTINCT ranks (sorted high to low) form a straight, return
    the rank of its top card, otherwise None.

    The wheel A-6-7-8-9 counts as a nine-high straight, so it is the
    lowest straight and loses to 6-7-8-9-T.
    """
    if ranks_desc[0] - ranks_desc[4] == 4:
        return ranks_desc[0]
    if tuple(ranks_desc) == (ACE, NINE, 2, 1, 0):  # A, 9, 8, 7, 6
        return NINE
    return None


def _score_ranks(ranks_desc: tuple[int, ...], is_flush: bool) -> int:
    """Score five cards given their ranks (sorted high to low) and whether
    all five share a suit. This is the 'classify' reference logic."""
    counts = Counter(ranks_desc)

    if len(counts) == 5:
        # Five different ranks: straight flush, flush, straight, or high card.
        top = _straight_top(ranks_desc)
        if top is not None:
            return _pack(STRAIGHT_FLUSH if is_flush else STRAIGHT, [top])
        if is_flush:
            return _pack(FLUSH, ranks_desc)
        return _pack(HIGH_CARD, ranks_desc)

    # Some rank repeats. Order ranks by (how many, rank) so the biggest
    # group comes first, e.g. K K K 7 7 -> [K, 7]; 9 9 6 6 A -> [9, 6, A].
    groups = sorted(counts.items(), key=lambda rc: (rc[1], rc[0]), reverse=True)
    shape = tuple(count for _, count in groups)
    ordered = [rank for rank, _ in groups]

    if shape == (4, 1):
        category = FOUR_OF_A_KIND
    elif shape == (3, 2):
        category = FULL_HOUSE
    elif shape == (3, 1, 1):
        category = THREE_OF_A_KIND
    elif shape == (2, 2, 1):
        category = TWO_PAIR
    else:  # (2, 1, 1, 1)
        category = ONE_PAIR
    return _pack(category, ordered)


def score_five(cards: Sequence[int]) -> int:
    """Score exactly five distinct cards. Higher is better; equal is a tie."""
    if len(cards) != 5 or len(set(cards)) != 5:
        raise ValueError("score_five needs exactly 5 distinct cards")
    ranks = tuple(sorted((c >> 2 for c in cards), reverse=True))
    is_flush = len({c & 3 for c in cards}) == 1
    return _score_ranks(ranks, is_flush)


# ---------------------------------------------------------------------------
# Lookup table (built lazily, cached at module level)
# ---------------------------------------------------------------------------

_TABLE: dict[int, int] | None = None  # 36-bit card mask -> score


def _build_table() -> dict[int, int]:
    """Score all 376,992 five-card hands and return {mask: score}."""
    table: dict[int, int] = {}
    # Many hands share the same ranks + flush flag, so cache those scores.
    by_ranks: dict[tuple[tuple[int, ...], bool], int] = {}
    for a, b, c, d, e in combinations(range(NUM_CARDS), 5):
        # combinations() gives ascending cards, so e has the highest rank.
        ranks = (e >> 2, d >> 2, c >> 2, b >> 2, a >> 2)
        is_flush = (a & 3) == (b & 3) == (c & 3) == (d & 3) == (e & 3)
        key = (ranks, is_flush)
        score = by_ranks.get(key)
        if score is None:
            score = _score_ranks(ranks, is_flush)
            by_ranks[key] = score
        table[(1 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << e)] = score
    return table


def get_table() -> dict[int, int]:
    """Return the {card mask: score} table, building it on first use.

    Treat it as read-only. A card mask has bit c set for each card c.
    """
    global _TABLE
    if _TABLE is None:
        _TABLE = _build_table()
    return _TABLE


def warmup() -> None:
    """Build the lookup table now (takes a few seconds the first time)."""
    get_table()


# The 10 ways to choose 2 of 5 hole cards and 3 of 5 board cards.
_PAIRS = tuple(combinations(range(5), 2))
_TRIPLES = tuple(combinations(range(5), 3))


def omaha_score(hole: Sequence[int], board: Sequence[int]) -> int:
    """Best score using EXACTLY 2 of the 5 hole cards and EXACTLY 3 of the
    5 board cards (10 x 10 = 100 combinations, 100 dict lookups)."""
    if len(hole) != 5 or len(board) != 5:
        raise ValueError("omaha_score needs 5 hole cards and 5 board cards")
    table = get_table()
    hole_bits = [1 << c for c in hole]
    board_bits = [1 << c for c in board]
    hole_masks = [hole_bits[i] | hole_bits[j] for i, j in _PAIRS]
    board_masks = [board_bits[i] | board_bits[j] | board_bits[k] for i, j, k in _TRIPLES]
    best = 0
    for hm in hole_masks:
        for bm in board_masks:
            score = table[hm | bm]
            if score > best:
                best = score
    return best
