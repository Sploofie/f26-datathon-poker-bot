"""
game.py - rules and state machine for Heads-Up Short Deck 5-Card PLO.

This module knows how ONE hand is played: blinds, whose turn it is, which
actions are legal, how big each raise is, when a street ends, and who wins.

It does not know about agents, time banks, or matches (see judge_engine.py).

Quick vocabulary (all amounts are integer chips):
    my_bet / opp_bet  chips each player has put in on THIS street
    pot               chips both players have put in during the WHOLE hand
    to_call           opp_bet - my_bet (how much more I must add to match)
    my_stack          chips I still have behind
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, fields
from enum import IntEnum
from typing import Any, Optional

from evaluator import NUM_CARDS, card_to_str, omaha_score

# ===========================================================================
# TUNABLE CONSTANTS
# These values are PROVISIONAL and may be tuned before the tournament.
# Every other module reads them from here, so change them only in this block.
# ===========================================================================
BIG_BLIND = 2
SMALL_BLIND = 1
STARTING_STACK = 80          # 40 big blinds; stacks reset to this every hand
RAISE_CAP_PER_STREET = 4     # max voluntary bets/raises per street (blinds don't count)
NUM_DEALS = 1000             # deals per match (2000 hands in total)
TIME_BANK_SECONDS = 8.0      # seconds of "thinking" per agent per half (network time is free)
NETWORK_GRACE_SECONDS = 0.001  # the first 1 ms of every request is not charged (covers network time)
ACTION_TIMEOUT_SECONDS = 1.0 # hard limit for one HTTP call to an agent
STARTUP_TIMEOUT_SECONDS = 15.0  # how long the judge waits for an agent to answer GET /
ERROR_LIMIT = 5              # errors in one half before an agent forfeits
CHOP_Z = 1.96                # z value for the chop band (1.96 = 95% interval)
AGENT1_DEFAULT_PORT = 5008     # default port for agent1 (and for agent.py)
AGENT2_DEFAULT_PORT = 5009     # default port for agent2
# ===========================================================================

NUM_HOLE_CARDS = 5
NUM_BOARD_CARDS = 5


class Street(IntEnum):
    """The four betting rounds of a hand."""

    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


# How many board cards are face up during each street.
BOARD_CARDS_VISIBLE = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}


class Action(IntEnum):
    """The four actions an agent can choose from."""

    FOLD = 0
    CHECK_CALL = 1
    RAISE_HALF = 2
    RAISE_POT = 3


RAISE_ACTIONS = (Action.RAISE_HALF, Action.RAISE_POT)


class IllegalActionError(Exception):
    """Raised by Game.apply() when the action is not legal right now."""


@dataclass(frozen=True)
class AgentConfig:
    """Match settings the judge sends to a bot at the start of each half.

    num_hands is the number of hands in one half-match (one per deal).
    """

    starting_stack: int = STARTING_STACK
    big_blind: int = BIG_BLIND
    small_blind: int = SMALL_BLIND
    num_hands: int = NUM_DEALS
    time_bank_seconds: float = TIME_BANK_SECONDS
    raise_cap: int = RAISE_CAP_PER_STREET

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentConfig":
        """Rebuild a config from to_dict() output. Unknown keys are ignored."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def button_for_deal(deal_index: int) -> int:
    """Return the button seat for a deal: seat deal_index % 2."""
    return deal_index % 2


def default_action(to_call: int) -> Action:
    """The action played for an agent that errors, forfeits, or runs out of
    time: CHECK_CALL if nothing is owed (a free check), otherwise FOLD."""
    return Action.CHECK_CALL if to_call == 0 else Action.FOLD


# ---------------------------------------------------------------------------
# Dealing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Deal:
    """The cards for one deal.

    hole_cards[0] belongs to seat 0 and hole_cards[1] to seat 1 (each sorted
    ascending). board holds all 5 community cards, even ones not shown yet.
    """

    hole_cards: tuple[tuple[int, ...], tuple[int, ...]]
    board: tuple[int, ...]

    @property
    def flop(self) -> tuple[int, ...]:
        """The first three board cards."""
        return self.board[0:3]

    @property
    def turn(self) -> int:
        """The fourth board card."""
        return self.board[3]

    @property
    def river(self) -> int:
        """The fifth board card."""
        return self.board[4]


def make_deal(match_seed: Any, deal_index: int) -> Deal:
    """Deal 15 cards deterministically from (match_seed, deal_index).

    Seat 0 gets the first 5, seat 1 the next 5, and the board is the last 5.
    The same inputs always give the same Deal on every machine.
    """
    rng = random.Random(f"{match_seed}:{deal_index}")
    cards = rng.sample(range(NUM_CARDS), 15)
    return Deal(
        hole_cards=(tuple(sorted(cards[0:5])), tuple(sorted(cards[5:10]))),
        board=tuple(cards[10:15]),
    )


# ---------------------------------------------------------------------------
# Data passed to agents
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservedAction:
    """One action in the hand's history, seen from one player's side.

    amount is the chips that action added; street_total is that player's
    total bet on the street afterwards. Blinds are not listed as actions.
    """

    street: Street
    is_me: bool
    action: Action
    amount: int
    street_total: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly data."""
        return {
            "street": self.street.name,
            "is_me": self.is_me,
            "action": self.action.name,
            "amount": self.amount,
            "street_total": self.street_total,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ObservedAction":
        """Rebuild an ObservedAction from to_dict() output."""
        return cls(
            street=Street[d["street"]],
            is_me=bool(d["is_me"]),
            action=Action[d["action"]],
            amount=int(d["amount"]),
            street_total=int(d["street_total"]),
        )


@dataclass(frozen=True)
class Observation:
    """Everything one player is allowed to see when it is their turn.

    Tuples are used instead of lists so an agent cannot change the engine.
    See agent.py for a description of every field.
    """

    hand_index: int
    street: Street
    hole_cards: tuple[int, ...]
    board: tuple[int, ...]
    pot: int
    to_call: int
    my_bet: int
    opp_bet: int
    my_stack: int
    opp_stack: int
    is_button: bool
    legal_actions: tuple[Action, ...]
    raise_to: dict[Action, int]
    min_raise_to: int
    raises_this_street: int
    action_history: tuple[ObservedAction, ...]
    big_blind: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly data (enums become their names)."""
        return {
            "hand_index": self.hand_index,
            "street": self.street.name,
            "hole_cards": list(self.hole_cards),
            "board": list(self.board),
            "pot": self.pot,
            "to_call": self.to_call,
            "my_bet": self.my_bet,
            "opp_bet": self.opp_bet,
            "my_stack": self.my_stack,
            "opp_stack": self.opp_stack,
            "is_button": self.is_button,
            "legal_actions": [a.name for a in self.legal_actions],
            "raise_to": {a.name: amt for a, amt in self.raise_to.items()},
            "min_raise_to": self.min_raise_to,
            "raises_this_street": self.raises_this_street,
            "action_history": [a.to_dict() for a in self.action_history],
            "big_blind": self.big_blind,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Observation":
        """Rebuild an Observation from to_dict() output (used by the agent
        server, so bots always receive a real Observation, never a dict)."""
        return cls(
            hand_index=int(d["hand_index"]),
            street=Street[d["street"]],
            hole_cards=tuple(int(c) for c in d["hole_cards"]),
            board=tuple(int(c) for c in d["board"]),
            pot=int(d["pot"]),
            to_call=int(d["to_call"]),
            my_bet=int(d["my_bet"]),
            opp_bet=int(d["opp_bet"]),
            my_stack=int(d["my_stack"]),
            opp_stack=int(d["opp_stack"]),
            is_button=bool(d["is_button"]),
            legal_actions=tuple(Action[name] for name in d["legal_actions"]),
            raise_to={Action[name]: int(amt) for name, amt in d["raise_to"].items()},
            min_raise_to=int(d["min_raise_to"]),
            raises_this_street=int(d["raises_this_street"]),
            action_history=tuple(ObservedAction.from_dict(a) for a in d["action_history"]),
            big_blind=int(d["big_blind"]),
        )


@dataclass(frozen=True)
class HandResult:
    """What a player learns when a hand ends (passed to on_hand_end).

    board holds only the cards that were actually revealed. opp_hole is the
    opponent's 5 cards if there was a showdown, otherwise None.
    winner is 'me', 'opp', or 'split'.
    """

    hand_index: int
    my_net: int
    pot: int
    went_to_showdown: bool
    board: tuple[int, ...]
    opp_hole: Optional[tuple[int, ...]]
    winner: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly data."""
        return {
            "hand_index": self.hand_index,
            "my_net": self.my_net,
            "pot": self.pot,
            "went_to_showdown": self.went_to_showdown,
            "board": list(self.board),
            "opp_hole": None if self.opp_hole is None else list(self.opp_hole),
            "winner": self.winner,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HandResult":
        """Rebuild a HandResult from to_dict() output."""
        opp = d["opp_hole"]
        return cls(
            hand_index=int(d["hand_index"]),
            my_net=int(d["my_net"]),
            pot=int(d["pot"]),
            went_to_showdown=bool(d["went_to_showdown"]),
            board=tuple(int(c) for c in d["board"]),
            opp_hole=None if opp is None else tuple(int(c) for c in opp),
            winner=str(d["winner"]),
        )


@dataclass(frozen=True)
class ActionRecord:
    """One action as the engine stores it (seat-based, not 'me'/'opp')."""

    seat: int
    street: Street
    action: Action
    amount: int
    street_total: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly data."""
        return {
            "seat": self.seat,
            "street": self.street.name,
            "action": self.action.name,
            "amount": self.amount,
            "street_total": self.street_total,
        }


@dataclass(frozen=True)
class GameResult:
    """The outcome of a finished hand, from the engine's (seat) view.

    nets[s] is seat s's chips won minus chips put in; nets always sum to 0.
    winner_seat is None for a split pot.
    """

    nets: tuple[int, int]
    pot: int
    went_to_showdown: bool
    winner_seat: Optional[int]
    board_revealed: tuple[int, ...]
    history: tuple[ActionRecord, ...]


# ---------------------------------------------------------------------------
# The Game state machine
# ---------------------------------------------------------------------------

class Game:
    """One hand of heads-up Short Deck PLO5.

    Typical loop:
        game = Game(deal, button_seat=0)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            game.apply(choose_action(obs))
        result = game.result()
    """

    def __init__(self, deal: Deal, button_seat: int, hand_index: int = 0) -> None:
        """Start a hand: reset stacks, post blinds, button acts first."""
        if button_seat not in (0, 1):
            raise ValueError("button_seat must be 0 or 1")
        self.deal = deal
        self.button_seat = button_seat
        self.hand_index = hand_index

        self.street = Street.PREFLOP
        self.stacks = [STARTING_STACK, STARTING_STACK]  # chips behind
        self.contributed = [0, 0]                       # chips in, whole hand
        self.bets = [0, 0]                              # chips in, this street
        self.acted = [False, False]                     # acted this street?
        self.raises_this_street = 0                     # voluntary raises only
        self.last_increment = BIG_BLIND                 # size of last raise step
        self.history: list[ActionRecord] = []
        self.is_terminal = False
        self._result: Optional[GameResult] = None

        # Button posts the small blind, the other seat posts the big blind.
        self._put_in(button_seat, SMALL_BLIND)
        self._put_in(1 - button_seat, BIG_BLIND)
        self.to_act: Optional[int] = button_seat  # None once the hand is over

    # ----- small helpers ---------------------------------------------------

    @property
    def pot(self) -> int:
        """All chips committed by both players in this hand."""
        return self.contributed[0] + self.contributed[1]

    def _put_in(self, seat: int, amount: int) -> None:
        """Move chips from a seat's stack into its bet."""
        self.stacks[seat] -= amount
        self.bets[seat] += amount
        self.contributed[seat] += amount

    def _to_call(self, seat: int) -> int:
        """How much this seat must add to match the other seat (never < 0)."""
        return max(0, self.bets[1 - seat] - self.bets[seat])

    def _can_raise(self, seat: int) -> bool:
        """True if RAISE_HALF and RAISE_POT are legal for this seat."""
        return (
            self.raises_this_street < RAISE_CAP_PER_STREET
            and self.stacks[seat] > self._to_call(seat)
            and self.stacks[1 - seat] > 0  # opponent is not all-in
        )

    def _min_raise_to(self, seat: int) -> int:
        """Smallest legal raise_to (an all-in may go below it)."""
        return self.bets[1 - seat] + max(self.last_increment, BIG_BLIND)

    def _raise_to(self, seat: int, action: Action) -> int:
        """The total street bet this seat would have after a raise action."""
        my_bet, opp_bet = self.bets[seat], self.bets[1 - seat]
        to_call = self._to_call(seat)
        pot = self.pot
        all_in_to = my_bet + self.stacks[seat]
        pot_raise_to = opp_bet + pot + to_call             # pot-limit maximum
        half_raise_to = opp_bet + (pot + to_call) // 2
        target = half_raise_to if action == Action.RAISE_HALF else pot_raise_to
        return min(max(target, self._min_raise_to(seat)), pot_raise_to, all_in_to)

    def _legal_for(self, seat: int) -> tuple[Action, ...]:
        """Legal actions for a seat, assuming it is that seat's turn."""
        actions = []
        if self._to_call(seat) > 0:
            actions.append(Action.FOLD)
        actions.append(Action.CHECK_CALL)
        if self._can_raise(seat):
            actions.extend(RAISE_ACTIONS)
        return tuple(actions)

    # ----- public API ------------------------------------------------------

    def legal_actions(self) -> tuple[Action, ...]:
        """Legal actions for the player to act (empty once the hand is over)."""
        if self.is_terminal:
            return ()
        return self._legal_for(self.to_act)

    def raise_to(self, action: Action) -> int:
        """raise_to that a raise action would produce for the player to act."""
        return self._raise_to(self.to_act, Action(action))

    def apply(self, action: Action | int) -> None:
        """Play an action for the player to act.

        Raises IllegalActionError if the hand is over or the action is illegal.
        """
        if self.is_terminal:
            raise IllegalActionError("the hand is already over")
        try:
            action = Action(action)
        except ValueError:
            raise IllegalActionError(f"{action!r} is not an Action") from None
        legal = self.legal_actions()
        if action not in legal:
            names = ", ".join(a.name for a in legal)
            raise IllegalActionError(f"{action.name} is illegal; legal: {names}")

        seat = self.to_act
        opp = 1 - seat

        if action == Action.FOLD:
            self._record(seat, action, 0)
            self._finish(winner_seat=opp, showdown=False)
            return

        if action == Action.CHECK_CALL:
            amount = min(self._to_call(seat), self.stacks[seat])
        else:  # a raise
            new_total = self._raise_to(seat, action)
            amount = new_total - self.bets[seat]
            self.last_increment = new_total - self.bets[opp]
            self.raises_this_street += 1
            self.acted[opp] = False  # the opponent must respond to the raise

        self._put_in(seat, amount)
        self.acted[seat] = True
        self._record(seat, action, amount)
        self._advance()

    def observation(self, seat: int) -> Observation:
        """What `seat` can see right now. legal_actions and raise_to are
        empty unless it is this seat's turn."""
        opp = 1 - seat
        my_turn = (not self.is_terminal) and seat == self.to_act
        legal = self._legal_for(seat) if my_turn else ()
        visible = BOARD_CARDS_VISIBLE[self.street]
        return Observation(
            hand_index=self.hand_index,
            street=self.street,
            hole_cards=self.deal.hole_cards[seat],
            board=self.deal.board[:visible],
            pot=self.pot,
            to_call=self._to_call(seat),
            my_bet=self.bets[seat],
            opp_bet=self.bets[opp],
            my_stack=self.stacks[seat],
            opp_stack=self.stacks[opp],
            is_button=(seat == self.button_seat),
            legal_actions=legal,
            raise_to={a: self._raise_to(seat, a) for a in legal if a in RAISE_ACTIONS},
            min_raise_to=self._min_raise_to(seat),
            raises_this_street=self.raises_this_street,
            action_history=tuple(
                ObservedAction(r.street, r.seat == seat, r.action, r.amount, r.street_total)
                for r in self.history
            ),
            big_blind=BIG_BLIND,
        )

    def result(self) -> GameResult:
        """The outcome of the hand. Only valid once is_terminal is True."""
        if self._result is None:
            raise RuntimeError("result() called before the hand is over")
        return self._result

    def hand_result(self, seat: int) -> HandResult:
        """The finished hand from `seat`'s point of view (for on_hand_end)."""
        res = self.result()
        if res.winner_seat is None:
            winner = "split"
        else:
            winner = "me" if res.winner_seat == seat else "opp"
        return HandResult(
            hand_index=self.hand_index,
            my_net=res.nets[seat],
            pot=res.pot,
            went_to_showdown=res.went_to_showdown,
            board=res.board_revealed,
            opp_hole=self.deal.hole_cards[1 - seat] if res.went_to_showdown else None,
            winner=winner,
        )

    def clone(self) -> "Game":
        """A cheap independent copy, e.g. for simulation-based agents.

        The Deal and the history records are immutable, so they are shared;
        only the small mutable lists are copied.
        """
        new = Game.__new__(Game)
        new.__dict__ = self.__dict__.copy()
        new.stacks = self.stacks[:]
        new.contributed = self.contributed[:]
        new.bets = self.bets[:]
        new.acted = self.acted[:]
        new.history = self.history[:]
        return new

    # ----- internal flow ---------------------------------------------------

    def _record(self, seat: int, action: Action, amount: int) -> None:
        """Append an action to the hand history."""
        self.history.append(ActionRecord(seat, self.street, action, amount, self.bets[seat]))

    def _advance(self) -> None:
        """After a non-fold action: end the street or pass the turn."""
        round_over = self.acted[0] and self.acted[1] and self.bets[0] == self.bets[1]
        if not round_over:
            self.to_act = 1 - self.to_act
            return

        if self.street == Street.RIVER:
            self._showdown()
        elif 0 in self.stacks:
            # Someone is all-in and bets match: run out the board, no more betting.
            self.street = Street.RIVER
            self._showdown()
        else:
            # Start the next street. The non-button acts first after the flop.
            self.street = Street(self.street + 1)
            self.bets = [0, 0]
            self.acted = [False, False]
            self.raises_this_street = 0
            self.last_increment = BIG_BLIND
            self.to_act = 1 - self.button_seat

    def _showdown(self) -> None:
        """Compare both hands with the Omaha rule and pay the pot."""
        s0 = omaha_score(self.deal.hole_cards[0], self.deal.board)
        s1 = omaha_score(self.deal.hole_cards[1], self.deal.board)
        if s0 > s1:
            winner = 0
        elif s1 > s0:
            winner = 1
        else:
            winner = None
        self._finish(winner_seat=winner, showdown=True)

    def _finish(self, winner_seat: Optional[int], showdown: bool) -> None:
        """Pay out the pot and freeze the hand."""
        pot = self.pot
        if winner_seat is None:
            received = [pot // 2, pot // 2]  # pot is always even at showdown
        else:
            received = [0, 0]
            received[winner_seat] = pot
        nets = (received[0] - self.contributed[0], received[1] - self.contributed[1])

        board_revealed = self.deal.board if showdown else self.deal.board[:BOARD_CARDS_VISIBLE[self.street]]
        self._result = GameResult(
            nets=nets,
            pot=pot,
            went_to_showdown=showdown,
            winner_seat=winner_seat,
            board_revealed=tuple(board_revealed),
            history=tuple(self.history),
        )
        self.is_terminal = True
        self.to_act = None

    def __repr__(self) -> str:
        """Short readable summary, handy when debugging."""
        board = " ".join(card_to_str(c) for c in self.deal.board[:BOARD_CARDS_VISIBLE[self.street]])
        return (f"Game(street={self.street.name}, to_act={self.to_act}, pot={self.pot}, "
                f"bets={self.bets}, stacks={self.stacks}, board=[{board}])")
