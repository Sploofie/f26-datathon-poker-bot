"""
judge_engine.py - referees Short Deck PLO5 bots that run as web servers.

A MATCH between agent1 and agent2 is two HALVES played on the SAME deals:
    Half 1: agent1 sits in seat 0, agent2 in seat 1.
    Half 2: the agents swap seats; the cards stay with the seats.
So each bot ends up holding exactly the cards its opponent held. This
"duplicate" format cancels most of the card luck.

Why the tournament plays each half separately
---------------------------------------------
At a showdown a bot sees its opponent's hole cards. If ONE bot process played
both halves, then in half 2 it could recognise a board from half 1 and know
exactly what its opponent is holding (it held those cards itself).

So the tournament runs every half in its OWN Kubernetes pod, with fresh bot
containers that have no shared storage and no network access. Both pods get
the same secret seed (from new_match_seed()), so they deal the same cards.
Nothing a bot learns in one half can reach the other half.

This file supports both ways of running:

    # One half, exactly what a tournament pod does:
    python judge_engine.py --half 1 --json h1.json
    python judge_engine.py --half 2 --json h2.json
    python judge_engine.py combine h1.json h2.json      # pair them -> verdict

    # Local practice: leave out --half to play both halves back to back
    # against the same two running bots (convenient, not leak-proof).
    python judge_engine.py --deals 200

Verdict: agent1's average result in big blinds per deal, with a 95%
confidence interval. Entirely above 0 -> AGENT1_WIN, entirely below 0 ->
AGENT2_WIN, otherwise CHOP (half a win each). A bot that forfeits either
half loses the match whatever the chips say (BOTH_FORFEIT if both did).

Time: each bot gets TIME_BANK_SECONDS of thinking time per half. The first
NETWORK_GRACE_SECONDS of every request is free, so network overhead is not
charged. A single request may take at most ACTION_TIMEOUT_SECONDS.

Hand results are not sent one request at a time: the judge attaches them to
the bot's next /act request (and sends any leftovers to /hand-end when a
half ends). That cuts the number of requests by about a third.

For the tournament's internal judge: import play_half() and store
HalfResult.to_dict() (in Postgres, for example). Later, load both halves
with HalfResult.from_dict() and call combine_halves() to get the verdict.

Standard library only (the judge does not need FastAPI).
"""

from __future__ import annotations

import argparse
import http.client
import io
import json
import math
import os
import secrets
import socket
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

from evaluator import card_to_str, warmup
from game import (
    ACTION_TIMEOUT_SECONDS,
    AGENT1_DEFAULT_PORT,
    AGENT2_DEFAULT_PORT,
    BIG_BLIND,
    CHOP_Z,
    ERROR_LIMIT,
    NETWORK_GRACE_SECONDS,
    NUM_DEALS,
    STARTUP_TIMEOUT_SECONDS,
    TIME_BANK_SECONDS,
    Action,
    AgentConfig,
    Game,
    HandResult,
    Observation,
    button_for_deal,
    default_action,
    make_deal,
)

MAX_STORED_ERRORS = 10

# Largest reply (status line + headers + body) the judge will read from a bot.
# Real replies are a few KB at most (an action plus any on_hand_end tracebacks).
MAX_RESPONSE_BYTES = 1_000_000

# Verdict strings (also printed on the final "RESULT:" line).
AGENT1_WIN = "AGENT1_WIN"
AGENT2_WIN = "AGENT2_WIN"
CHOP = "CHOP"
BOTH_FORFEIT = "BOTH_FORFEIT"  # both agents forfeited a half: both lose, 0 points each
HALF_DONE = "HALF_DONE"  # last line after a single half (no verdict yet)

# Marks a JSON file as a half result, so `combine` can reject other files.
HALF_RESULT_KIND = "half_result"


def new_match_seed() -> str:
    """A fresh, unguessable match seed for the tournament scheduler.

    Make ONE seed per match and give the same value to both half pods (that
    is what makes them deal the same cards). Never show it to the bots: with
    the seed, a bot could recreate every deal.
    """
    return secrets.token_hex(16)


def agent1_seat_for_half(half: int) -> int:
    """agent1 sits in seat 0 during half 1 and seat 1 during half 2."""
    if half not in (1, 2):
        raise ValueError(f"half must be 1 or 2, got {half!r}")
    return 0 if half == 1 else 1


# ---------------------------------------------------------------------------
# Per-bot statistics
# ---------------------------------------------------------------------------

@dataclass
class AgentStats:
    """Bookkeeping for one bot in one half.

    time_seconds is the thinking time charged to the bot's time bank: the
    wall-clock time of each request minus NETWORK_GRACE_SECONDS.
    wall_seconds is the full wall-clock time, for information only.
    """

    time_seconds: float = 0.0
    wall_seconds: float = 0.0
    calls: int = 0               # requests actually sent to the bot
    skipped_calls: int = 0       # requests skipped because the time bank was empty
    timed_out: bool = False      # True once the time bank ran out
    errors: int = 0
    forfeited: bool = False      # True once errors reached ERROR_LIMIT (or the half failed to start)
    error_messages: list[str] = field(default_factory=list)  # first 10, with tracebacks

    def to_dict(self, include_timing: bool = True) -> dict[str, Any]:
        """JSON-friendly data. include_timing=False drops the time used,
        the only part that differs between identical runs."""
        data: dict[str, Any] = {
            "calls": self.calls,
            "skipped_calls": self.skipped_calls,
            "timed_out": self.timed_out,
            "errors": self.errors,
            "forfeited": self.forfeited,
            "error_messages": list(self.error_messages),
        }
        if include_timing:
            data["time_seconds"] = round(self.time_seconds, 6)
            data["wall_seconds"] = round(self.wall_seconds, 6)
        return data

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentStats":
        """Rebuild stats from to_dict() output (missing timing becomes 0)."""
        return cls(
            time_seconds=float(d.get("time_seconds", 0.0)),
            wall_seconds=float(d.get("wall_seconds", 0.0)),
            calls=int(d.get("calls", 0)),
            skipped_calls=int(d.get("skipped_calls", 0)),
            timed_out=bool(d.get("timed_out", False)),
            errors=int(d.get("errors", 0)),
            forfeited=bool(d.get("forfeited", False)),
            error_messages=[str(m) for m in d.get("error_messages", [])],
        )


def coerce_action(raw: Any) -> Optional[Action]:
    """Turn a bot's answer into an Action, or None if it isn't one.

    Accepts an action name such as "RAISE_POT" or a plain int such as 3.
    Booleans are rejected even though Python treats them as ints.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, str):
        return Action.__members__.get(raw)
    if isinstance(raw, int):
        try:
            return Action(raw)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# HTTP with a hard deadline
# ---------------------------------------------------------------------------
# A plain socket timeout limits each single read, not the whole reply, so a
# bot that sends one byte every 0.9 s could hold the judge for as long as it
# liked. These classes give every request ONE total deadline and cap the
# reply size: once the deadline passes, the next read raises TimeoutError.

class ResponseTooLarge(ValueError):
    """The bot's reply was longer than MAX_RESPONSE_BYTES."""


class _DeadlineReader(io.RawIOBase):
    """Reads from a socket, never past a deadline or past max_bytes."""

    def __init__(self, sock: socket.socket, deadline: float, max_bytes: int) -> None:
        super().__init__()
        self._sock = sock
        self._deadline = deadline
        self._left = max_bytes

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("the reply did not finish before the deadline")
        if self._left <= 0:
            raise ResponseTooLarge(f"reply is longer than {MAX_RESPONSE_BYTES} bytes")
        self._sock.settimeout(remaining)
        view = memoryview(buffer)[:self._left]
        n = self._sock.recv_into(view)
        self._left -= n
        return n


class _DeadlineSocket:
    """Stands in for the socket when http.client builds a response reader."""

    def __init__(self, sock: socket.socket, deadline: float, max_bytes: int) -> None:
        self._reader = _DeadlineReader(sock, deadline, max_bytes)

    def makefile(self, mode: str, *args: Any, **kwargs: Any) -> io.BufferedReader:
        return io.BufferedReader(self._reader)


class _DeadlineHTTPConnection(http.client.HTTPConnection):
    """An HTTPConnection whose replies must arrive in full before `deadline`
    (a time.monotonic() value set before each request)."""

    deadline: float = math.inf

    def response_class(self, sock: socket.socket, *args: Any, **kwargs: Any) -> http.client.HTTPResponse:
        return http.client.HTTPResponse(_DeadlineSocket(sock, self.deadline, MAX_RESPONSE_BYTES),
                                        *args, **kwargs)


# ---------------------------------------------------------------------------
# Talking to one bot
# ---------------------------------------------------------------------------

class HttpAgentHandle:
    """Everything the judge needs to talk to one bot's web server.

    It sends the requests, times them, turns every problem into the default
    action (check if free, otherwise fold), and keeps AgentStats.

    Timing: each half the bot has config.time_bank_seconds of thinking time.
    Every request is timed, and all but the first NETWORK_GRACE_SECONDS is
    charged. One request may take at most ACTION_TIMEOUT_SECONDS. If the bank
    is empty, the bot is skipped and the default action is played.

    Finished hands are queued and sent along with the bot's next /act
    request, or to /hand-end at the end of the half.

    The judge cannot stop a computation already running inside the bot. The
    bot handles one request at a time, so after a timeout later requests may
    queue behind the slow one and time out too.
    """

    def __init__(self, base_url: str, config: AgentConfig, token: Optional[str] = None) -> None:
        """base_url looks like http://localhost:5008."""
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError(f"bot URL must look like http://host:port, got {base_url!r}")
        self.base_url = base_url
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.config = config
        self.token = token
        self.stats = AgentStats()
        self._conn: Optional[_DeadlineHTTPConnection] = None
        self._info: dict[str, Any] = {}
        self._pending_results: list[dict[str, Any]] = []  # finished hands not yet sent

    # ----- low-level HTTP ---------------------------------------------------

    def _close(self) -> None:
        """Drop the connection (a fresh one is opened on the next request)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _request(self, method: str, path: str, body: Optional[dict[str, Any]],
                 timeout: float) -> tuple[int, Any]:
        """Send one request on a reused (keep-alive) connection.

        The whole exchange (connecting, sending, and reading the full reply)
        must finish within `timeout` seconds, and the reply may be at most
        MAX_RESPONSE_BYTES long.

        Returns (status code, parsed JSON). Raises OSError / TimeoutError /
        ValueError on network or format problems.
        """
        if self._conn is None:
            self._conn = _DeadlineHTTPConnection(self.host, self.port, timeout=timeout)
        else:
            self._conn.timeout = timeout
            if self._conn.sock is not None:
                self._conn.sock.settimeout(timeout)
        self._conn.deadline = time.monotonic() + timeout
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Agent-Token"] = self.token
        payload = None if body is None else json.dumps(body)
        try:
            self._conn.request(method, path, body=payload, headers=headers)
            response = self._conn.getresponse()
            raw = response.read()
        except Exception:
            self._close()  # the connection is in an unknown state now
            raise
        try:
            data = json.loads(raw) if raw else None
        except ValueError as exc:
            raise ValueError(f"response was not JSON: {raw[:200]!r}") from exc
        return response.status, data

    def _record_error(self, message: str) -> None:
        """Count one error and forfeit once the limit is reached."""
        self.stats.errors += 1
        if len(self.stats.error_messages) < MAX_STORED_ERRORS:
            self.stats.error_messages.append(message)
        if self.stats.errors >= ERROR_LIMIT:
            self.stats.forfeited = True

    def _remaining_bank(self) -> float:
        """Seconds left in this half's time bank."""
        return self.config.time_bank_seconds - self.stats.time_seconds

    def _usable(self) -> bool:
        """True if the bot may be called right now."""
        if self.stats.forfeited:
            return False
        if self._remaining_bank() <= 0:
            self.stats.timed_out = True
            self.stats.skipped_calls += 1
            return False
        return True

    def _post(self, label: str, path: str, body: dict[str, Any]) -> Optional[dict[str, Any]]:
        """POST body to path and charge the time. Returns the JSON reply, or
        None after recording an error."""
        # Never wait longer than the per-request limit, or than the bank allows.
        bank_left = self._remaining_bank() + NETWORK_GRACE_SECONDS
        limited_by_bank = bank_left < ACTION_TIMEOUT_SECONDS
        timeout = max(0.001, min(ACTION_TIMEOUT_SECONDS, bank_left))
        start = time.perf_counter()
        try:
            status, data = self._request("POST", path, body, timeout)
        except (socket.timeout, TimeoutError):
            if limited_by_bank:
                # The time bank ran out during this request. That is not an
                # error: the bot just plays check-or-fold from now on.
                self.stats.timed_out = True
            else:
                self._record_error(f"{label} took longer than {ACTION_TIMEOUT_SECONDS:g}s")
            return None
        except (OSError, http.client.HTTPException) as exc:
            self._record_error(f"{label} connection error: {type(exc).__name__}: {exc}")
            return None
        except ValueError as exc:
            self._record_error(f"{label} malformed response: {exc}")
            return None
        finally:
            elapsed = time.perf_counter() - start
            self.stats.wall_seconds += elapsed
            self.stats.time_seconds += max(0.0, elapsed - NETWORK_GRACE_SECONDS)  # network time is free
            self.stats.calls += 1

        # Crashes inside on_hand_end() come back in a list; each one is an error.
        if isinstance(data, dict) and isinstance(data.get("hand_end_errors"), list):
            for info in data["hand_end_errors"]:
                if isinstance(info, dict):
                    self._record_error(f"on_hand_end raised {info.get('error')}\n{info.get('traceback', '')}")

        if status == 200 and isinstance(data, dict):
            return data
        if status == 500 and isinstance(data, dict) and "error" in data:
            # The bot's own code crashed: keep its traceback for the report.
            self._record_error(f"{label} raised {data['error']}\n{data.get('traceback', '')}")
        else:
            self._record_error(f"{label} returned HTTP {status}: {str(data)[:300]}")
        return None

    # ----- what the match loop calls ----------------------------------------

    def wait_healthy(self, timeout: float) -> bool:
        """Poll GET / until it answers 200 or `timeout` seconds pass.
        Remembers the bot's participant / agent_name for the report."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                status, data = self._request("GET", "/", None, timeout=2.0)
                if status == 200:
                    self._info = data if isinstance(data, dict) else {}
                    # Close this connection so the first real request opens a
                    # fresh one. Why: the judge may now wait up to
                    # STARTUP_TIMEOUT_SECONDS for the OTHER bot, and this bot's
                    # server closes idle connections after a few seconds
                    # (uvicorn: 5 s). Reusing that dead connection made
                    # /new-half fail, so a healthy bot forfeited the half.
                    self._close()
                    return True
            except Exception:
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.25)

    def describe(self) -> str:
        """'Participant / AgentName (url)' once the bot has answered GET /."""
        if self._info:
            return (f"{self._info.get('participant', '?')} / "
                    f"{self._info.get('agent_name', '?')} ({self.base_url})")
        return self.base_url

    def start_half(self) -> None:
        """Reset stats and tell the bot to start fresh. A failure forfeits the half.

        The bot is told only the match settings: not which half this is and
        not its seat. (Its position in each hand is in obs.is_button.)
        """
        self.stats = AgentStats()  # a NEW object, so the last half's stats stay intact
        self._pending_results = []
        reply = self._post("new_half", "/new-half", {"config": self.config.to_dict()})
        if reply is None:
            self.stats.forfeited = True

    def act(self, obs: Observation) -> Action:
        """Ask the bot for an action. Always returns a LEGAL action."""
        fallback = default_action(obs.to_call)
        if not self._usable():
            return fallback
        # Deliver every finished hand first, so on_hand_end runs before this decision.
        results, self._pending_results = self._pending_results, []
        reply = self._post("choose_action", "/act", {"observation": obs.to_dict(), "hand_results": results})
        if reply is None:
            return fallback
        if "action" not in reply:
            self._record_error(f"act reply has no 'action' field: {str(reply)[:200]}")
            return fallback
        action = coerce_action(reply["action"])
        if action is None:
            self._record_error(f"choose_action returned {reply['action']!r}, which is not an Action")
            return fallback
        if action not in obs.legal_actions:
            legal = ", ".join(a.name for a in obs.legal_actions)
            self._record_error(f"choose_action returned illegal action {action.name} (legal: {legal})")
            return fallback
        return action

    def hand_end(self, result: HandResult) -> None:
        """Queue a finished hand. It is sent with the bot's next /act request."""
        if not self.stats.forfeited:
            self._pending_results.append(result.to_dict())

    def flush_hand_results(self) -> None:
        """Send any queued hands to /hand-end (used at the end of a half).
        Errors count but never change a hand."""
        results, self._pending_results = self._pending_results, []
        if results and self._usable():
            self._post("on_hand_end", "/hand-end", {"results": results})


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class HalfResult:
    """One half of a match, recorded from the AGENTS' point of view.

    This is the unit of work in the tournament: one pod plays one half and
    stores this object (via to_dict()). Nets are stored per agent, not per
    seat, so pairing the two halves later is simple:
        agent1's result on deal d = half1.nets_agent1[d] + half2.nets_agent1[d]

    nets_agent1[d] is agent1's chips won (+) or lost (-) on deal d.
    nets_agent2[d] is always -nets_agent1[d] (poker here is zero-sum).
    """

    half: int                      # 1 or 2
    match_seed: str
    num_deals: int
    agent1_seat: int               # 0 in half 1, 1 in half 2
    nets_agent1: list[int]
    nets_agent2: list[int]
    stats_agent1: AgentStats
    stats_agent2: AgentStats
    agent1_name: str = "agent1"
    agent2_name: str = "agent2"
    elapsed_seconds: float = 0.0
    logs: Optional[list[dict[str, Any]]] = None

    def to_dict(self, include_timing: bool = True, include_logs: bool = True) -> dict[str, Any]:
        """JSON-friendly data. include_timing=False gives identical output
        for identical halves."""
        data: dict[str, Any] = {
            "kind": HALF_RESULT_KIND,
            "half": self.half,
            "match_seed": self.match_seed,
            "num_deals": self.num_deals,
            "agent1_seat": self.agent1_seat,
            "agent1_name": self.agent1_name,
            "agent2_name": self.agent2_name,
            "total_chips": {"agent1": sum(self.nets_agent1), "agent2": sum(self.nets_agent2)},
            "agent_stats": {
                "agent1": self.stats_agent1.to_dict(include_timing),
                "agent2": self.stats_agent2.to_dict(include_timing),
            },
            "nets_agent1": list(self.nets_agent1),
            "nets_agent2": list(self.nets_agent2),
        }
        if include_timing:
            data["elapsed_seconds"] = round(self.elapsed_seconds, 3)
        if include_logs and self.logs is not None:
            data["logs"] = self.logs
        return data

    def to_json(self, include_timing: bool = True, include_logs: bool = True,
                indent: Optional[int] = None) -> str:
        """The half result as a JSON string."""
        return json.dumps(self.to_dict(include_timing, include_logs), indent=indent)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HalfResult":
        """Rebuild a HalfResult from to_dict() output (e.g. a Postgres row or
        a --json file). Raises ValueError if it doesn't look like one."""
        if d.get("kind") != HALF_RESULT_KIND:
            raise ValueError("not a half result (did you pass a full-match JSON?)")
        nets1 = [int(x) for x in d["nets_agent1"]]
        nets2 = [int(x) for x in d["nets_agent2"]]
        num_deals = int(d["num_deals"])
        if len(nets1) != num_deals or len(nets2) != num_deals:
            raise ValueError("number of nets does not match num_deals")
        stats = d.get("agent_stats", {})
        return cls(
            half=int(d["half"]),
            match_seed=str(d["match_seed"]),
            num_deals=num_deals,
            agent1_seat=int(d["agent1_seat"]),
            nets_agent1=nets1,
            nets_agent2=nets2,
            stats_agent1=AgentStats.from_dict(stats.get("agent1", {})),
            stats_agent2=AgentStats.from_dict(stats.get("agent2", {})),
            agent1_name=str(d.get("agent1_name", "agent1")),
            agent2_name=str(d.get("agent2_name", "agent2")),
            elapsed_seconds=float(d.get("elapsed_seconds", 0.0)),
            logs=d.get("logs"),
        )


@dataclass
class MatchResult:
    """A finished duplicate match (two halves), scored from agent1's view.

    paired_bb[d] is agent1's result on deal d over both halves, in big blinds
    (agent2's is the negative). verdict is AGENT1_WIN, AGENT2_WIN or CHOP.
    """

    match_seed: str
    num_deals: int
    paired_bb: list[float]
    mean_bb: float
    std_bb: float
    stderr_bb: float
    ci: tuple[float, float]
    verdict: str
    points: dict[str, float]
    reason: str
    half_totals: dict[str, dict[str, int]]
    agent_stats: dict[str, list[AgentStats]]
    agent1_name: str = "agent1"
    agent2_name: str = "agent2"
    elapsed_seconds: float = 0.0
    logs: Optional[dict[str, list[dict[str, Any]]]] = None

    def to_dict(self, include_timing: bool = True, include_logs: bool = True) -> dict[str, Any]:
        """JSON-friendly data. include_timing=False gives identical output
        for identical matches."""
        data: dict[str, Any] = {
            "kind": "match_result",
            "match_seed": self.match_seed,
            "num_deals": self.num_deals,
            "agent1_name": self.agent1_name,
            "agent2_name": self.agent2_name,
            "verdict": self.verdict,
            "points": self.points,
            "reason": self.reason,
            "mean_bb_per_deal": self.mean_bb,
            "std_bb": self.std_bb,
            "stderr_bb": self.stderr_bb,
            "ci": list(self.ci),
            "chop_z": CHOP_Z,
            "half_totals_chips": self.half_totals,
            "agent_stats": {
                name: [s.to_dict(include_timing) for s in halves]
                for name, halves in self.agent_stats.items()
            },
            "paired_bb": self.paired_bb,
        }
        if include_timing:
            data["elapsed_seconds"] = round(self.elapsed_seconds, 3)
        if include_logs and self.logs is not None:
            data["logs"] = self.logs
        return data

    def to_json(self, include_timing: bool = True, include_logs: bool = True,
                indent: Optional[int] = None) -> str:
        """The result as a JSON string."""
        return json.dumps(self.to_dict(include_timing, include_logs), indent=indent)


def _hand_log(half: int, deal_index: int, button: int, game: Game) -> dict[str, Any]:
    """JSON-friendly record of one hand (for replaying it later)."""
    res = game.result()
    return {
        "half": half,
        "deal_index": deal_index,
        "button_seat": button,
        "hole_cards": [[card_to_str(c) for c in hole] for hole in game.deal.hole_cards],
        "board": [card_to_str(c) for c in game.deal.board],
        "board_revealed": [card_to_str(c) for c in res.board_revealed],
        "actions": [r.to_dict() for r in res.history],
        "pot": res.pot,
        "went_to_showdown": res.went_to_showdown,
        "nets": list(res.nets),
    }


# ---------------------------------------------------------------------------
# Playing and scoring
# ---------------------------------------------------------------------------

def play_half(agent1: HttpAgentHandle, agent2: HttpAgentHandle, match_seed: Any, half: int,
              num_deals: int = NUM_DEALS, keep_logs: bool = False) -> HalfResult:
    """Play ONE half: tell both bots to start fresh, then play num_deals hands.

    `half` decides the seats: in half 1 agent1 is seat 0, in half 2 agent1
    is seat 1. Both halves of a match must use the same match_seed and
    num_deals, so they deal the same cards.
    """
    match_seed = str(match_seed)            # store it the same way every time
    a1_seat = agent1_seat_for_half(half)    # also checks that half is 1 or 2
    # bots[seat] is the bot sitting in that seat.
    bots = (agent1, agent2) if a1_seat == 0 else (agent2, agent1)
    for bot in bots:
        bot.start_half()

    nets_by_seat: tuple[list[int], list[int]] = ([], [])
    logs: Optional[list[dict[str, Any]]] = [] if keep_logs else None
    start = time.perf_counter()

    for d in range(num_deals):
        button = button_for_deal(d)
        game = Game(make_deal(match_seed, d), button, hand_index=d)
        while not game.is_terminal:
            seat = game.to_act
            game.apply(bots[seat].act(game.observation(seat)))
        res = game.result()
        nets_by_seat[0].append(res.nets[0])
        nets_by_seat[1].append(res.nets[1])
        for seat in (0, 1):
            bots[seat].hand_end(game.hand_result(seat))
        if logs is not None:
            logs.append(_hand_log(half, d, button, game))

    for bot in bots:
        bot.flush_hand_results()  # deliver the last hands of the half

    return HalfResult(
        half=half,
        match_seed=match_seed,
        num_deals=num_deals,
        agent1_seat=a1_seat,
        nets_agent1=nets_by_seat[a1_seat],
        nets_agent2=nets_by_seat[1 - a1_seat],
        stats_agent1=agent1.stats,
        stats_agent2=agent2.stats,
        agent1_name=agent1.describe(),
        agent2_name=agent2.describe(),
        elapsed_seconds=time.perf_counter() - start,
        logs=logs,
    )


def decide_verdict(ci_low: float, ci_high: float) -> tuple[str, dict[str, float], str]:
    """Return (verdict, points, reason) for a confidence interval on agent1's
    bb/deal. Strictly above 0: agent1 wins. Strictly below 0: agent2 wins.
    Anything else (including touching 0) is a chop."""
    interval = f"interval [{ci_low:+.3f}, {ci_high:+.3f}]"
    if ci_low > 0:
        return AGENT1_WIN, {"agent1": 1.0, "agent2": 0.0}, f"{interval} is entirely above 0"
    if ci_high < 0:
        return AGENT2_WIN, {"agent1": 0.0, "agent2": 1.0}, f"{interval} is entirely below 0"
    return CHOP, {"agent1": 0.5, "agent2": 0.5}, f"{interval} includes 0"


def combine_halves(half1: HalfResult, half2: HalfResult) -> MatchResult:
    """Pair half 1 and half 2 of the same match and score agent1.

    Raises ValueError if the two halves don't belong together (wrong order,
    different seed, or different number of deals).
    """
    if half1.half != 1 or half2.half != 2:
        raise ValueError(f"expected half 1 then half 2, got half {half1.half} then half {half2.half}")
    if half1.match_seed != half2.match_seed:
        raise ValueError("the halves used different seeds, so they did not deal the same cards")
    if half1.num_deals != half2.num_deals:
        raise ValueError("the halves played a different number of deals")
    n = half1.num_deals

    # agent1 held seat 0's cards in half 1 and seat 1's cards in half 2.
    # Adding both results for the same deal cancels most of the card luck.
    paired_chips = [half1.nets_agent1[d] + half2.nets_agent1[d] for d in range(n)]
    paired_bb = [c / BIG_BLIND for c in paired_chips]

    mean = sum(paired_chips) / BIG_BLIND / n if n else 0.0
    std = statistics.stdev(paired_bb) if n > 1 else 0.0  # sample std (ddof = 1)
    stderr = std / math.sqrt(n) if n else 0.0
    ci = (mean - CHOP_Z * stderr, mean + CHOP_Z * stderr)
    verdict, points, reason = decide_verdict(*ci)

    # A forfeit in EITHER half loses the whole match, whatever the chips say
    # (the same rule the tournament applies). The bb numbers stay for reference.
    forfeited_in = {
        name: [h.half for h, s in ((half1, s1), (half2, s2)) if s.forfeited]
        for name, s1, s2 in (("agent1", half1.stats_agent1, half2.stats_agent1),
                             ("agent2", half1.stats_agent2, half2.stats_agent2))
    }
    losers = [name for name, halves in forfeited_in.items() if halves]
    if losers:
        detail = "; ".join(
            f"{name} forfeited " + " and ".join(f"half {h}" for h in forfeited_in[name])
            for name in losers)
        if len(losers) == 2:
            verdict, points = BOTH_FORFEIT, {"agent1": 0.0, "agent2": 0.0}
            reason = f"{detail}, so both lose"
        elif losers == ["agent1"]:
            verdict, points = AGENT2_WIN, {"agent1": 0.0, "agent2": 1.0}
            reason = f"{detail}, so agent1 loses"
        else:
            verdict, points = AGENT1_WIN, {"agent1": 1.0, "agent2": 0.0}
            reason = f"{detail}, so agent2 loses"

    logs = None
    if half1.logs is not None and half2.logs is not None:
        logs = {"half1": half1.logs, "half2": half2.logs}

    return MatchResult(
        match_seed=half1.match_seed,
        num_deals=n,
        paired_bb=paired_bb,
        mean_bb=mean,
        std_bb=std,
        stderr_bb=stderr,
        ci=ci,
        verdict=verdict,
        points=points,
        reason=reason,
        half_totals={
            "half1": {"agent1": sum(half1.nets_agent1), "agent2": sum(half1.nets_agent2)},
            "half2": {"agent1": sum(half2.nets_agent1), "agent2": sum(half2.nets_agent2)},
        },
        agent_stats={
            "agent1": [half1.stats_agent1, half2.stats_agent1],
            "agent2": [half1.stats_agent2, half2.stats_agent2],
        },
        agent1_name=half1.agent1_name,
        agent2_name=half1.agent2_name,
        elapsed_seconds=half1.elapsed_seconds + half2.elapsed_seconds,
        logs=logs,
    )


def play_match(agent1: HttpAgentHandle, agent2: HttpAgentHandle, match_seed: Any,
               num_deals: int = NUM_DEALS, keep_logs: bool = False) -> MatchResult:
    """LOCAL PRACTICE ONLY: play both halves back to back, then score them.

    The same two bot processes play both halves, so a bot could remember
    half 1's cards in half 2. That's fine for testing your own bot, but the
    tournament never does this: it plays each half in a separate pod.
    """
    half1 = play_half(agent1, agent2, match_seed, 1, num_deals, keep_logs)
    half2 = play_half(agent1, agent2, match_seed, 2, num_deals, keep_logs)
    return combine_halves(half1, half2)


# ---------------------------------------------------------------------------
# The readable reports
# ---------------------------------------------------------------------------

def _stats_line(label: str, s: AgentStats) -> str:
    """One line describing a bot's time, calls, errors and penalties."""
    flags = []
    if s.forfeited:
        flags.append("FORFEITED")
    if s.timed_out:
        flags.append(f"TIME BANK EMPTY ({s.skipped_calls} calls skipped)")
    extra = ("  " + ", ".join(flags)) if flags else ""
    return (f"  {label}: thinking {s.time_seconds:.2f}s of {TIME_BANK_SECONDS:g}s "
            f"(wall {s.wall_seconds:.2f}s), calls {s.calls}, errors {s.errors}{extra}")


def _first_error_lines(name: str, stats: list[AgentStats]) -> list[str]:
    """The first error (with traceback) a bot produced, if any."""
    first = next((s.error_messages[0] for s in stats if s.error_messages), None)
    if not first:
        return []
    return [f"First error from {name}:"] + ["  " + line for line in first.rstrip().splitlines()]


def _speed_line(hands: int, seconds: float) -> list[str]:
    """How fast the hands were played (skipped when no timing is known)."""
    if seconds <= 0:
        return []
    return [f"Speed: {hands} hands in {seconds:.2f}s ({hands / seconds:,.0f} hands/s)"]


def format_half_report(result: HalfResult) -> str:
    """The text printed after a single half. The last line is 'RESULT: HALF_DONE'.

    One half alone does not decide anything: its chip count is mostly card
    luck. Combine it with the other half to get a verdict.
    """
    n = result.num_deals
    a1 = sum(result.nets_agent1)
    a2 = sum(result.nets_agent2)
    lines = [
        f"agent1 = {result.agent1_name}  (seat {result.agent1_seat})",
        f"agent2 = {result.agent2_name}  (seat {1 - result.agent1_seat})",
        "",
        f"Half {result.half}: {n} deals, seed {result.match_seed}",
        f"Chips: agent1 {a1:+d} ({a1 / BIG_BLIND / n:+.3f} bb/deal), agent2 {a2:+d}",
        "(One half is mostly card luck; combine both halves for the verdict.)",
        "Faults:",
        _stats_line("agent1", result.stats_agent1),
        _stats_line("agent2", result.stats_agent2),
    ]
    lines += _first_error_lines("agent1", [result.stats_agent1])
    lines += _first_error_lines("agent2", [result.stats_agent2])
    lines += _speed_line(n, result.elapsed_seconds)
    lines.append(f"RESULT: {HALF_DONE}")
    return "\n".join(lines)


def format_report(result: MatchResult) -> str:
    """The text printed after a full match. The last line is always
    'RESULT: AGENT1_WIN', 'RESULT: AGENT2_WIN', 'RESULT: CHOP' or
    'RESULT: BOTH_FORFEIT'."""
    hands = 2 * result.num_deals
    lines = [
        f"agent1 = {result.agent1_name}",
        f"agent2 = {result.agent2_name}",
        "",
        f"Duplicate match: {result.num_deals} deals ({hands} hands), seed {result.match_seed}",
        f"agent1: {result.mean_bb:+.3f} bb/deal  (std {result.std_bb:.2f}, stderr {result.stderr_bb:.3f})",
        f"Confidence interval (z={CHOP_Z}): [{result.ci[0]:+.3f}, {result.ci[1]:+.3f}] bb/deal",
        f"Reason: {result.reason}",
    ]
    for half in ("half1", "half2"):
        t = result.half_totals[half]
        lines.append(f"{half}: agent1 {t['agent1']:+d} chips, agent2 {t['agent2']:+d} chips")

    lines.append("Faults:")
    for name, halves in result.agent_stats.items():
        for i, s in enumerate(halves, start=1):
            lines.append(_stats_line(f"{name} half{i}", s))
    for name, halves in result.agent_stats.items():
        lines += _first_error_lines(name, halves)
    lines += _speed_line(hands, result.elapsed_seconds)

    headline = {
        AGENT1_WIN: f"AGENT1 WINS  ({result.agent1_name})",
        AGENT2_WIN: f"AGENT2 WINS  ({result.agent2_name})",
        CHOP: "CHOP  (too close to call - half a win each)",
        BOTH_FORFEIT: "BOTH FORFEITED  (both agents lose)",
    }[result.verdict]
    bar = "=" * 60
    lines += [
        "",
        bar,
        f"  VERDICT: {headline}",
        f"  Points: agent1 {result.points['agent1']:g}, agent2 {result.points['agent2']:g}",
        bar,
        f"RESULT: {result.verdict}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

EXIT_OK = 0          # the half or match finished (whatever the verdict)
EXIT_BAD_ARGS = 2
EXIT_CANNOT_RUN = 3  # e.g. neither bot could be reached


def bot_url(value: str) -> str:
    """Accept a full URL (http://host:port) or just a port number (5008)."""
    return f"http://localhost:{value}" if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    """Options for playing. Each falls back to an environment variable,
    which is how a tournament pod configures the judge."""
    p = argparse.ArgumentParser(
        description="Play Short Deck PLO5 between two bot servers.",
        epilog="To score two saved halves: python judge_engine.py combine h1.json h2.json",
    )
    p.add_argument("--agent1", default=os.environ.get("AGENT1_URL", str(AGENT1_DEFAULT_PORT)),
                   help=f"agent1's URL or port (env AGENT1_URL, default {AGENT1_DEFAULT_PORT})")
    p.add_argument("--agent2", default=os.environ.get("AGENT2_URL", str(AGENT2_DEFAULT_PORT)),
                   help=f"agent2's URL or port (env AGENT2_URL, default {AGENT2_DEFAULT_PORT})")
    p.add_argument("--half", default=os.environ.get("HALF"),
                   help="play only half 1 or 2, like a tournament pod (env HALF); "
                        "leave it out to play both halves")
    p.add_argument("--seed", default=os.environ.get("MATCH_SEED", "1"),
                   help="match seed (env MATCH_SEED, default 1); same seed = same cards")
    p.add_argument("--deals", type=int, default=NUM_DEALS, help=f"deals per half (default {NUM_DEALS})")
    p.add_argument("--json", metavar="PATH", help="also write the full result as JSON")
    p.add_argument("--logs", action="store_true", help="include every hand in the JSON")
    return p


def build_combine_parser() -> argparse.ArgumentParser:
    """Options for `python judge_engine.py combine h1.json h2.json`."""
    p = argparse.ArgumentParser(prog="judge_engine.py combine",
                                description="Pair two saved halves and print the verdict.")
    p.add_argument("half1", help="JSON written by --half 1 --json PATH")
    p.add_argument("half2", help="JSON written by --half 2 --json PATH")
    p.add_argument("--json", metavar="PATH", help="also write the match result as JSON")
    return p


def _write_json(path: Optional[str], text: str) -> None:
    """Write text to path if a path was given."""
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def main_play(argv: list[str]) -> int:
    """Play one half (--half) or a full local match, and print a report.

    Tokens (only needed if the bots set AGENT_TOKEN) come from the
    AGENT1_TOKEN and AGENT2_TOKEN environment variables.
    """
    args = build_parser().parse_args(argv)  # exits with code 2 on bad options
    if args.deals < 1:
        print("--deals must be at least 1", file=sys.stderr)
        return EXIT_BAD_ARGS
    half: Optional[int] = None
    if args.half not in (None, ""):
        if str(args.half) not in ("1", "2"):
            print(f"--half must be 1 or 2, got {args.half!r}", file=sys.stderr)
            return EXIT_BAD_ARGS
        half = int(args.half)

    config = AgentConfig(num_hands=args.deals)
    try:
        bot1 = HttpAgentHandle(bot_url(args.agent1), config, os.environ.get("AGENT1_TOKEN"))
        bot2 = HttpAgentHandle(bot_url(args.agent2), config, os.environ.get("AGENT2_TOKEN"))
    except ValueError as exc:
        print(f"bad bot address: {exc}", file=sys.stderr)
        return EXIT_BAD_ARGS

    # A bot that never comes up still "plays": it forfeits and its seat
    # plays check-or-fold, so the half can still be scored.
    healthy = []
    for name, bot in (("agent1", bot1), ("agent2", bot2)):
        print(f"waiting for {name} at {bot.base_url} ...", flush=True)
        ok = bot.wait_healthy(STARTUP_TIMEOUT_SECONDS)
        if not ok:
            print(f"WARNING: {name} did not answer; it will forfeit.", flush=True)
        healthy.append(ok)
    if not any(healthy):
        print("neither bot is reachable; nothing was played", file=sys.stderr)
        return EXIT_CANNOT_RUN

    warmup()  # build the hand-ranking table before play starts
    try:
        if half is not None:
            half_result = play_half(bot1, bot2, args.seed, half, args.deals, keep_logs=args.logs)
            _write_json(args.json, half_result.to_json(indent=2))
            report = format_half_report(half_result)
        else:
            match_result = play_match(bot1, bot2, args.seed, args.deals, keep_logs=args.logs)
            _write_json(args.json, match_result.to_json(indent=2))
            report = format_report(match_result)
    except Exception:
        traceback.print_exc()
        return EXIT_CANNOT_RUN

    print(report, flush=True)
    return EXIT_OK


def main_combine(argv: list[str]) -> int:
    """Load two saved halves, score them, and print the match report."""
    args = build_combine_parser().parse_args(argv)
    try:
        halves = []
        for path in (args.half1, args.half2):
            with open(path, encoding="utf-8") as f:
                halves.append(HalfResult.from_dict(json.load(f)))
        result = combine_halves(halves[0], halves[1])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"cannot combine: {exc}", file=sys.stderr)
        return EXIT_BAD_ARGS
    _write_json(args.json, result.to_json(indent=2))
    print(format_report(result), flush=True)
    return EXIT_OK


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point: `combine ...` scores saved halves; anything else plays."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "combine":
        return main_combine(argv[1:])
    return main_play(argv)


if __name__ == "__main__":
    sys.exit(main())
