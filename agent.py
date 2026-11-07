"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Strategy summary:
  * Monte Carlo equity (vs a random opponent hand) using omaha_score, with a
    strict per-decision time budget so the 8 s half-match time bank is never
    exhausted.
  * Equity is discounted when facing bets (opponent's betting range is
    stronger than random), adjusted by a simple opponent-aggression estimate.
  * Compare equity with pot odds to fold / call / raise (half pot or pot).
  * Everything is wrapped in try/except with a legal-action fallback, because
    5 errors in a half forfeits the match.
"""

import os
import random
import time
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
warmup()

# rename as your team/bot's name
PARTICIPANT = "schlorps"
AGENT_NAME = "schlorps"


# ===== YOUR CODE HERE =====

ALL_CARDS = list(range(36))
RNG = random.Random()

# Time management (seconds). The bank is 8 s per half; we aim to spend only
# TARGET_COMPUTE of it on our own computation, leaving margin for request
# overhead (JSON parsing, HTTP) which is also timed by the judge.
TARGET_COMPUTE = 3.5
MAX_PER_DECISION = 0.006
MIN_PER_DECISION = 0.0004
MAX_ITERS = 400
OPP_HANDS_PER_BOARD = 3          # opponent hands sampled per board run-out


def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands)."""
    memory["hands_played"] = 0
    memory["time_used"] = 0.0
    memory["decisions"] = 0
    memory["opp_actions"] = 0
    memory["opp_raises"] = 0
    memory["hist_hand"] = -1
    memory["hist_seen"] = 0


# ---------- helpers ----------

def _name(a) -> str:
    return str(getattr(a, "name", a)).upper()


def _update_opp_stats(obs: Observation, memory: dict) -> None:
    """Count opponent actions/raises from this hand's action history (once each)."""
    if memory.get("hist_hand") != obs.hand_index:
        memory["hist_hand"] = obs.hand_index
        memory["hist_seen"] = 0
    hist = obs.action_history or []
    seen = memory.get("hist_seen", 0)
    for h in hist[seen:]:
        if not getattr(h, "is_me", False):
            memory["opp_actions"] = memory.get("opp_actions", 0) + 1
            if "RAISE" in _name(getattr(h, "action", "")):
                memory["opp_raises"] = memory.get("opp_raises", 0) + 1
    memory["hist_seen"] = len(hist)


def _aggression(memory: dict) -> float:
    """Smoothed opponent raise frequency (prior ~0.33)."""
    return (memory.get("opp_raises", 0) + 2.0) / (memory.get("opp_actions", 0) + 6.0)


def _budget(obs: Observation, memory: dict) -> float:
    """Seconds we may spend on this decision."""
    used = memory.get("time_used", 0.0)
    remaining_hands = max(1, 1000 - obs.hand_index)
    est_decisions = remaining_hands * 3.0
    b = (TARGET_COMPUTE - used) / est_decisions
    if TARGET_COMPUTE - used <= 0:
        return 0.0
    return max(0.0, min(MAX_PER_DECISION, b))


def _equity(obs: Observation, budget: float) -> float:
    """Monte Carlo equity vs a uniformly random opponent hand, within `budget` seconds."""
    hole = list(obs.hole_cards)
    board = list(obs.board)
    dead = set(hole) | set(board)
    unknown = [c for c in ALL_CARDS if c not in dead]
    need = 5 - len(board)
    k = OPP_HANDS_PER_BOARD
    draw = need + 5 * k
    if draw > len(unknown):
        k = max(1, (len(unknown) - need) // 5)
        draw = need + 5 * k

    t_end = time.perf_counter() + budget
    wins = 0.0
    total = 0
    it = 0
    sample = RNG.sample
    while it < MAX_ITERS:
        it += 1
        cards = sample(unknown, draw)
        full_board = board + cards[:need] if need else board
        mine = omaha_score(hole, full_board)
        pos = need
        for _ in range(k):
            opp = cards[pos:pos + 5]
            pos += 5
            theirs = omaha_score(opp, full_board)
            if mine > theirs:
                wins += 1.0
            elif mine == theirs:
                wins += 0.5
            total += 1
        if (it & 3) == 0 and time.perf_counter() >= t_end:
            break
        if it >= 6 and time.perf_counter() >= t_end + budget:  # hard stop safety
            break
    if total == 0:
        return 0.5
    return wins / total


def _cheap_heuristic(obs: Observation) -> Action:
    """Used when out of time budget or on any failure: very cheap and safe."""
    if obs.to_call == 0:
        return Action.CHECK_CALL
    if obs.to_call <= obs.big_blind:
        return Action.CHECK_CALL
    if obs.pot > 0 and obs.to_call <= 0.2 * obs.pot:
        return Action.CHECK_CALL
    return Action.FOLD


def _legalize(a, obs: Observation) -> Action:
    """Guarantee we return something legal."""
    la = list(obs.legal_actions)
    if a in la:
        # never fold when checking is free
        if a == Action.FOLD and obs.to_call == 0 and Action.CHECK_CALL in la:
            return Action.CHECK_CALL
        return a
    if Action.CHECK_CALL in la:
        return Action.CHECK_CALL
    return la[0]


def _decide(obs: Observation, memory: dict) -> Action:
    la = obs.legal_actions
    can_pot = Action.RAISE_POT in la
    can_half = Action.RAISE_HALF in la
    can_raise = can_pot or can_half

    _update_opp_stats(obs, memory)

    budget = _budget(obs, memory)
    if budget < MIN_PER_DECISION:
        return _cheap_heuristic(obs)

    eq = _equity(obs, budget)
    agg = _aggression(memory)
    to_call = obs.to_call
    pot = obs.pot

    # ----- facing a bet / raise (or preflop blind to complete) -----
    if to_call > 0:
        pot_before = max(1, pot - to_call)
        size = min(1.0, to_call / pot_before)
        discount = 0.03 + 0.12 * size           # opp range is stronger than random
        if agg > 0.5:
            discount *= 0.5                      # very aggressive opp bluffs more
        elif agg < 0.2:
            discount *= 1.3                      # passive opp: bets mean strength
        if obs.street == Street.PREFLOP and to_call <= obs.big_blind:
            discount = 0.0                       # just completing / calling a blind
        e = eq - discount

        need = to_call / float(pot + to_call)    # break-even equity
        if e < need + 0.02:
            return Action.FOLD

        heavy = obs.raises_this_street >= 3
        if can_raise and not heavy:
            if e >= 0.72 and can_pot:
                return Action.RAISE_POT
            if e >= 0.62:
                if can_half:
                    return Action.RAISE_HALF
                return Action.RAISE_POT
        elif can_raise and heavy and e >= 0.82:
            return Action.RAISE_POT if can_pot else Action.RAISE_HALF
        return Action.CHECK_CALL

    # ----- nothing to call: bet or check -----
    e = eq
    if can_raise:
        if e >= 0.66 and can_pot:
            return Action.RAISE_POT
        if e >= 0.55:
            return Action.RAISE_HALF if can_half else Action.RAISE_POT
        # light bluffing/semi-bluffing to stay unexploitable
        if 0.38 <= e < 0.5 and RNG.random() < 0.08:
            return Action.RAISE_HALF if can_half else Action.RAISE_POT
    return Action.CHECK_CALL


def choose_action(obs: Observation, memory: dict) -> Action:
    """Called every time it is your turn. Return one of obs.legal_actions."""
    t0 = time.perf_counter()
    try:
        action = _decide(obs, memory)
    except Exception:
        try:
            action = _cheap_heuristic(obs)
        except Exception:
            action = Action.CHECK_CALL
    try:
        memory["time_used"] = memory.get("time_used", 0.0) + (time.perf_counter() - t0)
        memory["decisions"] = memory.get("decisions", 0) + 1
    except Exception:
        pass
    try:
        return _legalize(action, obs)
    except Exception:
        return Action.CHECK_CALL


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand. Kept trivially safe: an exception
    here counts as an error toward the 5-error forfeit."""
    try:
        memory["hands_played"] = memory.get("hands_played", 0) + 1
    except Exception:
        pass

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
