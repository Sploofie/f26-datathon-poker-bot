"""
smoke_test.py - a quick "does this bot actually work?" check.

The build pipeline runs this INSIDE a freshly built agent image, under the
same restrictions as a real tournament match (no network, read-only files,
limited CPU and memory). Participants can run it the same way with
tools/check_build.sh before they submit.

It checks, in order:
  1. GET /          answers within the startup limit (15 s) and names the bot
  2. POST /new-half starts a half without crashing
  3. POST /act      answers each sample situation with a LEGAL action,
                    within the 1-second-per-decision limit
  4. POST /hand-end accepts a finished hand without crashing

It uses only the Python standard library, so it runs inside any agent image.
It does NOT import game.py: the sample situations are written out below, so
a participant's edited game.py can't change what is being tested.

Usage (inside the container):
    python smoke_test.py [--port 5008] [--startup 15]
Exit code 0 means every check passed; 1 means a check failed.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time

# Limits copied from game.py (keep these in sync with the tournament rules).
STARTUP_LIMIT_SECONDS = 15.0   # the server must answer GET / this quickly
DECISION_LIMIT_SECONDS = 1.0   # one /act request may take at most this long

# Sample situations: (description, observation). Each one was produced by
# game.py, so it is exactly what the judge would send in a real match.
SAMPLE_OBSERVATIONS = [
    (
        'preflop, button to act',
        {'hand_index': 0, 'street': 'PREFLOP', 'hole_cards': [1, 12, 17, 24, 28], 'board': [], 'pot': 3, 'to_call': 1, 'my_bet': 1, 'opp_bet': 2, 'my_stack': 79, 'opp_stack': 78, 'is_button': True, 'legal_actions': ['FOLD', 'CHECK_CALL', 'RAISE_HALF', 'RAISE_POT'], 'raise_to': {'RAISE_HALF': 4, 'RAISE_POT': 6}, 'min_raise_to': 4, 'raises_this_street': 0, 'action_history': [], 'big_blind': 2},
    ),
    (
        'preflop, facing a pot raise',
        {'hand_index': 0, 'street': 'PREFLOP', 'hole_cards': [4, 9, 16, 21, 27], 'board': [], 'pot': 8, 'to_call': 4, 'my_bet': 2, 'opp_bet': 6, 'my_stack': 78, 'opp_stack': 74, 'is_button': False, 'legal_actions': ['FOLD', 'CHECK_CALL', 'RAISE_HALF', 'RAISE_POT'], 'raise_to': {'RAISE_HALF': 12, 'RAISE_POT': 18}, 'min_raise_to': 10, 'raises_this_street': 1, 'action_history': [{'street': 'PREFLOP', 'is_me': False, 'action': 'RAISE_POT', 'amount': 5, 'street_total': 6}], 'big_blind': 2},
    ),
    (
        'flop, first to act',
        {'hand_index': 0, 'street': 'FLOP', 'hole_cards': [4, 9, 16, 21, 27], 'board': [22, 0, 29], 'pot': 12, 'to_call': 0, 'my_bet': 0, 'opp_bet': 0, 'my_stack': 74, 'opp_stack': 74, 'is_button': False, 'legal_actions': ['CHECK_CALL', 'RAISE_HALF', 'RAISE_POT'], 'raise_to': {'RAISE_HALF': 6, 'RAISE_POT': 12}, 'min_raise_to': 2, 'raises_this_street': 0, 'action_history': [{'street': 'PREFLOP', 'is_me': False, 'action': 'RAISE_POT', 'amount': 5, 'street_total': 6}, {'street': 'PREFLOP', 'is_me': True, 'action': 'CHECK_CALL', 'amount': 4, 'street_total': 6}], 'big_blind': 2},
    ),
    (
        'flop, facing an all-in (only FOLD or CHECK_CALL)',
        {'hand_index': 1, 'street': 'FLOP', 'hole_cards': [4, 9, 16, 21, 27], 'board': [22, 0, 29], 'pot': 134, 'to_call': 26, 'my_bet': 36, 'opp_bet': 62, 'my_stack': 26, 'opp_stack': 0, 'is_button': False, 'legal_actions': ['FOLD', 'CHECK_CALL'], 'raise_to': {}, 'min_raise_to': 88, 'raises_this_street': 2, 'action_history': [{'street': 'PREFLOP', 'is_me': False, 'action': 'RAISE_POT', 'amount': 5, 'street_total': 6}, {'street': 'PREFLOP', 'is_me': True, 'action': 'RAISE_POT', 'amount': 16, 'street_total': 18}, {'street': 'PREFLOP', 'is_me': False, 'action': 'CHECK_CALL', 'amount': 12, 'street_total': 18}, {'street': 'FLOP', 'is_me': True, 'action': 'RAISE_POT', 'amount': 36, 'street_total': 36}, {'street': 'FLOP', 'is_me': False, 'action': 'RAISE_POT', 'amount': 62, 'street_total': 62}], 'big_blind': 2},
    ),
    (
        'river, facing a half-pot bet',
        {'hand_index': 2, 'street': 'RIVER', 'hole_cards': [1, 13, 14, 16, 18], 'board': [7, 23, 15, 29, 9], 'pot': 6, 'to_call': 2, 'my_bet': 0, 'opp_bet': 2, 'my_stack': 78, 'opp_stack': 76, 'is_button': True, 'legal_actions': ['FOLD', 'CHECK_CALL', 'RAISE_HALF', 'RAISE_POT'], 'raise_to': {'RAISE_HALF': 6, 'RAISE_POT': 10}, 'min_raise_to': 4, 'raises_this_street': 1, 'action_history': [{'street': 'PREFLOP', 'is_me': True, 'action': 'CHECK_CALL', 'amount': 1, 'street_total': 2}, {'street': 'PREFLOP', 'is_me': False, 'action': 'CHECK_CALL', 'amount': 0, 'street_total': 2}, {'street': 'FLOP', 'is_me': False, 'action': 'CHECK_CALL', 'amount': 0, 'street_total': 0}, {'street': 'FLOP', 'is_me': True, 'action': 'CHECK_CALL', 'amount': 0, 'street_total': 0}, {'street': 'TURN', 'is_me': False, 'action': 'CHECK_CALL', 'amount': 0, 'street_total': 0}, {'street': 'TURN', 'is_me': True, 'action': 'CHECK_CALL', 'amount': 0, 'street_total': 0}, {'street': 'RIVER', 'is_me': False, 'action': 'RAISE_HALF', 'amount': 2, 'street_total': 2}], 'big_blind': 2},
    ),
]

# A finished hand, in the format the judge sends with /act and /hand-end.
SAMPLE_HAND_RESULT = {'hand_index': 1, 'my_net': 80, 'pot': 160, 'went_to_showdown': True, 'board': [22, 0, 29, 7, 10], 'opp_hole': [4, 9, 16, 21, 27], 'winner': 'me'}


class SmokeTestFailure(Exception):
    """One of the checks failed. The message says which one and why."""


def request(port: int, method: str, path: str, body: dict | None, timeout: float) -> tuple[int, object, float]:
    """Send one HTTP request to the bot on localhost.

    Returns (status code, parsed JSON reply, seconds taken).
    """
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("AGENT_TOKEN")
    if token:
        headers["X-Agent-Token"] = token  # the same header the judge sends
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    start = time.perf_counter()
    try:
        conn.request(method, path, body=None if body is None else json.dumps(body), headers=headers)
        response = conn.getresponse()
        raw = response.read()
    finally:
        conn.close()
    elapsed = time.perf_counter() - start
    try:
        data = json.loads(raw) if raw else None
    except ValueError:
        raise SmokeTestFailure(f"{method} {path} did not return JSON: {raw[:200]!r}") from None
    return response.status, data, elapsed


def describe_crash(data: object) -> str:
    """Pull the error and traceback out of a 500 reply, if there is one."""
    if isinstance(data, dict) and "error" in data:
        return f"{data['error']}\n{data.get('traceback', '')}"
    return str(data)[:500]


def check_hand_end_errors(data: object, where: str) -> None:
    """The server reports crashes inside on_hand_end() in 'hand_end_errors'."""
    if isinstance(data, dict) and data.get("hand_end_errors"):
        first = data["hand_end_errors"][0]
        raise SmokeTestFailure(f"on_hand_end() crashed during {where}:\n{describe_crash(first)}")


def wait_for_startup(port: int, limit: float) -> dict:
    """Step 1: poll GET / until the bot answers, or fail after `limit` seconds."""
    deadline = time.monotonic() + limit
    last_problem = "no answer yet"
    while time.monotonic() < deadline:
        try:
            status, data, _ = request(port, "GET", "/", None, timeout=2.0)
            if status == 200 and isinstance(data, dict):
                return data
            last_problem = f"GET / returned HTTP {status}: {str(data)[:200]}"
        except (OSError, http.client.HTTPException) as exc:
            last_problem = f"{type(exc).__name__}: {exc}"  # server not listening yet
        except SmokeTestFailure as exc:
            last_problem = str(exc)
        time.sleep(0.25)
    raise SmokeTestFailure(
        f"the server did not answer GET / within {limit:g} s ({last_problem}). "
        "Does agent.py still read $PORT and listen on 0.0.0.0? Is startup work too slow?"
    )


def run_checks(port: int, startup_limit: float) -> list[str]:
    """Run every check. Returns lines describing what passed; raises on failure."""
    passed = []

    # 1. Startup and identity.
    started = time.monotonic()
    info = wait_for_startup(port, startup_limit)
    name = f"{info.get('participant', '?')} / {info.get('agent_name', '?')}"
    passed.append(f"GET / answered after {time.monotonic() - started:.1f} s: {name}")

    # 2. Start a half (the judge sends only the match settings).
    config = {"starting_stack": 80, "big_blind": 2, "small_blind": 1,
              "num_hands": 1000, "time_bank_seconds": 8.0, "raise_cap": 4}
    status, data, _ = request(port, "POST", "/new-half", {"config": config}, timeout=10.0)
    if status != 200:
        raise SmokeTestFailure(f"POST /new-half returned HTTP {status}:\n{describe_crash(data)}")
    passed.append("POST /new-half worked")

    # 3. Ask for a decision in each sample situation. The first request also
    #    delivers a finished hand, the way the judge batches hand results.
    pending = [SAMPLE_HAND_RESULT]
    for label, obs in SAMPLE_OBSERVATIONS:
        body = {"observation": obs, "hand_results": pending}
        pending = []
        try:
            status, data, elapsed = request(port, "POST", "/act", body, timeout=DECISION_LIMIT_SECONDS + 1.0)
        except TimeoutError:
            raise SmokeTestFailure(f"/act ({label}) did not answer within {DECISION_LIMIT_SECONDS + 1:g} s") from None
        if status != 200:
            raise SmokeTestFailure(f"/act ({label}) returned HTTP {status}:\n{describe_crash(data)}")
        check_hand_end_errors(data, f"/act ({label})")
        action = data.get("action") if isinstance(data, dict) else None
        if action not in obs["legal_actions"]:
            raise SmokeTestFailure(
                f"/act ({label}) answered {action!r}, but the legal actions were {obs['legal_actions']}")
        if elapsed > DECISION_LIMIT_SECONDS:
            raise SmokeTestFailure(
                f"/act ({label}) took {elapsed:.2f} s; the limit is {DECISION_LIMIT_SECONDS:g} s per decision. "
                "Move slow setup (loading models, building tables) to startup.")
        passed.append(f"/act {label}: {action} in {elapsed * 1000:.0f} ms")

    # 4. End of half: the leftover finished hands arrive at /hand-end.
    status, data, _ = request(port, "POST", "/hand-end", {"results": [SAMPLE_HAND_RESULT]}, timeout=10.0)
    if status != 200:
        raise SmokeTestFailure(f"POST /hand-end returned HTTP {status}:\n{describe_crash(data)}")
    check_hand_end_errors(data, "/hand-end")
    passed.append("POST /hand-end worked")
    return passed


def main() -> int:
    """Parse options, run the checks, print a readable report."""
    parser = argparse.ArgumentParser(description="Smoke-test a running poker bot server.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5008")))
    parser.add_argument("--startup", type=float, default=STARTUP_LIMIT_SECONDS,
                        help="seconds to wait for GET / (default 15)")
    args = parser.parse_args()
    try:
        for line in run_checks(args.port, args.startup):
            print(f"  ok  {line}")
    except SmokeTestFailure as exc:
        print(f"SMOKE TEST FAILED: {exc}")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
