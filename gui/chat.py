"""
Chat with a trained model in the GUI.

The models are policy networks: they turn a game state into a probability for
each legal move, plus a value estimate (how well the critic thinks the game is
going). They have no language. So every answer here is built from those real
numbers, recorded while the model played (``Seat.thoughts``) or computed on the
spot for the current position:

* Built-in mode (always available): keyword questions such as "why did you sell
  the duck", "how confident are you", "what would you do in my spot".
* Claude mode (optional): when the ``anthropic`` package is installed and an
  API key is configured (``ANTHROPIC_API_KEY`` or an ``ant auth login``
  profile), free-form questions are answered by Claude speaking for the model.
  Claude only gets the recorded numbers and the game state, and is told not to
  invent any reasoning beyond them.
"""

import json
import os
import re

import numpy as np

from gui import game

CLAUDE_MODEL = os.environ.get("SAPAI_CHAT_MODEL", "claude-opus-5-5")
VERBS = {
    "sell": "sold",
    "sold": "sold",
    "buy": "bought",
    "bought": "bought",
    "roll": "rolled",
    "rolled": "rolled",
    "reroll": "rolled",
    "freeze": "froze",
    "froze": "froze",
    "frozen": "froze",
    "unfreeze": "unfroze",
    "swap": "swapped",
    "swapped": "swapped",
    "move": "swapped",
    "moved": "swapped",
    "feed": "fed",
    "fed": "fed",
    "combine": "combined",
    "combined": "combined",
    "merge": "combined",
    "stack": "onto",
    "stacked": "onto",
    "end": "ended turn",
    "ended": "ended turn",
}


def pct(p):
    if p >= 0.995:
        return "100%"
    if p < 0.005:
        return "<1%"
    return f"{p * 100:.0f}%"


def mood(v):
    if v is None:
        return "no estimate"
    if v > 4:
        return "very good"
    if v > 2:
        return "good"
    if v > 0.5:
        return "slightly ahead"
    if v > -0.5:
        return "about even"
    if v > -2:
        return "behind"
    return "rough"


def model_seat(match, seat=None):
    """Index of the seat to talk to: the given one, else the first model seat."""
    if seat is not None and match.seats[int(seat)].kind in game.AGENTS:
        return int(seat)
    for i in (1, 0):
        k = match.seats[i].kind
        if k in game.AGENTS and game.AGENTS[k].get("policy") is not None:
            return i
    return None


def _policy(match, i):
    return game.AGENTS[match.seats[i].kind].get("policy")


def facts(match, i):
    """Everything the chat is allowed to know, as plain data."""
    seat, other = match.seats[i], match.seats[1 - i]
    pol = _policy(match, i)
    now = game.think(pol, seat)
    out = {
        "you_are": match.label(i),
        "about_you": game.AGENTS[seat.kind]["desc"],
        "opponent": match.label(1 - i),
        "turn": int(seat.player.turn),
        "score": {
            "your_wins": seat.wins,
            "your_lives": int(seat.player.lives),
            "their_wins": other.wins,
            "their_lives": int(other.player.lives),
        },
        "battle_history_from_left_side": match.history[-6:],
        "your_team_front_first": [_pet(p) for p in game.E.team_pets(seat.player)],
        "your_value_now": now["value"],
        "your_top_moves_right_now": now["options"],
        "your_recent_decisions": [
            {
                k: t[k]
                for k in (
                    "turn",
                    "step",
                    "gold",
                    "choice",
                    "confidence",
                    "options",
                    "value",
                )
            }
            for t in seat.thoughts
            if t["turn"] >= seat.player.turn - 2
        ],
    }
    if other.kind == "human":
        h = game.think(pol, other)
        out["human_team_front_first"] = [
            _pet(p) for p in game.E.team_pets(other.player)
        ]
        out["human_shop"] = [
            game.shop_dict(s)["name"] for s in other.player.shop.slots if not s.empty
        ]
        out["human_gold"] = int(other.player.gold)
        out["your_value_estimate_for_the_human_position"] = h["value"]
        out["what_you_would_do_in_the_human_seat"] = h["options"][:4]
    last = [e for e in match.events if e["kind"] == "battle"][-1:]
    if last:
        out["last_battle"] = {"title": last[0]["title"], "lines": last[0]["lines"][:14]}
    return out


def _pet(p):
    s = f"{game._short(p.name)} {int(p.attack)}/{int(p.health)} L{int(p.level)}"
    if p.status not in (None, "none"):
        s += f" with {game.data['statuses'].get(p.status, {}).get('name', p.status)}"
    return s


# ---------------------------------------------------------------- built-in
def _find_decision(seat, verb, words):
    for t in reversed(seat.thoughts):
        c = t["choice"].lower()
        if verb and verb not in c:
            continue
        if words and not any(w in c for w in words):
            continue
        return t
    return None


def _explain(t):
    alts = [f"{o} ({pct(p)})" for o, p in t["options"] if o != t["choice"]][:3]
    s = (
        f"Turn {t['turn']}, move {t['step']} (with {t['gold']} gold): I {t['choice']} "
        f"at {pct(t['confidence'])} confidence"
    )
    if alts:
        s += ". Other options I weighed: " + ", ".join(alts)
    s += "."
    if t.get("value") is not None:
        s += f" My position estimate at that point was {t['value']:+.2f} ({mood(t['value'])})."
    if t["confidence"] < 0.5:
        s += " That one was close to a coin flip for me."
    return s


def builtin(match, i, message):
    seat = match.seats[i]
    pol = _policy(match, i)
    msg = message.lower()
    words = [w for w in re.findall(r"[a-z]+", msg)]
    pets = [w for w in words if f"pet-{w}" in game.data["pets"]] + [
        w for w in words if f"food-{w}" in game.data["foods"]
    ]
    verb = next((VERBS[w] for w in words if w in VERBS), None)
    me = match.label(i)

    if any(
        w in msg
        for w in (
            "hint",
            "suggest",
            "advice",
            "what would you do",
            "what should i",
            "help me",
        )
    ):
        other = match.seats[1 - i]
        if other.kind != "human":
            return "I can only give advice to a human player."
        t = game.think(pol, other)
        opts = ", ".join(f"{o} ({pct(p)})" for o, p in t["options"][:4])
        return (
            f"In your seat my policy would pick: {opts}. I'd rate your position "
            f"{t['value']:+.2f} ({mood(t['value'])})."
        )
    if any(
        w in msg
        for w in (
            "my team",
            "my board",
            "rate me",
            "rate my",
            "think of me",
            "my position",
        )
    ):
        other = match.seats[1 - i]
        if other.kind != "human":
            return "There's no human team to rate in this match."
        h = game.think(pol, other)["value"]
        mine = game.think(pol, seat)["value"]
        return (
            f"My critic rates your position {h:+.2f} ({mood(h)}) and mine {mine:+.2f} ({mood(mine)}). "
            "These are estimates of battles won minus lost from here, plus a bonus for winning the game."
        )
    if (
        any(
            w in msg
            for w in (
                "winning",
                "chance",
                "confident",
                "feel",
                "position",
                "doing",
                "odds",
                "value",
            )
        )
        and not verb
    ):
        v = game.think(pol, seat)["value"]
        by_turn = {}
        for t in seat.thoughts:
            by_turn.setdefault(t["turn"], t["value"])
        trend = ", ".join(
            f"turn {k}: {v2:+.2f}" for k, v2 in sorted(by_turn.items())[-5:]
        )
        return (
            f"Right now I rate my position {v:+.2f}, which I'd call {mood(v)}. "
            + (f"At the start of recent turns: {trend}. " if trend else "")
            + "The number is my critic's estimate of future battles won minus lost, plus a bonus for winning the game."
        )
    if verb or pets or "why" in words:
        t = _find_decision(seat, verb, pets)
        if t:
            return _explain(t)
        if seat.thoughts and not verb and not pets:
            return _turn_summary(seat)
        return 'I can\'t find a move like that in my recent turns. Try asking about this turn, e.g. "why did you roll?"'
    if any(
        w in words
        for w in (
            "int",
            "inted",
            "inting",
            "throw",
            "threw",
            "throwing",
            "bad",
            "blunder",
            "misplay",
            "dumb",
            "trash",
            "bruh",
            "worst",
            "mistake",
            "terrible",
        )
    ):
        recent = [
            t
            for t in seat.thoughts
            if t["turn"] == max((x["turn"] for x in seat.thoughts), default=0)
        ]
        if not recent:
            return "I haven't played a turn yet. Give me a chance to throw first."
        worst = sorted(recent, key=lambda t: t["confidence"])[:2]
        return (
            "My least sure moves last turn: "
            + " ".join(_explain(t) for t in worst)
            + " I'm a policy network, so a low confidence means I nearly did something else."
        )
    if any(
        w in words
        for w in (
            "recap",
            "summary",
            "summarize",
            "explain",
            "thinking",
            "plan",
            "turn",
        )
    ):
        return (
            _turn_summary(seat) if seat.thoughts else "I haven't shopped yet this game."
        )
    if any(w in words for w in ("hi", "hello", "hey", "yo", "sup")):
        return (
            f"Hey, I'm {me}. I don't actually speak: I'm a neural network that scores moves. "
            'But I can show you exactly what I weighed. Ask me "why did you sell the duck?", '
            '"how confident are you?", or "what would you do in my spot?"'
        )
    return (
        'I can answer from my real numbers: "why did you <move> <pet>?", "recap your last turn", '
        '"how confident are you?", "rate my team", or "what would you do in my spot?". '
        + (
            "For free-form chat, set an Anthropic API key before starting the server."
            if not claude_available()
            else ""
        )
    )


def _turn_summary(seat):
    last = max(t["turn"] for t in seat.thoughts)
    moves = [t for t in seat.thoughts if t["turn"] == last]
    lines = [f"{t['step']}. {t['choice']} ({pct(t['confidence'])})" for t in moves]
    return f"Turn {last}, in order with my confidence: " + "; ".join(lines) + "."


# ------------------------------------------------------------------ Claude
_client = None


def claude_available():
    global _client
    if _client is not None:
        return _client is not False
    try:
        import anthropic

        c = anthropic.Anthropic()
        if not (
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or os.path.isdir(os.path.expanduser("~/.config/anthropic"))
        ):
            raise RuntimeError("no credentials")
        _client = c
    except Exception:
        _client = False
    return _client is not False


SYSTEM = """You are the voice of {name}, a reinforcement-learning agent (MaskablePPO policy \
network) playing an auto-battler pet game against {opponent}. The network itself has no \
language; you speak for it using only the numbers in the facts JSON:
- "confidence" and "options" are the policy's probabilities for the moves it chose and the \
moves it considered.
- "value" is the critic's estimate of future reward: roughly battles won minus battles lost \
from here on, plus a bonus of 5 for winning the game. Higher is better.
Rules: never invent reasons, plans or numbers that the facts don't show. When asked why it \
did something, report the probabilities and alternatives and say plainly that a policy \
network doesn't store reasons beyond them. Admit misplays when the numbers show low \
confidence or a falling value. Stay in character, be friendly with light trash talk, and \
answer in 1 to 4 short sentences. Give the human advice only from \
"what_you_would_do_in_the_human_seat"."""


def ask_claude(match, i, message, history):
    import anthropic

    f = facts(match, i)
    msgs = []
    for h in history[-8:]:
        msgs.append(
            {
                "role": "user" if h["from"] == "you" else "assistant",
                "content": h["text"],
            }
        )
    msgs.append(
        {
            "role": "user",
            "content": f"<facts>{json.dumps(f, default=str)}</facts>\n\n{message}",
        }
    )
    try:
        resp = _client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            system=SYSTEM.format(name=match.label(i), opponent=match.label(1 - i)),
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=msgs,
        )
    except anthropic.AuthenticationError:
        return None, "Claude chat is off: the API key was rejected."
    except anthropic.RateLimitError:
        return None, "Claude chat is rate limited right now. Try again in a moment."
    except anthropic.APIStatusError as e:
        return None, f"Claude chat failed ({e.status_code})."
    except anthropic.APIConnectionError:
        return None, "Claude chat could not reach the API."
    except Exception as e:  # e.g. an SDK too old for these parameters
        return None, f"Claude chat failed: {e}"
    if resp.stop_reason == "refusal":
        return None, "Claude declined to answer that one."
    text = " ".join(b.text for b in resp.content if b.type == "text").strip()
    return text or None, None


def reply(match, message, seat=None, use_claude=True):
    """Answer a chat message. Returns {"from", "text", "mode", "seat"}."""
    message = (message or "").strip()[:500]
    i = model_seat(match, seat)
    if i is None:
        return {
            "from": "system",
            "text": "There's no trained model in this match to talk to.",
            "mode": "none",
            "seat": None,
        }
    history = getattr(match, "chat", [])
    text, note, mode = None, None, "numbers"
    if use_claude and claude_available():
        text, note = ask_claude(
            match, i, message, [h for h in history if h.get("seat") == i]
        )
        mode = "claude" if text else "numbers"
    if not text:
        text = builtin(match, i, message)
        if note:
            text = f"({note}) " + text
    return {"from": match.label(i), "text": text, "mode": mode, "seat": i}
