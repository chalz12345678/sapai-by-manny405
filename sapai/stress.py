"""
Stress test the engine with random battles and random shop play, checking
rules that must always hold.

    python -m sapai.stress                       # a quick run
    python -m sapai.stress --battles 20000 --games 500 --seed 1

Battles: random teams of random pets, levels, stats and perks.
  - no errors, and every battle finishes (a time limit catches loops)
  - the same random seed gives the same result
  - a battle that does not depend on chance gives the mirror result when
    the teams swap sides (neither side is favoured)
  - no pet ever has more than the maximum stats, no team has more than the
    maximum team size, and no survivor has 0 or less health
Shop play: random legal actions for whole 1v1 matches.
  - no errors from legal actions
  - gold never negative, team size and stats within the limits, shop size
    within the limit, lives and wins add up

Problems are grouped by kind, with the first example of each.
"""

import argparse
import collections
import random
import signal
import sys
import traceback

import numpy as np

from sapai.actions import END_TURN, action_mask, apply_action
from sapai.battle import Battle
from sapai.data import RULES, data
from sapai.match import Match
from sapai.pets import Pet
from sapai.teams import Team

SHOP_PETS = [k for k, p in data["pets"].items() if isinstance(p["tier"], int)]
PERKS = ["none"] + sorted(data["statuses"])
FLIP = {0: 1, 1: 0, 2: 2}


class TimeLimit(Exception):
    pass


def _alarm(signum, frame):
    raise TimeLimit()


def random_team(rng):
    pets = []
    for _ in range(rng.randint(1, RULES["max_team_size"])):
        p = Pet(rng.choice(SHOP_PETS))
        p.level = rng.choice([1, 1, 2, 3])
        p._attack = rng.randint(1, 20)
        p._health = rng.randint(1, 20)
        p.status = rng.choice(PERKS) if rng.random() < 0.4 else "none"
        pets.append(p)
    return Team(pets)


def team_problems(team, where):
    problems = []
    filled = [s.pet for s in team if not s.empty]
    if len(filled) > RULES["max_team_size"]:
        problems.append(f"{where}: {len(filled)} pets on a team")
    for p in filled:
        a = p._attack + p._until_end_of_battle_attack_buff
        h = p._health + p._until_end_of_battle_health_buff
        if min(a, RULES["max_stat"]) != p.attack and p.status in ("none",):
            problems.append(f"{where}: shown attack differs from stats")
        if p.attack > RULES["max_stat"] and p.status not in (
            "status-poison-attack",
            "status-bone-attack",
            "status-steak-attack",
        ):
            problems.append(f"{where}: attack above the maximum")
        if p.health > RULES["max_stat"]:
            problems.append(f"{where}: health above the maximum")
    return problems


def ending(result, b):
    """The result and every survivor (name, stats, level, perk) on both sides"""
    return (result, describe(b.t0), describe(b.t1))


def mirror(end):
    result, a, b = end
    return (FLIP[result], b, a)


def run_battle(t0, t1, seed, time_limit=2):
    np.random.seed(seed)
    signal.alarm(time_limit)
    try:
        b = Battle(t0, t1)
        result = b.battle()
    finally:
        signal.alarm(0)
    return result, b


def stress_battles(n, seed, problems, examples, checks):
    rng = random.Random(seed)
    for i in range(n):
        t0, t1 = random_team(rng), random_team(rng)
        where = f"battle {i}"
        try:
            r1, b = run_battle(t0, t1, seed=i)
            end1 = ending(r1, b)
            r2, b2 = run_battle(t0, t1, seed=i)
            if ending(r2, b2) != end1:
                note(problems, examples, "same seed, different ending", where, t0, t1)
            for team, side in ((b.t0, "team 0"), (b.t1, "team 1")):
                for msg in team_problems(team, side):
                    note(problems, examples, msg.split(": ", 1)[1], where, t0, t1)
                for s in team:
                    if not s.empty and s.pet.health <= 0:
                        note(
                            problems,
                            examples,
                            "a fainted pet is left on the team",
                            where,
                            t0,
                            t1,
                        )
            checks["battles"] += 1
            ### Side fairness, only for battles whose whole ending does not
            ###   depend on chance (the same over several seeds)
            endings = {end1} | {
                ending(*run_battle(t0, t1, seed=i + k)) for k in (1000, 2000)
            }
            if len(endings) == 1:
                checks["battles without chance (side check)"] += 1
                swapped = {
                    mirror(ending(*run_battle(t1, t0, seed=i + k)))
                    for k in (0, 1000, 2000)
                }
                if swapped != {end1}:
                    other = sorted(swapped - {end1})[0] if swapped - {end1} else ""
                    note(
                        problems,
                        examples,
                        "swapping sides changes a battle with no chance in it",
                        where,
                        t0,
                        t1,
                        extra=f"ending {end1}\n    swapped {other}",
                    )
        except TimeLimit:
            note(problems, examples, "battle did not finish", where, t0, t1)
        except Exception as e:
            note(
                problems,
                examples,
                f"error: {type(e).__name__}: {str(e)[:80]}",
                where,
                t0,
                t1,
                extra=traceback.format_exc(limit=4),
            )


def stress_matches(n, seed, problems, examples, checks, max_turns=30):
    rng = random.Random(seed + 1)
    for g in range(n):
        m = Match(seed=seed * 7919 + g, max_turns=max_turns)
        where = f"match {g}"
        try:
            while not m.over:
                for i, p in enumerate(m.players):
                    for _ in range(25):
                        mask = action_mask(p)
                        legal = [a for a in np.flatnonzero(mask) if a != END_TURN]
                        if not legal or rng.random() < 0.08:
                            break
                        action = int(rng.choice(legal))
                        checks["shop actions"] += 1
                        gold, turn = p.gold, p.turn
                        signal.alarm(2)
                        try:
                            apply_action(p, action)
                        finally:
                            signal.alarm(0)
                        if p.gold < 0:
                            note(
                                problems,
                                examples,
                                "gold went negative",
                                where,
                                extra=f"action {action}",
                            )
                        if len(p.shop.slots) > RULES["max_shop_slots"]:
                            note(problems, examples, "shop over the slot limit", where)
                        for msg in team_problems(p.team, f"player {i}"):
                            note(
                                problems,
                                examples,
                                msg.split(": ", 1)[1] + " (shop)",
                                where,
                            )
                m.battle()
                checks["match turns"] += 1
            lives_lost = sum(
                m.lives_lost(r["turn"]) for r in m.history if r["result"] in (0, 1)
            )
            spent = sum(m.start_lives - p.lives for p in m.players)
            if spent != lives_lost:
                note(problems, examples, "lives lost do not add up", where)
            if sum(m.wins) != sum(r["result"] in (0, 1) for r in m.history):
                note(problems, examples, "wins do not add up", where)
        except TimeLimit:
            note(problems, examples, "shop action did not finish", where)
        except Exception as e:
            note(
                problems,
                examples,
                f"error: {type(e).__name__}: {str(e)[:80]}",
                where,
                extra=traceback.format_exc(limit=4),
            )


def note(problems, examples, kind, where, t0=None, t1=None, extra=""):
    problems[kind] += 1
    if kind not in examples:
        teams = ""
        if t0 is not None:
            teams = f"\n    team 0: {describe(t0)}\n    team 1: {describe(t1)}"
        examples[kind] = f"{where}{teams}" + (f"\n    {extra}" if extra else "")


def describe(team):
    out = []
    for s in team:
        if s.empty:
            continue
        p = s.pet
        perk = "" if p.status == "none" else " " + data["statuses"][p.status]["name"]
        out.append(
            f"{data['pets'][p.name]['name']} {p._attack}/{p._health} L{p.level}{perk}"
        )
    return ", ".join(out)


def run(battles=2000, games=50, seed=0):
    """
    Run the stress test. Returns ({problem: count}, {problem: example},
    {check: how many times it ran})
    """
    old = signal.signal(signal.SIGALRM, _alarm)
    problems, examples = collections.Counter(), {}
    checks = collections.Counter()
    try:
        stress_battles(battles, seed, problems, examples, checks)
        stress_matches(games, seed, problems, examples, checks)
    finally:
        signal.signal(signal.SIGALRM, old)
    return problems, examples, checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--battles", type=int, default=2000)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    problems, examples, checks = run(args.battles, args.games, args.seed)
    print(f"{args.battles} battles and {args.games} matches")
    for check, count in checks.items():
        print(f"  {count} {check}")
    if not problems:
        print("No problems found")
        return 0
    for kind, count in problems.most_common():
        print(f"\n{count} x {kind}\n  first: {examples[kind]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
