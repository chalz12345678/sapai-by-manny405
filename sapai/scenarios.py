"""
Battle scenarios: situations from the real game, written down so the engine
can be checked against them. See tests/scenarios/README.md for the format.

    python -m sapai.scenarios                 # run every scenario
    python -m sapai.scenarios tests/scenarios/perks.yaml
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

from sapai.battle import Battle
from sapai.data import RULES, data
from sapai.pets import Pet
from sapai.teams import Team

SCENARIO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests", "scenarios"
)
RESULTS = {0: "win", 1: "loss", 2: "draw"}

PET_IDS = {p["name"].lower(): k for k, p in data["pets"].items() if k != "pet-none"}
PERK_IDS = {s["name"].lower(): k for k, s in data["statuses"].items()}

PET_SPEC = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z ]*?)\s+(?P<attack>\d+)/(?P<health>\d+)"
    r"(?:\s+L(?P<level>[123]))?"
    r"(?:\s+(?P<perk>[A-Za-z][A-Za-z ]*))?$"
)


################################################################################
#### Parsing
################################################################################


def parse_pet(spec):
    """
    "Hedgehog 3/2", "Hedgehog 3/2 L2" or "Hedgehog 3/2 L2 Garlic" (front of
    the team first). Also accepts a dict with name, attack, health, level and
    perk.
    """
    if isinstance(spec, dict):
        d = dict(spec)
    else:
        m = PET_SPEC.match(spec.strip())
        if not m:
            raise ValueError(
                f"cannot read pet {spec!r}: expected 'Name A/H [L2] [Perk]'"
            )
        d = m.groupdict()
    name = d["name"].strip().lower()
    if name not in PET_IDS:
        raise ValueError(f"unknown pet {d['name']!r}")
    perk = (d.get("perk") or "").strip().lower()
    if perk and perk not in PERK_IDS:
        raise ValueError(f"unknown perk {d['perk']!r}: use a name like Garlic or Melon")
    return {
        "id": PET_IDS[name],
        "attack": int(d["attack"]),
        "health": int(d["health"]),
        "level": int(d.get("level") or 1),
        "perk": PERK_IDS[perk] if perk else "none",
    }


def make_team(specs):
    pets = []
    for spec in specs:
        s = parse_pet(spec)
        pet = Pet(s["id"])
        pet.level = s["level"]
        pet._attack = s["attack"]
        pet._health = s["health"]
        pet.status = s["perk"]
        pets.append(pet)
    return Team(pets)


def shown_stats(pet):
    """Attack and health as the game shows them (perks not included)"""
    cap = RULES["max_stat"]
    return (
        min(pet._attack + pet._until_end_of_battle_attack_buff, cap),
        min(pet._health + pet._until_end_of_battle_health_buff, cap),
    )


def describe(team):
    out = []
    for slot in team:
        if slot.empty:
            continue
        p = slot.pet
        a, h = shown_stats(p)
        text = f"{data['pets'][p.name]['name']} {a}/{h}"
        if p.level > 1:
            text += f" L{p.level}"
        if p.status != "none":
            text += f" {data['statuses'][p.status]['name']}"
        out.append(text)
    return out


def load(paths=None):
    """Scenarios from YAML files (all files in tests/scenarios by default)"""
    import yaml

    if paths is None:
        paths = sorted(glob.glob(os.path.join(SCENARIO_DIR, "*.yaml")))
    scenarios = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for i, s in enumerate(yaml.safe_load(f) or []):
                s = dict(s)
                s["file"] = os.path.basename(path)
                s.setdefault("name", f"{s['file']} #{i + 1}")
                scenarios.append(s)
    return scenarios


################################################################################
#### Running
################################################################################


def run_battle(scenario, seed):
    np.random.seed(seed)
    b = Battle(make_team(scenario["team"]), make_team(scenario["enemy"]))
    result = RESULTS[b.battle()]
    return result, describe(b.t0), describe(b.t1)


def check(scenario, runs=None):
    """
    Problems with the engine on a scenario, as a list of strings (empty when
    it matches). A scenario with an exact result is run several times with
    different seeds, so hidden randomness shows up as a problem too.
    """
    problems = []
    if "odds" in scenario:
        runs = runs or scenario.get("runs", 400)
        counts = {"win": 0, "loss": 0, "draw": 0}
        for seed in range(runs):
            counts[run_battle(scenario, seed)[0]] += 1
        tolerance = scenario.get("tolerance", 0.08)
        for outcome in counts:
            expected = scenario["odds"].get(outcome, 0)
            got = counts[outcome] / runs
            if abs(got - expected) > tolerance:
                problems.append(
                    f"{outcome}: expected {expected:.0%}, got {got:.0%} over {runs} battles"
                )
        return problems

    if scenario.get("seen"):
        ### One battle seen in the game: the engine must be able to end the
        ###   same way (same result and survivors) on at least one seed
        runs = runs or scenario.get("runs", 200)
        want_team = scenario.get("survivors", {}).get("team")
        want_enemy = scenario.get("survivors", {}).get("enemy")
        want_team = describe(make_team(want_team)) if want_team is not None else None
        want_enemy = describe(make_team(want_enemy)) if want_enemy is not None else None
        results = {}
        for seed in range(runs):
            result, team, enemy = run_battle(scenario, seed)
            results[result] = results.get(result, 0) + 1
            if (
                ("result" not in scenario or result == scenario["result"])
                and (want_team is None or team == want_team)
                and (want_enemy is None or enemy == want_enemy)
            ):
                return []
        odds = ", ".join(f"{k} {v / runs:.0%}" for k, v in sorted(results.items()))
        return [f"the engine never ends this way in {runs} battles ({odds})"]

    runs = runs or scenario.get("runs", 5)
    seen = set()
    for seed in range(runs):
        result, team, enemy = run_battle(scenario, seed)
        seen.add((result, tuple(team), tuple(enemy)))
        if "result" in scenario and result != scenario["result"]:
            problems.append(f"expected a {scenario['result']}, got a {result}")
        for side, got in (("team", team), ("enemy", enemy)):
            expected = scenario.get("survivors", {}).get(side)
            if expected is not None:
                expected = describe(make_team(expected))
                if got != expected:
                    problems.append(f"{side} survivors: expected {expected}, got {got}")
        if problems:
            return [f"seed {seed}: {p}" for p in problems]
    if len(seen) > 1 and "odds" not in scenario:
        problems.append(
            f"the battle is random ({len(seen)} different endings over {runs} seeds),"
            " so give odds instead of an exact result"
        )
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("files", nargs="*", help="scenario files (default: all)")
    args = parser.parse_args(argv)
    scenarios = load(args.files or None)
    failed = 0
    for s in scenarios:
        problems = check(s)
        mark = "ok  " if not problems else "FAIL"
        tag = "verified" if s.get("verified") else "unverified"
        print(f"{mark} [{tag}] {s['file']}: {s['name']}")
        for p in problems:
            print(f"       {p}")
        failed += bool(problems)
    print(f"{len(scenarios) - failed}/{len(scenarios)} scenarios match the engine")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
