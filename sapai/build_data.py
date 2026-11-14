"""
Build the engine's game data from the pack file.

    python -m sapai.build_data            # rebuild sapai/game_data.json
    python -m sapai.build_data --check    # fail if game_data.json is out of date

data/turtle_pack.yaml is the single source of truth for pets, foods, perks
(statuses) and the per-turn rules. Edit it when the game is patched, then run
this command and the tests (see docs/PATCHING.md). The engine itself only
reads sapai/game_data.json, so playing needs no YAML library; building does
(pip install pyyaml).
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK_PATH = os.path.join(HERE, "..", "data", "turtle_pack.yaml")
OUT_PATH = os.path.join(HERE, "game_data.json")


def load_pack(path=PACK_PATH):
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


################################################################################
#### Building
################################################################################


def shop_probabilities(items, turns, slots_key, pack):
    """
    Every shop slot is drawn uniformly from the items of the tiers available
    that turn. items: {id: tier}
    """
    probs = {k: [] for k in items}
    for turn_id, td in turns.items():
        avail = [k for k, tier in items.items() if tier <= td["tiersAvailable"]]
        p_slot = 1 / len(avail)
        p_shop = 1 - (1 - p_slot) ** td[slots_key]
        for k in avail:
            probs[k].append(
                {
                    "kind": "shop",
                    "turn": turn_id,
                    "perShop": {pack: p_shop},
                    "perSlot": {pack: p_slot},
                }
            )
    return probs


def build(pack_data):
    """The engine's data dict from the pack file's contents"""
    pack = pack_data["pack"]["internal_id"]

    turns = {}
    for t in pack_data["rules"]["turns"]:
        i = t["index"]
        turns[f"turn-{i}"] = {"name": f"Turn {i}", "id": f"turn-{i}"}
        turns[f"turn-{i}"].update(t)

    shop_pets = pack_data["pets"]
    pet_probs = shop_probabilities(
        {p["id"]: p["tier"] for p in shop_pets}, turns, "animalShopSlots", pack
    )
    pets = {}
    for p in shop_pets + pack_data["summoned_pets"]:
        entry = {"name": p["name"], "id": p["id"]}
        if "image" in p:
            entry["image"] = p["image"]
        entry["tier"] = p["tier"]
        entry["baseAttack"] = p["attack"]
        entry["baseHealth"] = p["health"]
        entry["packs"] = [pack]
        if "status" in p:
            entry["status"] = p["status"]
        if "levels" in p:
            for level in (1, 2, 3):
                lv = dict(p["levels"][level])
                ability = {
                    "description": lv.pop("text"),
                    "trigger": p["trigger"],
                    "triggeredBy": p["triggered_by"],
                    "effect": lv.pop("effect"),
                }
                ability.update(lv)
                entry[f"level{level}Ability"] = ability
        if p["id"] in pet_probs:
            entry["probabilities"] = pet_probs[p["id"]]
        pets[p["id"]] = entry

    food_probs = shop_probabilities(
        {f["id"]: f["tier"] for f in pack_data["foods"]}, turns, "foodShopSlots", pack
    )
    foods = {}
    for f in pack_data["foods"] + pack_data["pet_created_foods"]:
        entry = {"name": f["name"], "id": f["id"]}
        if "image" in f:
            entry["image"] = f["image"]
        if "notes" in f:
            entry["notes"] = f["notes"]
        entry["tier"] = f["tier"]
        entry["packs"] = [pack]
        entry["cost"] = f["cost"]
        entry["ability"] = {
            "description": f["text"],
            "triggeredBy": {"kind": "Self"},
            "trigger": "Buy",
            "effect": f["effect"],
        }
        if "perk" in f:
            entry["perk"] = f["perk"]
        if f["id"] in food_probs:
            entry["probabilities"] = food_probs[f["id"]]
        foods[f["id"]] = entry

    statuses = {}
    for s in pack_data["statuses"]:
        entry = {"name": s["name"], "id": s["id"]}
        if "image" in s:
            entry["image"] = s["image"]
        ability = {
            "description": s["text"],
            "triggeredBy": s["triggered_by"],
            "trigger": s["trigger"],
            "effect": s["effect"],
        }
        ability.update(
            {
                k: v
                for k, v in s.items()
                if k not in ("id", "name", "image", "text", "trigger", "triggered_by")
                and k != "effect"
            }
        )
        entry["ability"] = ability
        statuses[s["id"]] = entry

    rules = {k: v for k, v in pack_data["rules"].items() if k != "turns"}
    return {
        "pack": pack_data["pack"],
        "rules": rules,
        "pets": pets,
        "foods": foods,
        "statuses": statuses,
        "turns": turns,
    }


################################################################################
#### Checks
################################################################################


def _walk(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from _walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v)


def validate(data):
    """
    Problems with the built data, as a list of strings: references to pets,
    foods or perks that do not exist, missing fields and bad values.
    """
    problems = []
    pets, foods, statuses = data["pets"], data["foods"], data["statuses"]
    for kind, items in [("pet", pets), ("food", foods), ("status", statuses)]:
        for key, item in items.items():
            if item["id"] != key:
                problems.append(f"{kind} {key}: id does not match its key")
    for key, p in pets.items():
        if p["tier"] != "Summoned" and not 1 <= p["tier"] <= 6:
            problems.append(f"{key}: tier {p['tier']} is not 1-6")
        for stat in ("baseAttack", "baseHealth"):
            ### "?" for summoned pets whose stats are set by their summoner
            if p[stat] == "?" and p["tier"] == "Summoned":
                continue
            if not isinstance(p[stat], int) or not 0 <= p[stat] <= 50:
                problems.append(f"{key}: {stat} {p[stat]} is not 0-50")
        if p.get("status") and p["status"] not in statuses:
            problems.append(f"{key}: unknown status {p['status']}")

    ### Every pet, food and status an effect refers to must exist
    sources = [(k, p) for k, p in pets.items()]
    sources += [(k, f) for k, f in foods.items()]
    sources += [(k, s) for k, s in statuses.items()]
    for key, item in sources:
        for d in _walk(item):
            if d.get("kind") in ("SummonPet",) and "pet" in d:
                if d["pet"] not in pets:
                    problems.append(f"{key}: summons unknown pet {d['pet']}")
            if d.get("kind") == "StockFood" and d.get("food") not in foods:
                problems.append(f"{key}: stocks unknown food {d.get('food')}")
            if d.get("kind") == "ApplyStatus" and d.get("status") not in statuses:
                problems.append(f"{key}: applies unknown status {d.get('status')}")
    for i, t in enumerate(data["turns"].values(), start=1):
        if t["index"] != i:
            problems.append(f"turn {t['id']}: index {t['index']} is out of order")
    return problems


def engine_support():
    """
    The triggers, effect kinds and target kinds the engine implements, read
    from its source so the lists cannot go stale. A trigger counts when the
    engine compares an ability's trigger to it; Tiger's and Cat's passives
    (CastsAbility, PurchaseFood) are handled by pet name.
    """
    import inspect

    import sapai.effects
    import sapai.pets
    import sapai.player
    import sapai.status

    source = "".join(
        inspect.getsource(m)
        for m in (sapai.pets, sapai.player, sapai.status, sapai.effects)
    )
    triggers = set(re.findall(r'\["trigger"\] *[!=]= *"(\w+)"', source))
    triggers |= {"CastsAbility", "PurchaseFood", "none"}
    ### Perks: how a status is used is decided by the status itself
    triggers |= {"WhenAttacking", "WhenDamaged"}
    effects = {
        name
        for name, f in inspect.getmembers(sapai.effects, inspect.isfunction)
        if f.__module__ == "sapai.effects" and name[0].isupper() or name == "none"
    }
    targets = set(
        re.findall(r'kind == "(\w+)"', inspect.getsource(sapai.effects.get_target))
    )
    return triggers, effects, targets


def _engine_source():
    import inspect

    import sapai.battle
    import sapai.effects
    import sapai.pets
    import sapai.player
    import sapai.status

    return "".join(
        inspect.getsource(m)
        for m in (sapai.battle, sapai.effects, sapai.pets, sapai.player, sapai.status)
    )


def unsupported(data):
    """
    Triggers, effects and targets in the data that the engine does not run.

    Pet abilities run through sapai.effects, so their triggers, effect kinds
    and targets must be ones it implements. Foods that only change stats or
    give a perk are applied generically; any other food is handled by name,
    so its id must appear in the engine. Perks are handled by status id.
    """
    triggers, effects, targets = engine_support()
    source = _engine_source()
    problems = []

    def check_effect(name, effect):
        for d in _walk(effect):
            if "kind" not in d:
                continue
            ### Effects sit at the top or in an AllOf/OneOf list; any other
            ###   dict with a kind is a target (or a "from"/"to" selector)
            is_effect = d is effect or any(
                d is e for x in _walk(effect) for e in x.get("effects", [])
            )
            if is_effect and d["kind"] not in effects:
                problems.append(f"{name}: effect {d['kind']} is not implemented")
            elif not is_effect and d["kind"] not in targets | effects:
                problems.append(f"{name}: target {d['kind']} is not implemented")

    for key, p in data["pets"].items():
        for level in (1, 2, 3):
            ability = p.get(f"level{level}Ability")
            if not ability:
                continue
            name = f"{key} level {level}"
            if ability["trigger"] not in triggers:
                problems.append(
                    f"{name}: trigger {ability['trigger']} is not implemented"
                )
            check_effect(name, ability["effect"])

    for key, f in data["foods"].items():
        effect = f["ability"]["effect"]
        allowed = targets | {"PurchaseTarget", "ModifyStats", "ApplyStatus"}
        generic = effect["kind"] in ("ModifyStats", "ApplyStatus") and all(
            d["kind"] in allowed for d in _walk(effect) if "kind" in d
        )
        if not generic and f'"{key}"' not in source:
            problems.append(
                f"{key}: effect {effect['kind']} is not a stat change or perk,"
                " and the engine has no code for this food"
            )

    for key in data["statuses"]:
        if f'"{key}"' not in source:
            problems.append(f"{key}: the engine has no code for this perk")
    return problems


NUMBER = re.compile(r"(?<![\w.])(\d+)(?:%)?")
WORD_NUMBERS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


### Phrases whose numbers describe a condition rather than an amount
IGNORED_PHRASES = re.compile(
    r"\((?:[^)]*)\)"  # notes in brackets
    r"|\blevel \d\b|\btier \d\b"
    r"|\bTriggers 1 time\b"
    r"|\bone\b",
    re.IGNORECASE,
)


def text_numbers(text):
    """Amounts in an ability's text, including the words two to six"""
    text = IGNORED_PHRASES.sub(" ", text)
    found = {int(x) for x in NUMBER.findall(text)}
    for word, n in WORD_NUMBERS.items():
        if re.search(rf"\b{word}\b", text, re.IGNORECASE):
            found.add(n)
    return found


def effect_numbers(effect):
    found = set()
    for d in _walk(effect):
        for k, v in d.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                found.add(int(v))
            if k == "effects" and isinstance(v, list):
                found.add(len(v))
    return found


def text_mismatches(data):
    """
    Pet abilities whose text has a number that the effect does not use. When
    a patch changes a number in the text but not in the effect (or the other
    way round), it shows up here. Returns {pet id: {level: missing numbers}}.
    """
    out = {}
    for key, p in data["pets"].items():
        for level in (1, 2, 3):
            ability = p.get(f"level{level}Ability")
            if not ability:
                continue
            numbers = effect_numbers(ability["effect"])
            numbers |= {
                v
                for k, v in ability.items()
                if k in ("maxTriggers", "everyN") and isinstance(v, int)
            }
            ### Summoned pets' stats count too (e.g. "Summon one 2/2 ...")
            for d in _walk(ability["effect"]):
                if d.get("kind") == "SummonPet" and d.get("pet") in data["pets"]:
                    summoned = data["pets"][d["pet"]]
                    numbers |= {summoned["baseAttack"], summoned["baseHealth"]}
            missing = text_numbers(ability["description"]) - numbers
            if missing:
                out.setdefault(key, {})[level] = sorted(missing)
    return out


################################################################################
#### Writing
################################################################################


def dumps(data):
    return json.dumps(data, indent=1, ensure_ascii=False) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--pack", default=PACK_PATH)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument(
        "--check", action="store_true", help="fail if the output is out of date"
    )
    args = parser.parse_args(argv)

    data = build(load_pack(args.pack))
    problems = validate(data) + unsupported(data)
    if problems:
        print("Problems in the pack file:")
        for p in problems:
            print("  " + p)
        return 1
    text = dumps(data)
    if args.check:
        with open(args.out, encoding="utf-8") as f:
            if f.read() != text:
                print(f"{args.out} is out of date: run python -m sapai.build_data")
                return 1
        print(f"{args.out} is up to date")
        return 0
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(
        f"Wrote {os.path.normpath(args.out)}: {len(data['pets'])} pets,"
        f" {len(data['foods'])} foods, {len(data['statuses'])} statuses,"
        f" {len(data['turns'])} turns"
    )
    mismatches = text_mismatches(data)
    if mismatches:
        print("Ability text numbers not found in the effects (check these):")
        for key, levels in mismatches.items():
            print(f"  {key}: {levels}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
