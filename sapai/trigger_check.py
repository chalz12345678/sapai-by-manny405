"""
Check that no pet stops another pet's ability from firing.

    python -m sapai.trigger_check

For every pet with an ability, its trigger situation is set up (selling it,
buying it, the start or end of a turn, start of battle, fainting, being hurt,
knocking out an enemy, a friend being summoned, ...) once next to a plain
filler pet and once next to each of the other pets, in front of it and
behind it. Every trigger in the engine is watched to see which abilities
fire. A pet whose ability fires next to the filler but not next to some
other pet is reported: either that other pet really prevents it (for example
Whale swallowing it before start of battle), or it is a bug like Duck not
triggering when sold behind a Pig.

Each case runs on several random seeds; it is reported when the ability
fires less often than it does next to the filler.
"""

import argparse
import collections
import contextlib
import sys

import numpy as np

from sapai.battle import Battle
from sapai.data import data
from sapai.pets import Pet
from sapai.player import Player
from sapai.scenarios import make_team
from sapai.shop import Shop
from sapai.teams import Team

FILLER = "pet-fish"
SEEDS = range(4)

PETS = sorted(
    k
    for k, p in data["pets"].items()
    if isinstance(p["tier"], int) and p.get("level1Ability")
)


def trigger_of(name):
    a = data["pets"][name]["level1Ability"]
    return a["trigger"], a["triggeredBy"]["kind"]


################################################################################
#### Watching triggers
################################################################################


def due_for(pet_obj, attr, args, kwargs):
    """
    Whether this call means pet_obj's ability is due: its trigger situation
    has really happened (it fainted, a friend fainted, it was hurt and lived,
    a friend was summoned, ...). Abilities that are due but do not fire are
    what the check looks for.
    """
    if attr not in TRACKED:
        return False
    ability = pet_obj.ability
    by = ability.get("triggeredBy", {}).get("kind")
    trigger = args[0] if args else kwargs.get("trigger")
    alive = pet_obj._health > 0
    if attr == "faint_trigger":
        oteam = kwargs.get("oteam", args[2] if len(args) > 2 else None)
        if by == "Self":
            return trigger is pet_obj
        return trigger is not pet_obj and alive and pet_obj.team is not oteam
    if attr == "friend_summoned_trigger":
        return trigger is not pet_obj and alive
    if attr == "hurt_trigger":
        return alive
    if attr == "after_attack_trigger":
        team = pet_obj.team
        if not alive or not team.check_friend(pet_obj):
            return False
        if by == "Self":
            return team.index(pet_obj) == 0
        ahead = team.get_ahead(pet_obj, n=1)
        return len(ahead) > 0 and team.index(ahead[0]) == 0
    if attr == "friend_hurt_trigger":
        return alive and trigger is not pet_obj and trigger._health > 0
    return True


### Triggers whose situations due_for can recognise; for the others (shop
###   situations) the situation itself guarantees the trigger is due
TRACKED = {
    "faint_trigger",
    "friend_summoned_trigger",
    "hurt_trigger",
    "after_attack_trigger",
    "before_attack_trigger",
    "sob_trigger",
    "knockout_trigger",
    "friend_hurt_trigger",
}


@contextlib.contextmanager
def watch():
    """
    Record the names of pets whose ability fires, as (pet, method), and the
    abilities that were due, as (pet, "due:" + method)
    """
    fired = []
    originals = {}
    for attr in dir(Pet):
        if not attr.endswith("_trigger"):
            continue
        original = getattr(Pet, attr)
        originals[attr] = original

        def wrapped(self, *args, _original=original, _attr=attr, **kwargs):
            if self.name != "pet-none" and due_for(self, _attr, args, kwargs):
                fired.append((self.name, "due:" + _attr))
            result = _original(self, *args, **kwargs)
            activated = result[0] if isinstance(result, tuple) else bool(result)
            if activated:
                fired.append((self.name, _attr))
            return result

        setattr(Pet, attr, wrapped)

    ### Tiger: the ability of the pet ahead runs again; tiger_func returns
    ###   [first targets, repeated targets] when it does
    import sapai.pets as pets_module

    original_tiger = pets_module.tiger_func

    def tiger_wrapped(func, te_fainted, *args):
        apet = args[0]
        team = apet.team
        if team is not None and team.battle:
            if team.check_friend(apet):
                behind = team.get_behind(apet, n=1)
            else:
                behind = [x for x in list(team)[args[1][1] + 1 :] if not x.empty][:1]
            if (
                behind
                and behind[0].pet.name == "pet-tiger"
                and behind[0].pet.health > 0
            ):
                fired.append((apet.name, "due:tiger_repeat"))
        result = original_tiger(func, te_fainted, *args)
        targets = result[0]
        if len(targets) == 2 and all(isinstance(t, list) for t in targets):
            fired.append((args[0].name, "tiger_repeat"))
        return result

    pets_module.tiger_func = tiger_wrapped
    try:
        yield fired
    finally:
        for attr, original in originals.items():
            setattr(Pet, attr, original)
        pets_module.tiger_func = original_tiger


def pet(name, attack=None, health=None, level=1):
    p = Pet(name)
    p.level = level
    if attack is not None:
        p._attack = attack
    if health is not None:
        p._health = health
    return p


################################################################################
#### Situations: each sets up the trigger for pet p next to pet q (q in front
#### when q_ahead) and returns whether p's ability fired
################################################################################


def fired_for(fired, name, method=None):
    """(whether name's ability fired, whether it was due)"""
    did = any(n == name and (method is None or m == method) for n, m in fired)
    if method is None or (method not in TRACKED and method != "tiger_repeat"):
        return did, True
    due = [m for n, m in fired if n == name and m == "due:" + method]
    if method == "friend_hurt_trigger":
        ### Wolverine fires on every 4th friend hurt
        return did, len(due) >= 4
    return did, len(due) > 0


def shop_sell(p, q, q_ahead):
    team = [pet(q), pet(p)] if q_ahead else [pet(p), pet(q)]
    player = Player(team=Team(team))
    target = [s.pet for s in player.team if s.pet.name == p][0]
    with watch() as fired:
        player.sell(target)
    return fired_for(fired, p, "sell_trigger")


def shop_buy(p, q, q_ahead):
    player = Player(shop=Shop([p.replace("pet-", "")]), team=Team([pet(q)]))
    with watch() as fired:
        player.buy_pet(0)
    return fired_for(fired, p)


def shop_turn(method, lf_winner=True, q_level=1):
    def run(p, q, q_ahead):
        team = (
            [pet(q, level=q_level), pet(p)]
            if q_ahead
            else [pet(p), pet(q, level=q_level)]
        )
        player = Player(team=Team(team))
        player.lf_winner = lf_winner
        with watch() as fired:
            if method == "sot_trigger":
                player.start_turn(winner=lf_winner)
            else:
                player.end_turn()
        return fired_for(fired, p, method)

    return run


def shop_eat(feed_self):
    def run(p, q, q_ahead):
        team = [pet(q), pet(p)] if q_ahead else [pet(p), pet(q)]
        player = Player(shop=Shop(["apple"]), team=Team(team))
        names = [s.pet.name for s in player.team]
        target = names.index(p if feed_self else q)
        player.gold = 10
        with watch() as fired:
            player.buy_food(0, target)
        return fired_for(fired, p)

    return run


def shop_buy_friend(p, q, q_ahead):
    """Dragon: buying a tier 1 pet (an Ant) with p and q on the team"""
    team = [pet(q), pet(p)] if q_ahead else [pet(p), pet(q)]
    player = Player(shop=Shop(["ant"]), team=Team(team))
    with watch() as fired:
        player.buy_pet(0)
    return fired_for(fired, p)


def shop_levelup(p, q, q_ahead):
    team = [pet(q), pet(p), pet(p)] if q_ahead else [pet(p), pet(p), pet(q)]
    a = pet(p)
    a.experience = 1
    team[team.index([x for x in team if x.name == p][0])] = a
    player = Player(team=Team(team))
    idx = [i for i, s in enumerate(player.team) if s.pet.name == p]
    with watch() as fired:
        player.combine(idx[0], idx[1])
    return fired_for(fired, p, "levelup_trigger")


def battle_fired(t0, t1, name, method=None, seed=0):
    np.random.seed(seed)
    with watch() as fired:
        Battle(t0, t1).battle()
    return fired_for(fired, name, method)


def battle_sob(p, q, q_ahead, seed=0):
    team = [pet(q, 2, 20), pet(p, 2, 20)] if q_ahead else [pet(p, 2, 20), pet(q, 2, 20)]
    return battle_fired(
        Team(team), make_team(["Pig 1/50", "Pig 1/50"]), p, "sob_trigger", seed
    )


def battle_faint_together(p, q, q_ahead, seed=0):
    """p and q (both 1/1) are hit in the same attack by a Chili attacker"""
    team = [pet(q, 1, 1), pet(p, 1, 1)] if q_ahead else [pet(p, 1, 1), pet(q, 1, 1)]
    return battle_fired(
        Team(team), make_team(["Pig 50/50 Chili"]), p, "faint_trigger", seed
    )


def battle_faint_front(p, q, q_ahead, seed=0):
    """p faints at the front; q is behind it"""
    team = [pet(p, 1, 1), pet(q, 1, 30)]
    return battle_fired(Team(team), make_team(["Pig 5/50"]), p, "faint_trigger", seed)


def battle_friend_faints(p, q, q_ahead, seed=0):
    """q faints at the front; p (Shark, Fly, Ox) is right behind it"""
    team = [pet(q, 1, 1), pet(p, 1, 40)]
    return battle_fired(Team(team), make_team(["Pig 5/50"]), p, "faint_trigger", seed)


def battle_hurt(p, q, q_ahead, seed=0):
    """p is hit and survives: at the front, or second by Chili splash"""
    team = [pet(q, 1, 30), pet(p, 1, 30)] if q_ahead else [pet(p, 1, 30), pet(q, 1, 30)]
    return battle_fired(
        Team(team), make_team(["Pig 1/50 Chili"]), p, "hurt_trigger", seed
    )


def battle_attack(method):
    def run(p, q, q_ahead, seed=0):
        if method == "after_attack_trigger" and trigger_of(p)[1] == "FriendAhead":
            team = [pet(q, 1, 40), pet(p, 1, 40)]
        else:
            team = [pet(p, 1, 40), pet(q, 1, 40)]
        return battle_fired(Team(team), make_team(["Pig 1/50"]), p, method, seed)

    return run


def battle_knockout(p, q, q_ahead, seed=0):
    team = [pet(p, 10, 40), pet(q, 1, 40)]
    return battle_fired(
        Team(team), make_team(["Fish 1/1", "Fish 1/1"]), p, "knockout_trigger", seed
    )


def battle_summoned(p, q, q_ahead, seed=0):
    """A Cricket faints at the front and summons a Zombie Cricket"""
    team = [pet("pet-cricket", 1, 1), pet(q, 1, 40), pet(p, 1, 40)]
    if q == "pet-cricket":
        return None
    return battle_fired(
        Team(team), make_team(["Pig 5/50"]), p, "friend_summoned_trigger", seed
    )


def battle_wolverine(p, q, q_ahead, seed=0):
    """q is hurt every round (at the front, or second by Chili splash)"""
    if q_ahead:
        team, enemy = [pet(q, 1, 12), pet(p, 1, 40)], "Pig 1/50"
    else:
        team, enemy = [pet(p, 1, 40), pet(q, 1, 12)], "Pig 1/50 Chili"
    return battle_fired(Team(team), make_team([enemy]), p, "friend_hurt_trigger", seed)


################################################################################
#### Chains: the trigger comes from another pet's ability, with a third pet r
#### in front of them or behind them
################################################################################


def chain(team_for, enemy, method):
    def run(p, r, r_ahead, seed=0):
        team = team_for(p, r, r_ahead)
        if team is None:
            return None
        return battle_fired(Team(team), make_team(enemy), p, method, seed)

    run.battle = True
    return run


def with_r(pets, r, r_ahead):
    return [r] + pets if r_ahead else pets + [r]


def hedgehog_then(p_health):
    """A Hedgehog faints at the front and deals 2 damage to p behind it"""

    def team_for(p, r, r_ahead):
        if r == "pet-hedgehog":
            return None
        pair = [pet("pet-hedgehog", 1, 1), pet(p, 1, p_health)]
        return with_r(pair, pet(r, 1, 40), r_ahead)

    return team_for


def shot_by_dolphin(p_health):
    """An enemy Dolphin shoots p (the lowest health pet) at start of battle"""

    def team_for(p, r, r_ahead):
        return with_r([pet(p, 1, p_health)], pet(r, 1, 40), r_ahead)

    return team_for


def sheep_then(p, r, r_ahead):
    """A Sheep faints at the front and summons two Rams"""
    if r == "pet-sheep":
        return None
    return with_r([pet("pet-sheep", 1, 1), pet(p, 1, 40)], pet(r, 1, 40), r_ahead)


def tiger_team(front=None, p_stats=(2, 20)):
    """Tiger directly behind p (and front ahead of p, if given)"""

    def team_for(p, r, r_ahead):
        if r == "pet-tiger" or r == front:
            return None
        group = [pet(p, *p_stats), pet("pet-tiger", 1, 40)]
        if front is not None:
            group = [pet(front, 1, 1)] + group
        return with_r(group, pet(r, 1, 40), r_ahead)

    return team_for


TIGER = {
    ("StartOfBattle", "Self"): (tiger_team(), ["Pig 1/50", "Pig 1/50"]),
    ("Faint", "Self"): (tiger_team(p_stats=(1, 1)), ["Pig 5/50"]),
    ("Faint", "EachFriend"): (tiger_team("pet-fish", (1, 40)), ["Pig 5/50"]),
    ("Faint", "FriendAhead"): (tiger_team("pet-fish", (1, 40)), ["Pig 5/50"]),
    ("Hurt", "Self"): (tiger_team(p_stats=(1, 30)), ["Pig 1/50"]),
    ("BeforeAttack", "Self"): (tiger_team(p_stats=(1, 40)), ["Pig 1/50"]),
    ("AfterAttack", "Self"): (tiger_team(p_stats=(1, 40)), ["Pig 1/50"]),
    ("AfterAttack", "FriendAhead"): (tiger_team("pet-fish", (1, 40)), ["Pig 1/50"]),
    ("KnockOut", "Self"): (tiger_team(p_stats=(10, 40)), ["Fish 1/1", "Fish 1/1"]),
    ("Summoned", "EachFriend"): (tiger_team("pet-cricket", (1, 40)), ["Pig 5/50"]),
}

SITUATIONS = {
    ("Sell", "Self"): [("sold", shop_sell)],
    ("Buy", "Self"): [("bought", shop_buy)],
    ("StartOfTurn", "Self"): [("start of turn", shop_turn("sot_trigger"))],
    ("EndOfTurn", "Self"): [("end of turn", shop_turn("eot_trigger"))],
    ("EndOfTurnAfterLoss", "Self"): [
        ("end of turn after a loss", shop_turn("eot_trigger", lf_winner=False))
    ],
    ("EndOfTurnWithLvl3Friend", "Self"): [
        ("end of turn, level 3 friend", shop_turn("eot_trigger", q_level=3))
    ],
    ("EatsShopFood", "Self"): [("eats food", shop_eat(True))],
    ("EatsShopFood", "EachFriend"): [("friend eats food", shop_eat(False))],
    ("PurchaseFood", "Self"): [("food bought", shop_eat(False))],
    ("BuyTier1Animal", "Self"): [("tier 1 pet bought", shop_buy_friend)],
    ("LevelUp", "Self"): [("levels up", shop_levelup)],
    ("StartOfBattle", "Self"): [("start of battle", battle_sob)],
    ("Faint", "Self"): [
        ("faints at the front", battle_faint_front),
        ("faints with a friend", battle_faint_together),
        (
            "knocked out by Hedgehog's faint",
            chain(hedgehog_then(1), ["Pig 5/50"], "faint_trigger"),
        ),
        (
            "knocked out by an enemy Dolphin",
            chain(shot_by_dolphin(2), ["Dolphin 1/50"], "faint_trigger"),
        ),
    ],
    ("Faint", "EachFriend"): [("friend faints", battle_friend_faints)],
    ("Faint", "FriendAhead"): [("friend ahead faints", battle_friend_faints)],
    ("Hurt", "Self"): [
        ("hurt", battle_hurt),
        (
            "hurt by Hedgehog's faint",
            chain(hedgehog_then(30), ["Pig 5/50"], "hurt_trigger"),
        ),
        (
            "hurt by an enemy Dolphin",
            chain(shot_by_dolphin(30), ["Dolphin 1/50"], "hurt_trigger"),
        ),
    ],
    ("FriendHurt", "EachFriend"): [("friend hurt", battle_wolverine)],
    ("BeforeAttack", "Self"): [
        ("before attacking", battle_attack("before_attack_trigger"))
    ],
    ("AfterAttack", "Self"): [
        ("after attacking", battle_attack("after_attack_trigger"))
    ],
    ("AfterAttack", "FriendAhead"): [
        ("friend ahead attacks", battle_attack("after_attack_trigger"))
    ],
    ("KnockOut", "Self"): [("knocks out", battle_knockout)],
    ("Summoned", "EachFriend"): [
        ("friend summoned", battle_summoned),
        (
            "Rams summoned by Sheep",
            chain(sheep_then, ["Pig 5/50"], "friend_summoned_trigger"),
        ),
    ],
}

BATTLE = {
    battle_sob,
    battle_faint_together,
    battle_faint_front,
    battle_friend_faints,
    battle_hurt,
    battle_knockout,
    battle_summoned,
    battle_wolverine,
}

for _key, (_team_for, _enemy) in TIGER.items():
    SITUATIONS.setdefault(_key, []).append(
        ("repeated by Tiger behind it", chain(_team_for, _enemy, "tiger_repeat"))
    )


def rate(situation, p, q, q_ahead):
    """
    Share of the seeds where p's ability was due on which it fired, None if
    the situation does not apply or never happened, or an error message
    """
    is_battle = (
        situation in BATTLE
        or getattr(situation, "battle", False)
        or "battle" in getattr(situation, "__qualname__", "")
    )
    due_seeds = fired_seeds = 0
    for seed in SEEDS:
        np.random.seed(seed)
        try:
            r = (
                situation(p, q, q_ahead, seed=seed)
                if is_battle
                else situation(p, q, q_ahead)
            )
        except Exception as e:
            return f"error: {type(e).__name__}: {str(e)[:60]}"
        if r is None:
            return None
        did, due = r if isinstance(r, tuple) else (bool(r), True)
        if due:
            due_seeds += 1
            fired_seeds += bool(did)
    if due_seeds == 0:
        return None
    return fired_seeds / due_seeds


### Pets that really stop another pet's ability: Whale swallows the friend
###   ahead at start of battle, so it never uses its start of battle ability
###   and never gets to knock anything out
EXPECTED = {"pet-whale"}


def run(pets=None, others=None):
    """
    pets: the pets whose abilities are checked (default: all)
    others: the pets put next to them (default: all)

    Returns (problems, checked): problems is a list of
    (pet, situation, other pet, in front or behind, rate with the other pet,
    rate with the filler); checked counts the cases run.
    """
    problems = []
    checked = 0
    for p in pets or PETS:
        for label, situation in SITUATIONS.get(trigger_of(p), []):
            for q_ahead in (True, False):
                base = rate(situation, p, FILLER, q_ahead)
                if base is None:
                    ### The filler arrangement does not apply (for example
                    ###   no friend ahead); the other pets are still checked
                    base = 1.0
                elif not isinstance(base, float) or base == 0:
                    problems.append((p, label, FILLER, q_ahead, base, base))
                    continue
                for q in others or PETS:
                    if q in (p, FILLER):
                        continue
                    r = rate(situation, p, q, q_ahead)
                    if r is None:
                        continue
                    checked += 1
                    if not isinstance(r, float) or r < base:
                        problems.append((p, label, q, q_ahead, r, base))
    return problems, checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("pets", nargs="*", help="pets to check (default: all)")
    args = parser.parse_args(argv)
    pets = [x if x.startswith("pet-") else "pet-" + x for x in args.pets] or None
    problems, checked = run(pets)
    print(f"{checked} cases checked")
    expected = [x for x in problems if x[2] in EXPECTED]
    problems = [x for x in problems if x[2] not in EXPECTED]
    if expected:
        print(f"{len(expected)} cases where Whale swallows the pet first (expected)")
    if not problems:
        print("No pet stops another pet's ability")
    by = collections.defaultdict(list)
    for p, label, q, q_ahead, r, base in problems:
        by[(p, label)].append((q, q_ahead, r, base))
    for (p, label), rows in sorted(by.items()):
        print(f"\n{p[4:]} ({label}):")
        for q, q_ahead, r, base in rows:
            where = "in front" if q_ahead else "behind"
            print(f"  next to {q[4:]} {where}: fired {r} (with filler: {base})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
