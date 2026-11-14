"""
Shop actions as a fixed, numbered list, with the legal ones for a player and
a way to apply one. Used by the training environment, the stress tests and
anything else that plays the shop (no gymnasium needed).

    from sapai.actions import ACTIONS, END_TURN, action_mask, apply_action

    mask = action_mask(player)          # which of the N_ACTIONS are legal
    apply_action(player, 5)             # anything but END_TURN

Ending the turn is not done here: whoever runs the game ends it (for example
Match.battle()).
"""

import itertools

import numpy as np

### These match the observation layout of the trained models, so they are
###   fixed here rather than read from the rules
MAX_TEAM = 5
MAX_SHOP = 7

### Action layout: name -> (start index, list of arguments for each action)
PAIRS = list(itertools.combinations(range(MAX_TEAM), 2))
_ACTION_ARGS = [
    ("end_turn", [()]),
    ("roll", [()]),
    ("buy_pet", [(i,) for i in range(MAX_SHOP)]),
    ("buy_combine", [(i, j) for i in range(MAX_SHOP) for j in range(MAX_TEAM)]),
    ("buy_food", [(i, j) for i in range(MAX_SHOP) for j in range(MAX_TEAM)]),
    ("buy_food_untargeted", [(i,) for i in range(MAX_SHOP)]),
    ("sell", [(j,) for j in range(MAX_TEAM)]),
    ("freeze", [(i,) for i in range(MAX_SHOP)]),
    ("swap", PAIRS),
    ("combine", PAIRS),
]
ACTIONS = []
ACTION_START = {}
for _name, _args in _ACTION_ARGS:
    ACTION_START[_name] = len(ACTIONS)
    ACTIONS += [(_name, a) for a in _args]
N_ACTIONS = len(ACTIONS)
END_TURN = ACTION_START["end_turn"]


def food_needs_target(food):
    """Whether the player chooses which pet eats the food"""
    effect = food.effect
    target = effect.get("target", effect.get("to", {}))
    return target.get("kind") == "PurchaseTarget"


def team_pets(player):
    return [slot.pet for slot in player.team if not slot.empty]


def action_mask(player):
    """Boolean mask of legal actions for the player's current shop phase"""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[END_TURN] = True
    gold = player.gold
    team = team_pets(player)
    shop = player.shop.slots
    if gold >= 1:
        mask[ACTION_START["roll"]] = True

    for i, slot in enumerate(shop[:MAX_SHOP]):
        if slot.empty:
            continue
        mask[ACTION_START["freeze"] + i] = True
        affordable = slot.cost <= gold
        if slot.slot_type in ["pet", "levelup"]:
            if affordable and len(team) < MAX_TEAM:
                mask[ACTION_START["buy_pet"] + i] = True
            for j, pet in enumerate(team):
                if affordable and pet.name == slot.obj.name and pet.level < 3:
                    mask[ACTION_START["buy_combine"] + i * MAX_TEAM + j] = True
        elif slot.slot_type == "food" and affordable:
            if food_needs_target(slot.obj):
                for j in range(len(team)):
                    mask[ACTION_START["buy_food"] + i * MAX_TEAM + j] = True
            else:
                mask[ACTION_START["buy_food_untargeted"] + i] = True

    for j in range(len(team)):
        mask[ACTION_START["sell"] + j] = True
    for k, (a, b) in enumerate(PAIRS):
        if b < len(team):
            mask[ACTION_START["swap"] + k] = True
            if team[a].name == team[b].name and team[a].level < 3:
                mask[ACTION_START["combine"] + k] = True
    return mask


def apply_action(player, action):
    """Apply one shop action (anything but end_turn) to the player"""
    name, args = ACTIONS[action]
    if name == "roll":
        player.roll()
    elif name == "buy_pet":
        player.buy_pet(args[0])
    elif name == "buy_combine":
        player.buy_combine(args[0], args[1])
    elif name == "buy_food":
        player.buy_food(args[0], args[1])
    elif name == "buy_food_untargeted":
        player.buy_food(args[0])
    elif name == "sell":
        player.sell(args[0])
    elif name == "freeze":
        if player.shop[args[0]].frozen:
            player.unfreeze(args[0])
        else:
            player.freeze(args[0])
    elif name == "swap":
        a, b = args
        order = list(range(len(player.team)))
        order[a], order[b] = order[b], order[a]
        player.reorder(order)
    elif name == "combine":
        player.combine(args[0], args[1])
    else:
        raise ValueError(f"Not a shop action: {name}")
    ### Keep the team compact so team slot j is the j-th pet from the front
    player.team.move_forward()
