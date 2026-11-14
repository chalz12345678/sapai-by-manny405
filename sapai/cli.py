"""
Play a game of Super Auto Pets (Turtle Pack) in the terminal.

    python -m sapai.cli
    python -m sapai.cli --seed 1

Each turn you shop: buy, sell, roll, freeze and rearrange pets. Then you end the
turn and your team battles a computer opponent that shops with the same rules.
Each lost battle costs a life; make your opponent run out of lives first.

"""

import argparse
import os
import random
import re
import shlex

import numpy as np

from sapai.battle import Battle
from sapai.data import data, RULES
from sapai.match import Match, lives_lost
from sapai.teams import Team
from sapai.foods import Food
from sapai.pets import Pet
from sapai.player import Player

HELP = """Commands (numbers are the slot numbers shown in the shop and team):
  buy S          buy shop pet S, or shop food S if it targets random pets
  buy S T        feed shop food S to team pet T, or merge shop pet S into
                 team pet T if they are the same pet
  sell T         sell team pet T
  roll           reroll the shop (1 gold)
  freeze S       freeze or unfreeze shop slot S (frozen slots are kept)
  move T1 T2     move team pet T1 to position T2 (0 is the front)
  combine T1 T2  merge team pet T2 into team pet T1 (same pet)
  info NAME      show a pet or food, for example: info ant
  end            end the turn and battle
  log            show the step by step log of the last battle
  help           show this help
  quit           quit the game"""

XP_TO_LEVEL = {1: 2, 2: 3}


def display_name(name):
    for kind in ["pets", "foods", "statuses"]:
        if name in data[kind]:
            return data[kind][name]["name"]
    return name


PET_REPR = re.compile(r"< (pet-[a-z-]+) (-?\d+)-(-?\d+) (\S+) (\d)-\d+ >")


def pretty(text):
    """Turn '< pet-ant 2-1 none 1-0 >' into 'Ant 2/1'"""

    def repl(match):
        name, attack, health = match.group(1), match.group(2), match.group(3)
        return f"{display_name(name)} {attack}/{health}"

    return PET_REPR.sub(repl, str(text))


EFFECT_TEXT = {
    "AllOf": "uses its ability on",
    "ApplyStatus": "gives a perk to",
    "DealDamage": "deals damage to",
    "GainGold": "gains gold",
    "ModifyStats": "buffs",
    "ReduceHealth": "removes health from",
    "RemoveHealth": "removes health from",
    "RepeatAbility": "repeats an ability on",
    "RespawnPet": "comes back as",
    "SummonPet": "summons",
    "SummonRandomPet": "summons",
    "Swallow": "swallows",
    "TransferAbility": "copies the ability of",
    "TransferStats": "copies stats to",
}


def pet_parts(text):
    """All pets in a battle log string as (name, attack, health)"""
    return [(display_name(m[0]), m[1], m[2]) for m in PET_REPR.findall(str(text))]


def side(team_idx):
    return "You" if int(team_idx) == 0 else "Opponent"


def log_line(entry):
    """Readable text for one battle history entry, or None to skip it"""
    if len(entry) < 4 or type(entry[0]) != str:
        return None
    what, where, pet, targets = entry[0], entry[1], entry[2], entry[3]
    extra = entry[4] if len(entry) > 4 else {}
    pet = pet_parts(pet)
    if len(pet) == 0:
        return None
    name = pet[0][0]

    if what == "Attack":
        before = [pet_parts(x)[0] for x in extra.get("before", [entry[2], targets[0]])]
        after = [pet[0], pet_parts(targets[0])[0]]
        damage = extra.get("damage")
        text = (
            f"Your {before[0][0]} {before[0][1]}/{before[0][2]} and their"
            + f" {before[1][0]} {before[1][1]}/{before[1][2]} attack each other"
        )
        if damage is not None:
            text += (
                f": yours takes {damage[0]} (now {after[0][1]}/{after[0][2]}),"
                + f" theirs takes {damage[1]} (now {after[1][1]}/{after[1][2]})"
            )
        return text

    if what == "splash":
        target = pet_parts(targets[0])[0]
        damage = extra.get("damage", [5])[0]
        return (
            f"{side(where[0])}: {name}'s Chili hits {target[0]} for {damage}"
            + f" (now {target[1]}/{target[2]})"
        )

    if what == "Fainted":
        return f"{side(where[0])}: {name} faints"

    verb = EFFECT_TEXT.get(what, f"uses {what} on")
    hit = []
    for target in targets:
        hit += [f"{n} (now {a}/{h})" for n, a, h in pet_parts(target)]
    if what == "GainGold":
        return f"{side(where[0])}: {name} {verb}"
    return f"{side(where[0])}: {name} {verb} " + (", ".join(hit) or "nothing")


def food_needs_target(food):
    """Whether the player picks which pet eats the food"""
    effect = food.effect
    target = effect.get("target", effect.get("to", {}))
    return target.get("kind") == "PurchaseTarget"


def describe_pet(pet):
    stats = f"{pet.attack}/{pet.health}"
    if pet.level < 3:
        level = f"Lv{pet.level} ({pet.experience}/{XP_TO_LEVEL[pet.level]} xp)"
    else:
        level = "Lv3"
    perk = ""
    if pet.status != "none":
        perk = f"  [{display_name(pet.status)}]"
    return f"{display_name(pet.name):<14} {stats:>6}  {level:<14}{perk}"


class Opponent:
    """
    Computer opponent. Shops with the same gold and shop rules as the player:
    merges duplicates, fills its team, then feeds food, and rolls when it has
    nothing useful to buy.
    """

    def __init__(self, rng):
        self.rng = rng
        self.player = Player()

    def shop_turn(self, view=None):
        p = self.player
        for _ in range(40):
            if p.gold <= 0:
                break
            if not self.take_action():
                if p.gold >= 1 and self.rng.random() < 0.5:
                    p.roll()
                else:
                    break
        p.team.move_forward()

    def take_action(self):
        p = self.player
        shop = p.shop
        team_pets = [x.pet for x in p.team if not x.empty]
        affordable = [
            i for i, slot in enumerate(shop) if not slot.empty and slot.cost <= p.gold
        ]
        pets = [i for i in affordable if shop[i].slot_type in ["pet", "levelup"]]
        foods = [i for i in affordable if shop[i].slot_type == "food"]

        ### Merge a shop pet into the same pet on the team
        for i in pets:
            for team_pet in team_pets:
                if team_pet.name == shop[i].obj.name and team_pet.level < 3:
                    p.buy_combine(i, team_pet)
                    return True

        ### Fill the team, preferring higher tier pets
        if pets and len(team_pets) < 5:
            best = max(pets, key=lambda i: (shop[i].obj.tier, self.rng.random()))
            p.buy_pet(best)
            return True

        ### Feed food to a random pet
        if foods and team_pets:
            i = self.rng.choice(foods)
            if food_needs_target(shop[i].obj):
                p.buy_food(i, self.rng.choice(team_pets))
            else:
                p.buy_food(i)
            return True
        return False


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
RL_PACKAGES = "torch gymnasium sb3-contrib"


class ModelBot:
    """
    Opponent played by a trained model (see models/). It shops each turn with
    its own Player, like the greedy Opponent.
    """

    def __init__(self, path):
        try:
            from sb3_contrib import MaskablePPO

            from sapai.rl.opponents import ModelOpponent
        except ImportError as e:
            raise SystemExit(
                f"Playing against a trained model needs {RL_PACKAGES}"
                + f" (missing: {e.name}). Install them with:\n"
                + f"    pip install {RL_PACKAGES}"
            )
        self.path = path
        self.model_opponent = ModelOpponent(MaskablePPO.load(path, device="cpu"))
        self.model_opponent.reset()

        self.player = Player()

    def shop_turn(self, view=None):
        """Shop with what the model can see: its side of the match"""
        self.model_opponent.shop(self.player, view)


def find_model(kind, models_dir=MODELS_DIR):
    """Path of the model in models_dir with the most training steps, e.g.
    models/token_6.3M.zip for kind "token" """
    best = None
    if os.path.isdir(models_dir):
        for name in os.listdir(models_dir):
            match = re.fullmatch(kind + r"_(\d+(?:\.\d+)?)M\.zip", name)
            if match and (best is None or float(match.group(1)) > best[0]):
                best = (float(match.group(1)), os.path.join(models_dir, name))
    if best is None:
        raise SystemExit(f"No {kind} model found in {os.path.normpath(models_dir)}")
    return best[1]


def make_opponent(spec, rng, models_dir=MODELS_DIR):
    """
    Opponent for "greedy", "token" or "mlp" (the most trained model of that
    kind in models/), or a path to a saved model
    """
    if spec == "greedy":
        return Opponent(rng), "greedy bot"
    if spec in ["token", "mlp"]:
        path = find_model(spec, models_dir)
    elif os.path.exists(spec):
        path = spec
    else:
        raise SystemExit(
            f"Unknown opponent '{spec}': use greedy, token, mlp or a model file"
        )
    name = os.path.splitext(os.path.basename(path))[0]
    return ModelBot(path), f"trained model {name}"


class Game:
    def __init__(
        self,
        seed=None,
        lives=RULES["starting_lives"],
        wins_needed=RULES["wins_to_win"],
        input_fn=input,
        print_fn=print,
        opponent="greedy",
        models_dir=MODELS_DIR,
    ):
        if seed is not None:
            np.random.seed(seed)
        self.rng = random.Random(seed)
        self.input = input_fn
        self.print = print_fn
        self.player = Player(lives=lives)
        self.opponent, self.opponent_name = make_opponent(
            opponent, self.rng, models_dir
        )
        ### The same opponent every turn; its team from the last battle is
        ###   shown before you shop
        self.match = Match(
            lives=lives,
            wins_to_win=wins_needed,
            players=[self.player, self.opponent.player],
        )
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.max_lives = lives
        self.wins_needed = wins_needed
        self.last_battle = None
        self.last_result = None

    ############################################################################
    #### Display
    ############################################################################

    def wins_text(self, wins):
        """Wins, out of the wins needed if there is a win limit"""
        return f"{wins}/{self.wins_needed}" if self.wins_needed else str(wins)

    def show(self):
        p = self.player
        self.print("")
        self.print(
            f"=== Turn {p.turn} | Gold {p.gold} | Lives {p.lives}/{self.max_lives}"
            + f" | Wins {self.wins_text(self.wins)} ==="
        )
        o = self.opponent.player
        self.print(
            f"Opponent: Lives {o.lives}/{self.max_lives}"
            + f" | Wins {self.wins_text(self.match.wins[1])}"
        )
        last = self.match.opponent_last_team(0)
        if last is not None:
            self.print("Opponent's team in the last battle: " + self.team_line(last))
        self.print("Shop:")
        for i, slot in enumerate(p.shop):
            if slot.empty:
                continue
            frozen = " (frozen)" if slot.frozen else ""
            obj = slot.obj
            tag = ""
            if slot.slot_type in ["pet", "levelup"]:
                tag = " (level-up reward)" if slot.slot_type == "levelup" else ""
                text = (
                    f"{display_name(obj.name):<14} {obj.attack}/{obj.health:<3}"
                    + f" tier {obj.tier}  {obj.ability['description']}"
                )
            else:
                text = f"{display_name(obj.name):<14} food    {obj.fd['description']}"
            self.print(f"  {i}  {slot.cost}g  {text}{tag}{frozen}")
        self.print("Team (0 is the front):")
        if len(p.team) == 0:
            self.print("  (empty)")
        for i, slot in enumerate(p.team):
            if slot.empty:
                continue
            pet = slot.pet
            self.print(f"  {i}  {describe_pet(pet)}  {pet.ability['description']}")

    def show_info(self, name):
        name = name.lower().replace(" ", "-")
        for prefix, kind in [("pet-", "pets"), ("food-", "foods")]:
            key = name if name.startswith(prefix) else prefix + name
            if key in data[kind] and key not in ["pet-none", "food-none"]:
                fd = data[kind][key]
                if kind == "pets":
                    self.print(
                        f"{fd['name']}: tier {fd['tier']}, "
                        + f"{fd['baseAttack']}/{fd['baseHealth']}"
                    )
                    for level in [1, 2, 3]:
                        ability = fd.get(f"level{level}Ability")
                        if ability:
                            self.print(f"  Lv{level}: {ability['description']}")
                else:
                    self.print(
                        f"{fd['name']}: tier {fd['tier']}, cost {fd['cost']} gold"
                    )
                    self.print(f"  {fd['ability']['description']}")
                    if "perk" in fd:
                        self.print(f"  Perk: {fd['perk']}")
                return
        self.print(f"Unknown pet or food: {name}")

    def show_log(self):
        if self.last_battle is None:
            self.print("No battle yet.")
            return
        for phase_name, phases in self.last_battle.battle_history.items():
            if phase_name == "init":
                continue
            lines = []
            for phase, entries in phases.items():
                if phase.startswith("phase_move"):
                    continue
                for entry in entries:
                    line = log_line(entry)
                    if line:
                        lines.append("    " + line)
            if lines:
                if phase_name == "start":
                    self.print("  Start of battle")
                else:
                    number = int(phase_name.split()[-1]) + 1
                    self.print(f"  Attack {number}")
                for line in lines:
                    self.print(line)

        b = self.last_battle
        result = {0: "You won", 1: "You lost", 2: "Draw"}[self.last_result]
        self.print(f"  Result: {result}.")
        self.print(f"    Your survivors: {self.team_line(b.t0)}")
        self.print(f"    Their survivors: {self.team_line(b.t1)}")

    ############################################################################
    #### Commands
    ############################################################################

    def slot_number(self, text, slots, kind):
        idx = int(text)
        if idx < 0 or idx >= len(slots) or slots[idx].empty:
            raise ValueError(f"No {kind} in slot {idx}")
        return idx

    def cmd_buy(self, args):
        p = self.player
        if len(args) not in [1, 2]:
            raise ValueError("Usage: buy S [T]")
        s = self.slot_number(args[0], p.shop, "shop item")
        slot = p.shop[s]
        t = None
        if len(args) == 2:
            t = self.slot_number(args[1], p.team, "team pet")
        if slot.cost > p.gold:
            raise ValueError(f"Not enough gold: costs {slot.cost}, you have {p.gold}")

        if slot.slot_type == "food":
            if food_needs_target(slot.obj):
                if t is None:
                    raise ValueError("Choose which team pet eats it: buy S T")
                p.buy_food(s, t)
            else:
                p.buy_food(s)
            return

        if t is None:
            if len(p.team) >= 5:
                raise ValueError(
                    "Team is full. Sell a pet, or merge: buy S T with the same pet"
                )
            p.buy_pet(s)
        else:
            team_pet = p.team[t].pet
            if team_pet.name != slot.obj.name:
                raise ValueError("Can only merge into the same pet")
            if team_pet.level == 3:
                raise ValueError("That pet is already level 3")
            p.buy_combine(s, t)

    def cmd_move(self, args):
        p = self.player
        if len(args) != 2:
            raise ValueError("Usage: move T1 T2")
        p.team.move_forward()
        n = len(p.team)
        a, b = int(args[0]), int(args[1])
        if not (0 <= a < n) or not (0 <= b < n):
            raise ValueError(f"Positions must be between 0 and {n - 1}")
        order = list(range(n))
        order.insert(b, order.pop(a))
        p.reorder(order)

    def cmd_combine(self, args):
        p = self.player
        if len(args) != 2:
            raise ValueError("Usage: combine T1 T2")
        a = self.slot_number(args[0], p.team, "team pet")
        b = self.slot_number(args[1], p.team, "team pet")
        if a == b:
            raise ValueError("Choose two different pets")
        if p.team[a].pet.name != p.team[b].pet.name:
            raise ValueError("Can only merge the same pet")
        if p.team[a].pet.level == 3:
            raise ValueError("That pet is already level 3")
        p.combine(a, b)

    def handle(self, line):
        """
        Run one command. Returns 'end' to battle, 'quit' to stop, 'changed'
        when the shop or team changed.
        """
        p = self.player
        try:
            words = shlex.split(line)
        except ValueError:
            words = line.split()
        if len(words) == 0:
            return None
        cmd, args = words[0].lower(), words[1:]
        if cmd in ["help", "h", "?"]:
            self.print(HELP)
        elif cmd in ["quit", "exit", "q"]:
            return "quit"
        elif cmd == "end":
            return "end"
        elif cmd == "log":
            self.show_log()
        elif cmd == "info":
            if not args:
                raise ValueError("Usage: info NAME")
            self.show_info(" ".join(args))
        elif cmd == "buy":
            self.cmd_buy(args)
        elif cmd == "sell":
            if len(args) != 1:
                raise ValueError("Usage: sell T")
            t = self.slot_number(args[0], p.team, "team pet")
            p.sell(t)
        elif cmd == "roll":
            if p.gold < 1:
                raise ValueError("Not enough gold to roll")
            p.roll()
        elif cmd == "freeze":
            if len(args) != 1:
                raise ValueError("Usage: freeze S")
            s = self.slot_number(args[0], p.shop, "shop item")
            if p.shop[s].frozen:
                p.unfreeze(s)
            else:
                p.freeze(s)
        elif cmd == "move":
            self.cmd_move(args)
        elif cmd == "combine":
            self.cmd_combine(args)
        else:
            raise ValueError(f"Unknown command '{cmd}'. Type help for commands.")
        if cmd in ["buy", "sell", "roll", "freeze", "move", "combine"]:
            return "changed"
        return None

    ############################################################################
    #### Game loop
    ############################################################################

    def battle(self):
        p = self.player
        turn = p.turn

        ### The opponent shops at the same time, without seeing your team
        self.opponent.shop_turn(self.match.view(1))
        p.team.move_forward()

        record = self.match.battle()
        result = record["result"]
        b = self.match.last_battle
        self.last_battle = b
        self.last_result = result

        self.print("")
        self.print("--- Battle ---")
        self.print("You:      " + self.team_line(Team.from_state(record["teams"][0])))
        self.print("Opponent: " + self.team_line(Team.from_state(record["teams"][1])))

        lost = lives_lost(turn)
        plural = "life" if lost == 1 else "lives"
        if result == 0:
            self.wins += 1
            self.print(f"You won! ({self.wins_text(self.wins)} wins)")
            self.print(f"Your opponent loses {lost} {plural}.")
        elif result == 1:
            self.losses += 1
            self.print(f"You lost and lose {lost} {plural}.")
        else:
            self.draws += 1
            self.print("Draw.")
        self.print("Survivors (you): " + self.team_line(b.t0))
        self.print("Survivors (opp): " + self.team_line(b.t1))
        self.print("Type log to see what happened in the battle.")
        return result

    @staticmethod
    def team_line(team):
        pets = [
            f"{display_name(x.pet.name)} {x.pet.attack}/{x.pet.health}"
            for x in team
            if not x.empty and x.pet.health > 0
        ]
        return ", ".join(pets) if pets else "(none)"

    def game_over(self):
        m = self.match
        if not m.over:
            return False
        turn = self.player.turn
        if m.winner == 0:
            if self.wins_needed and self.wins >= self.wins_needed:
                self.print(f"\nYou win the game with {self.wins} wins on turn {turn}!")
            else:
                self.print(f"\nYou win! Your opponent ran out of lives on turn {turn}.")
        elif m.winner == 1:
            if self.wins_needed and m.wins[1] >= self.wins_needed:
                self.print(
                    f"\nGame over. Your opponent reached {m.wins[1]} wins on turn {turn}."
                )
            else:
                self.print(
                    f"\nGame over. You ran out of lives with {self.wins} wins"
                    + f" on turn {turn}."
                )
        else:
            self.print(f"\nThe game is a draw on turn {turn}.")
        return True

    def run(self):
        self.print("Super Auto Pets (Turtle Pack) in sapai. Type help for commands.")
        self.print(f"Your opponent: {self.opponent_name}")
        self.show()
        while True:
            try:
                line = self.input("> ")
            except EOFError:
                return "quit"
            try:
                action = self.handle(line)
            except Exception as e:
                self.print(f"Can't do that: {e}")
                continue
            if action == "quit":
                self.print("Bye!")
                return "quit"
            if action == "end":
                self.battle()
                if self.game_over():
                    return {0: "won", 1: "lost"}.get(self.match.winner, "draw")
                self.show()
            elif action == "changed":
                self.show()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--lives", type=int, default=RULES["starting_lives"], help="starting lives"
    )
    parser.add_argument(
        "--wins",
        type=int,
        default=RULES["wins_to_win"],
        help="battle wins that also end the game (default: none, only lives)",
    )
    parser.add_argument(
        "--opponent",
        default="greedy",
        help="greedy (default), token or mlp (the most trained model of that"
        + " kind in models/), or a path to a saved model",
    )
    args = parser.parse_args(argv)
    Game(
        seed=args.seed,
        lives=args.lives,
        wins_needed=args.wins,
        opponent=args.opponent,
    ).run()


if __name__ == "__main__":
    main()
