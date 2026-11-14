# Updating sapai after a game patch

All game data lives in one file, **`data/turtle_pack.yaml`**: every pet, food,
perk, the general rules (gold, costs, limits) and the per-turn rules (shop
slots, tiers, lives lost). The engine never reads that file directly. It
reads `sapai/game_data.json`, which is built from it.

## The workflow

1. Edit `data/turtle_pack.yaml`, and update `pack.game_version` at the top.
2. Rebuild:

   ```bash
   python -m sapai.build_data
   ```

   The build stops with a list of problems if the file refers to something
   that does not exist (a pet, food or perk), or uses a trigger, effect or
   target that the engine does not implement. It also warns when a number in
   an ability's text is not used by its effect, which usually means the text
   was patched but the effect was not, or the other way round.
3. Run the tests:

   ```bash
   python -m unittest discover -s tests
   ```

   Tests that check a pet's old numbers will fail. That is expected: update
   each one to the new numbers, which also confirms the engine does what the
   patch notes say.
4. Add real-game scenarios for anything the patch changed (see
   `tests/scenarios/README.md`).
5. Retrain or re-check the models in `models/`. They learned the old game.

`python -m sapai.build_data --check` fails if `game_data.json` was not
rebuilt after the YAML changed. The test suite runs the same check.

## Common changes

**A pet's stats or tier.** Change `attack`, `health` or `tier`. Shop odds
are worked out from the tiers, so nothing else is needed.

**A number in an ability.** Change it in the effect and the text of each level
that changed. For example, to make Ant give +2/+2 at level 1:

```yaml
    1:
      text: 'Faint: Give one random friend +2 attack and +2 health.'
      effect:
        kind: ModifyStats
        target: {kind: RandomFriend, n: 1}
        attackAmount: 2
        healthAmount: 2
        untilEndOfBattle: false
```

**How many times an ability works per turn.** `maxTriggers` on the level.

**A food's cost, tier or amounts.** Change `cost`, `tier`, or the amounts in
its `effect`.

**A perk's amount** (Garlic, Melon, Meat Bone, Steak, Chili, Bread, Cake...).
Change the amount in the perk's `effect` under `statuses`. The engine reads
`damageModifier`, Chili's `amount`, Bread's `healthAmount` and Cake's `amount`
from there.

**Shop slots, tier unlocks or lives lost per turn.** `rules.turns`. The last
turn listed applies to every later turn.

**Gold, roll cost, pet cost, team size, stat cap, lives, wins.** `rules`.

**A new pet that uses existing mechanics.** Add an entry under `pets` with an
`id` of the form `pet-name`. Copy a pet with a similar ability and change it.

**A new mechanic.** If a pet needs a trigger, effect or target that is not in
the lists below, the build will say so. That needs engine code: a new
function in `sapai/effects.py` for an effect, a new case in
`get_target` for a target, or a new trigger check in `sapai/pets.py` and the
place in the game where it fires. Add tests for it.

## What the engine supports

These lists are read from the engine's code by `sapai.build_data`, so the
build is always the final word.

**Triggers:** AfterAttack, BeforeAttack, Buy, BuyAfterLoss, BuyFood,
BuyTier1Animal, CastsAbility (Tiger), EatsShopFood, EndOfTurn,
EndOfTurnAfterLoss, EndOfTurnWith3PlusGold, EndOfTurnWith4OrLessAnimals,
EndOfTurnWithLvl3Friend, Faint, FriendHurt, Hurt, KnockOut, LevelUp,
PurchaseFood (Cat), Sell, StartOfBattle, StartOfTurn, Summoned, and for perks
WhenAttacking and WhenDamaged.

**Effects:** AllOf, ApplyStatus, DealDamage, DiscountFood, Evolve,
FoodMultiplier, GainAbility, GainExperience, GainGold, ModifyStats, OneOf,
ReduceHealth, RefillShops, RemoveHealth, RepeatAbility, RespawnPet, StockFood,
SummonPet, SummonRandomPet, Swallow, TransferAbility, TransferStats.

**Targets:** AdjacentAnimals, AdjacentFriends, All, DifferentTierAnimals,
EachEnemy, EachFriend, EachShopAnimal, FirstEnemy, FriendAhead (n),
FriendBehind (n), HighestHealthEnemy, HighestHealthFriend, LastEnemy,
LeftMostFriend, Level2And3Friends, LowestHealthEnemy, NonWeakEnemy,
RandomEnemy (n), RandomFriend (n), RandomLevel2And3Friends, RightMostFriend,
Self, StrongestFriend, TriggeringEntity; for foods also PurchaseTarget.

`triggered_by` says whose event fires the trigger: `Self`, `EachFriend` (for
example Horse on a friend being summoned) or `FriendAhead` (for example
Kangaroo after the friend ahead attacks).

## Things still handled by name in the code

A few behaviours are coded by pet or food id rather than read from the data,
so a patch that changes how they work (not just their numbers) needs code:
Tiger, Cat, Chocolate, Sleeping Pill, Canned Food, and how perks combine when
pets merge (`get_combined_status` in `sapai/player.py`).
