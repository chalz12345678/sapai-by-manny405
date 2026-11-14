# sapai rule book

The rules the engine follows, each with where it comes from, how sure we are
that the real game works the same way, and the test that checks it. The goal
is for every rule here to be confirmed in the real game.

- **Game version:** Turtle Pack, data checked against the game on 2026-11-11
  and confirmed current (`pack.game_version` in `data/turtle_pack.yaml`).
- **Numbers** (gold, costs, limits, lives, per-turn rules, pet and food
  values) live in `data/turtle_pack.yaml`; this book describes the rules
  around them. See `docs/PATCHING.md` for updating after a patch.

**Status key**

| Status | Meaning |
|---|---|
| ✅ game | Confirmed in the real game (a `verified: true` scenario or a note of how) |
| 📖 text | Follows the in-game ability or food text, not yet seen in the game |
| 🤔 assumed | Our best understanding of the game; needs checking |
| ❓ open | Unknown; the engine's current choice is listed. See Open questions |

Nothing is ✅ yet: confirming rules in the game is the next step (see
`tests/scenarios/README.md`).

## 1. Match (1v1)

| # | Rule | Status | Test |
|---|---|---|---|
| 1.1 | Two players meet every turn; there is no random opponent (arena mode is gone) | ✅ decided | `tests/test_match.py`, `tests/test_rl.py` |
| 1.2 | Both players shop at the same time and cannot see each other's shop or current team | ✅ game | `sapai/match.py` |
| 1.3 | After a battle each player knows the other's team as it entered that battle (pets, levels, stats, perks), and the other's lives and wins | ✅ game | `test_each_player_sees_the_other_team_as_it_entered_battle` |
| 1.4 | The winner of a battle gains a win; the loser loses lives (rule 1.5); a win or a draw costs no lives | ✅ decided | `test_winner_gains_a_win_and_loser_loses_lives_for_the_turn`, `test_draw_costs_no_lives` |
| 1.5 | A loss always costs 1 life, whatever the turn | ✅ decided | `test_a_loss_always_costs_one_life` |
| 1.6 | Players start with 6 lives | ✅ decided | `test_starts_with_the_rules` |
| 1.7 | Only running out of lives ends the match: the last player with lives left wins; wins do not end it | ✅ decided | `test_wins_alone_do_not_end_a_match`, `test_ends_when_a_player_runs_out_of_lives` |

## 2. Turn and shop

| # | Rule | Status | Test |
|---|---|---|---|
| 2.1 | Every turn starts with 10 gold; unspent gold is lost | ✅ game | `test_turtle_pack.py` |
| 2.2 | Pets cost 3 gold; foods cost their listed cost | ✅ game | `test_turtle_pack.py::test_foods` |
| 2.3 | A roll costs 1 gold and replaces every unfrozen shop slot | ✅ game | `test_shop*.py` |
| 2.4 | Shop slots: 3 pets and 1 food on turns 1-4, 4 pets and 2 foods on turns 5-8, 5 pets and 2 foods from turn 9 | ✅ game | `tests/test_shop_rules.py::test_slots_by_turn` |
| 2.5 | Tiers unlock every two turns: tier 2 on turn 3, ... tier 6 on turn 11 | ✅ game | `test_shop_only_rolls_turtle_pack` |
| 2.6 | Each shop slot is drawn uniformly from every pet (or food) of the unlocked tiers | 🤔 assumed | `test_shop_only_rolls_turtle_pack` |
| 2.7 | Frozen slots stay through rolls and into the next turn | ✅ game | `test_shop*.py` |
| 2.8 | A team holds at most 5 pets. The shop holds at most 9 slots in all: extra food (Worm, Pigeon, Cow) or level-up pets go into free slots; when it is full, the rightmost pet that is not frozen is removed, then (if every pet is frozen) the rightmost unfrozen food; if everything is frozen, the new item is lost | ✅ game | `tests/test_shop_rules.py::TestFullShop` |
| 2.9 | Selling gives gold equal to the pet's level, plus 1 per Cake end of turn | ✅ game | `test_sell_triggers.py` |
| 2.11 | Buying a Cow replaces all food in the shop, frozen or not, with two free Milks (Better or Best Milk at level 2 or 3); the Milks always show, replacing pets if the shop is full | ✅ game | `tests/test_shop_rules.py::TestCowMilk` |
| 2.10 | Attack and health are capped at 50, but perk bonus damage goes on top (a 50 attack pet with Steak hits for 70, through Melon) | ✅ game | `tests/test_shop_rules.py::TestPerkDamageAboveTheCap` |

## 3. Levels and merging

| # | Rule | Status | Test |
|---|---|---|---|
| 3.1 | A pet is worth copies: level 2 at 3 copies, level 3 at 6 | ✅ game | `tests/test_merging.py::test_copies` |
| 3.2 | Merging two pets of the same kind adds the copies of both (two half-stacked pets make a level 2 with 1 extra) | ✅ game | `tests/test_merging.py` |
| 3.3 | The merged pet keeps the higher attack and the higher health of the two | ✅ game | `test_merging_keeps_the_higher_stats_plus_one` |
| 3.4 | Merging adds +1/+1 for every copy the merged-in pet is worth (two level 2 pets: +3/+3), capped at 50 | ✅ decided | `test_merging_adds_one_per_copy_merged_in` |
| 3.5 | The merged pet keeps the higher-tier perk of the two (ties: the pet kept) | ❓ open (Q6) | none yet |
| 3.6 | Each level gained adds two level-up pets from the next tier; only one can be bought (buying one removes the other) and only one can be frozen. Level-up abilities (Fish) fire | ✅ game | `tests/test_shop_rules.py::TestLevelUpChoice` |
| 3.7 | Chocolate gives 1 experience | ✅ game | `test_turtle_pack.py` |

## 4. Battle

| # | Rule | Status | Test |
|---|---|---|---|
| 4.1 | Teams move to the front before the battle; empty slots close up | ✅ game | `tests/scenarios/basics.yaml` |
| 4.2 | Start-of-battle abilities happen once, before the first attack | ✅ game | `tests/scenarios/abilities.yaml` |
| 4.3 | Each round the two front pets hit each other at the same time, each dealing its attack | ✅ game | `tests/scenarios/basics.yaml` |
| 4.4 | A pet at 0 health or less faints; its faint ability happens and the team moves up | ✅ game | `tests/scenarios/basics.yaml` |
| 4.5 | Abilities in the same phase go in order of attack, highest first | ✅ confirmed | `basics.yaml` (verified) |
| 4.5a | Equal attack goes by health (higher first), then at random | 🤔 assumed (Q7) | `test_battles.py` |
| 4.11 | A Hurt ability still fires when the hit knocks the pet out (Blowfish, Camel), before the pet faints | ✅ confirmed | `basics.yaml` (verified) |
| 4.10 | A pet knocked out during start of battle (for example Dolphin) still uses its own start of battle ability | ✅ confirmed | `basics.yaml` (verified) |
| 4.6 | The battle ends when a team has no pets; both empty is a draw | ✅ game | `tests/scenarios/basics.yaml` |
| 4.7 | A summon that finds the team full does nothing | ✅ game | `test_battle_fixes.py` |
| 4.8 | Summoned pets appear where the summoning pet was | ✅ game | `test_battles.py` |
| 4.9 | Neither side is favoured: swapping the teams mirrors the result | 🤔 assumed | `tests/test_stress.py` |

## 5. Perks

| # | Rule | Status | Test |
|---|---|---|---|
| 5.1 | A pet holds one perk; a new perk replaces the old one | ✅ game | `test_food_fixes.py` |
| 5.2 | Garlic: 2 less damage, minimum 1, from attacks and abilities | ✅ game | `perks.yaml` |
| 5.3 | Melon: 20 less damage once, then it is used up | ✅ game | `perks.yaml` |
| 5.4 | Meat Bone: attacks deal 3 more damage (not abilities) | ✅ game | `perks.yaml` |
| 5.5 | Steak: the next attack deals 20 more damage, once | ✅ game | `test_battles.py` |
| 5.6 | Chili: attacks also deal 5 damage to the pet behind the target | ✅ game | `test_battle_fixes.py` |
| 5.7 | Mushroom: on fainting, the pet comes back as a 1/1 without the perk | ✅ game | `perks.yaml` |
| 5.8 | Honey: on fainting, a 1/1 Bee is summoned in its place | ✅ game | `perks.yaml` |
| 5.9 | Bread: end of turn, +7 health until the next turn | ✅ game | `test_turtle_pack.py` |
| 5.10 | Cake: end of turn, +1 sell value | ✅ game | `test_turtle_pack.py` |
| 5.11 | Perks are shown separately from attack: a Meat Bone pet shows its own attack | ✅ game | `sapai/scenarios.py` |

## 6. Pets and foods

Every pet's and food's ability is listed with its text and effect in
`data/turtle_pack.yaml`. All 60 pets were checked against their text at every
level (`tests/scenarios/tier*.yaml`, 153 battles; `tests/test_audit_tier*.py`,
88 shop tests). These general rules came out of that:

| # | Rule | Status | Test |
|---|---|---|---|
| 6.1 | "Friend" never means the pet itself (Crab, Monkey, Bison's level 3 friend) | 📖 text | `tier1_2.yaml`, `test_audit_tier3_4.py`, `test_audit_tier5_6.py` |
| 6.2 | "This pet's attack" is the attack shown, without perk bonuses such as Meat Bone (Dodo, Badger, Rooster, Leopard) | ✅ game | `tier3_4.yaml` |
| 6.3 | A hit fully blocked by Coconut or Melon does not hurt the pet, so Hurt abilities do not fire | ✅ game | `tier1_2.yaml`, `tier5_6.yaml` |
| 6.4 | A pet knocked out in an attack does not use its "friend ahead attacks" ability (Kangaroo, Snake) | 🤔 assumed | `tier1_2.yaml`, `tier5_6.yaml` |
| 6.5 | "Friend faints" abilities ignore enemies fainting (Fly) | ✅ game | `regressions.yaml` |
| 6.6 | "Friend ahead faints" means the friend ahead when it fainted, even if other faint abilities run first (Ox) | ✅ game | `tier3_4.yaml` |
| 6.7 | A pet summoned onto the enemy team (Rat's Dirty Rat) is a friend of the enemy's pets (Horse buffs it) | ✅ game | `tier1_2.yaml` |
| 6.8 | A pet swallowed by Whale counts as fainting: its faint ability fires, and its start of battle ability still runs if it has not acted yet | ✅ game | `tier3_4.yaml` (verified) |
| 6.9 | Rhino repeats only when its damage knocks a pet out | ✅ game | `tier5_6.yaml` |
| 6.10 | A copied ability (Parrot) lasts until the end of the battle, and only battle abilities can be copied (not Monkey's, Penguin's or other shop abilities) | ✅ game | `test_audit_tier3_4.py` |
| 6.11 | Cat's double effect and its uses apply only to food that gives attack or health | ✅ game | `test_audit_tier5_6.py` |
| 6.12 | "Summoned: gain Peanut" is about the Scorpion itself, not friends being summoned | ✅ game | `tier5_6.yaml` |
| 6.13 | Temporary buffs are capped too: stat plus buff is at most 50 | 🤔 assumed | `tier5_6.yaml` |
| 6.14 | Skunk removes a share of the health shown, temporary buffs included, leaving at least 1 | 🤔 assumed | `test_audit_tier3_4.py` |
| 6.15 | Worm and Pigeon always stock their food; on a full shop it replaces a pet (rule 2.8) | ✅ game | `test_audit_tier1_2.py::TestLateGameStock` |

## Open questions

These need someone who plays the game to answer. The engine's current
behaviour is in brackets.

- **Q1. Game version.** Answered: the data in `data/turtle_pack.yaml`
  (checked 2026-11-11) is current.
- **Q2. Match rules.** Answered: 6 lives, a loss costs 1 life, a win or
  draw costs none, and only running out of lives ends the match.
- **Q15. More opponents.** Planned: matches against 1 to 8 opponents, each
  seen as it was in the last battle, with training picking how many at
  random. Open: who do you battle each turn when there are several
  opponents, and how are wins and the match winner decided?
- **Q3. Real-game cases.** Screenshots or recordings of tricky battles,
  especially ability order and summons, to turn 🤔 and 📖 rules into ✅.
- **Q4. Arena mode.** Answered: removed. Training, evaluation and the
  terminal game all play 1v1 matches.
- **Q5. Merge stats.** Answered: +1/+1 per copy merged in.
- **Q6. Merge perks.** Which perk does a merged pet keep? (the higher-tier
  perk; on a tie the perk of the pet being kept)
- **Q7. Ability order ties.** Highest attack first is confirmed. With equal
  attack, the engine goes by health, then at random; two player guides say
  the same, but nothing official. (health, then random)
- **Q8. Shop size.** Answered: 9 slots in all; see rule 2.8.
- **Q9. Rounding.** Seen in the game: Leopard rounded down but Skunk rounded
  up. The rule behind it is still to find. (down, minimum 1, for all but
  Skunk, which rounds to nearest)
- **Q10. Same-moment faints.** When pets faint together, do Shark, Fly, Ant
  and Flamingo count or target the others that fainted at that moment?
  (handled in order of attack)
- **Q11. Hurt that knocks out.** Answered: a Hurt ability (Camel, Blowfish,
  Peacock, Gorilla) still fires when the hit knocks the pet out. Still open:
  does Wolverine count a friend hurt by a hit that knocks it out? (no)
- **Q12. Start of battle knock-outs.** Answered: yes, it still uses its own.
- **Q13. Per-turn limits.** Are uses "per turn" (Fly, Ox, Dragon, Cat)
  shared between the shop and that turn's battle? (yes)
- **Q16. Two summons, one free slot.** A Cricket faints on a full team with a
  Fly on it: which summon takes the slot, Cricket's Zombie Cricket or Fly's
  Zombie Fly? (the pet that acts first by attack, then health: a 1 attack
  Cricket next to a healthier Fly gives a Zombie Fly)
- **Q14. Smaller cases.**
  - Whale's released pet stats (base, +2/+2 at level 2, +5/+5 at level 3).
  - Canned Food counts as food that gives attack or health for Cat: confirmed.
  - Do summoned pets count as tier 1 for Rhino? (no)
  - Does a perk's attack bonus count for ability order? (yes)
  - Buying counts as summoned for Horse, buying to merge does not: confirmed.
  - Worm's and Pigeon's foods are removed by a roll: confirmed.
