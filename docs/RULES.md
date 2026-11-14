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
| 1.2 | Both players shop at the same time and cannot see each other's shop or current team | 🤔 assumed | `sapai/match.py` |
| 1.3 | After a battle each player knows the other's team as it entered that battle (pets, levels, stats, perks), and the other's lives and wins | 🤔 assumed | `test_each_player_sees_the_other_team_as_it_entered_battle` |
| 1.4 | The winner of a battle gains a win; the loser loses lives (rule 1.5); a win or a draw costs no lives | ✅ decided | `test_winner_gains_a_win_and_loser_loses_lives_for_the_turn`, `test_draw_costs_no_lives` |
| 1.5 | A loss always costs 1 life, whatever the turn | ✅ decided | `test_a_loss_always_costs_one_life` |
| 1.6 | Players start with 6 lives | ✅ decided | `test_starts_with_the_rules` |
| 1.7 | The first player to 10 wins, or the last with lives left, wins the match | ❓ open (Q2) | `test_ends_at_the_wins_needed`, `test_ends_when_a_player_runs_out_of_lives` |

## 2. Turn and shop

| # | Rule | Status | Test |
|---|---|---|---|
| 2.1 | Every turn starts with 10 gold; unspent gold is lost | 🤔 assumed | `test_turtle_pack.py` |
| 2.2 | Pets cost 3 gold; foods cost their listed cost | 📖 text | `test_turtle_pack.py::test_foods` |
| 2.3 | A roll costs 1 gold and replaces every unfrozen shop slot | 🤔 assumed | `test_shop*.py` |
| 2.4 | Shop pet slots: 3 on turns 1-4, 4 on turns 5-8, 5 from turn 9. Food slots: 1 on turns 1-2, then 2 | 🤔 assumed | `test_turtle_pack.py` |
| 2.5 | Tiers unlock every two turns: tier 2 on turn 3, ... tier 6 on turn 11 | 🤔 assumed | `test_shop_only_rolls_turtle_pack` |
| 2.6 | Each shop slot is drawn uniformly from every pet (or food) of the unlocked tiers | 🤔 assumed | `test_shop_only_rolls_turtle_pack` |
| 2.7 | Frozen slots stay through rolls and into the next turn | 🤔 assumed | `test_shop*.py` |
| 2.8 | A team holds at most 5 pets; the shop at most 7 slots | 🤔 assumed | `tests/test_stress.py` |
| 2.9 | Selling gives gold equal to the pet's level, plus 1 per Cake end of turn | 📖 text | `test_sell_triggers.py` |
| 2.10 | Pet stats are capped at 50 attack and 50 health | 🤔 assumed | `tests/test_stress.py` |

## 3. Levels and merging

| # | Rule | Status | Test |
|---|---|---|---|
| 3.1 | A pet is worth copies: level 2 at 3 copies, level 3 at 6 | 🤔 assumed | `tests/test_merging.py::test_copies` |
| 3.2 | Merging two pets of the same kind adds the copies of both (two half-stacked pets make a level 2 with 1 extra) | 🤔 assumed | `tests/test_merging.py` |
| 3.3 | The merged pet keeps the higher attack and the higher health of the two, +1 each | 🤔 assumed | `test_merging_keeps_the_higher_stats_plus_one` |
| 3.4 | When the merged-in pet is worth more than 1 copy, the stat bonus is still +1/+1 | ❓ open (Q5) | none yet |
| 3.5 | The merged pet keeps the higher-tier perk of the two (ties: the pet kept) | ❓ open (Q6) | none yet |
| 3.6 | Each level gained adds a level-up reward pet (one tier above the shop) to the shop and fires level-up abilities | 🤔 assumed | `test_from_level_1_to_level_3_gives_two_rewards` |
| 3.7 | Chocolate gives 1 experience | 📖 text | `test_turtle_pack.py` |

## 4. Battle

| # | Rule | Status | Test |
|---|---|---|---|
| 4.1 | Teams move to the front before the battle; empty slots close up | 🤔 assumed | `tests/scenarios/basics.yaml` |
| 4.2 | Start-of-battle abilities happen once, before the first attack | 📖 text | `tests/scenarios/abilities.yaml` |
| 4.3 | Each round the two front pets hit each other at the same time, each dealing its attack | 🤔 assumed | `tests/scenarios/basics.yaml` |
| 4.4 | A pet at 0 health or less faints; its faint ability happens and the team moves up | 🤔 assumed | `tests/scenarios/basics.yaml` |
| 4.5 | Abilities in the same phase go in order of attack, highest first; equal attack goes by health, then at random | ❓ open (Q7) | `test_battles.py` |
| 4.6 | The battle ends when a team has no pets; both empty is a draw | 🤔 assumed | `tests/scenarios/basics.yaml` |
| 4.7 | A summon that finds the team full does nothing | 🤔 assumed | `test_battle_fixes.py` |
| 4.8 | Summoned pets appear where the summoning pet was | 🤔 assumed | `test_battles.py` |
| 4.9 | Neither side is favoured: swapping the teams mirrors the result | 🤔 assumed | `tests/test_stress.py` |

## 5. Perks

| # | Rule | Status | Test |
|---|---|---|---|
| 5.1 | A pet holds one perk; a new perk replaces the old one | 🤔 assumed | `test_food_fixes.py` |
| 5.2 | Garlic: 2 less damage, minimum 1, from attacks and abilities | 📖 text | `perks.yaml` |
| 5.3 | Melon: 20 less damage once, then it is used up | 📖 text | `perks.yaml` |
| 5.4 | Meat Bone: attacks deal 3 more damage (not abilities) | 📖 text | `perks.yaml` |
| 5.5 | Steak: the next attack deals 20 more damage, once | 📖 text | `test_battles.py` |
| 5.6 | Chili: attacks also deal 5 damage to the pet behind the target | 📖 text | `test_battle_fixes.py` |
| 5.7 | Mushroom: on fainting, the pet comes back as a 1/1 without the perk | 📖 text | `perks.yaml` |
| 5.8 | Honey: on fainting, a 1/1 Bee is summoned in its place | 📖 text | `perks.yaml` |
| 5.9 | Bread: end of turn, +7 health until the next turn | 📖 text | `test_turtle_pack.py` |
| 5.10 | Cake: end of turn, +1 sell value | 📖 text | `test_turtle_pack.py` |
| 5.11 | Perks are shown separately from attack: a Meat Bone pet shows its own attack | 🤔 assumed | `sapai/scenarios.py` |

## 6. Pets and foods

Every pet's and food's ability is listed with its text and effect in
`data/turtle_pack.yaml`. All 60 pets were checked against their text at every
level (`tests/scenarios/tier*.yaml`, 153 battles; `tests/test_audit_tier*.py`,
88 shop tests). These general rules came out of that:

| # | Rule | Status | Test |
|---|---|---|---|
| 6.1 | "Friend" never means the pet itself (Crab, Monkey, Bison's level 3 friend) | 📖 text | `tier1_2.yaml`, `test_audit_tier3_4.py`, `test_audit_tier5_6.py` |
| 6.2 | "This pet's attack" is the attack shown, without perk bonuses such as Meat Bone (Dodo, Badger, Rooster, Leopard) | 📖 text | `tier3_4.yaml` |
| 6.3 | A hit fully blocked by Coconut or Melon does not hurt the pet, so Hurt abilities do not fire | 🤔 assumed | `tier1_2.yaml`, `tier5_6.yaml` |
| 6.4 | A pet knocked out in an attack does not use its "friend ahead attacks" ability (Kangaroo, Snake) | 🤔 assumed | `tier1_2.yaml`, `tier5_6.yaml` |
| 6.5 | "Friend faints" abilities ignore enemies fainting (Fly) | 📖 text | `regressions.yaml` |
| 6.6 | "Friend ahead faints" means the friend ahead when it fainted, even if other faint abilities run first (Ox) | 🤔 assumed | `tier3_4.yaml` |
| 6.7 | A pet summoned onto the enemy team (Rat's Dirty Rat) is a friend of the enemy's pets (Horse buffs it) | 🤔 assumed | `tier1_2.yaml` |
| 6.8 | A pet swallowed by Whale leaves the team without fainting and without its start of battle ability | 📖 text | `tier3_4.yaml` |
| 6.9 | Rhino repeats only when its damage knocks a pet out | 📖 text | `tier5_6.yaml` |
| 6.10 | A copied ability (Parrot) lasts until the end of the battle | 📖 text | `test_audit_tier3_4.py` |
| 6.11 | Cat's double effect and its uses apply only to food that gives attack or health | 📖 text | `test_audit_tier5_6.py` |
| 6.12 | "Summoned: gain Peanut" is about the Scorpion itself, not friends being summoned | 📖 text | `tier5_6.yaml` |
| 6.13 | Temporary buffs are capped too: stat plus buff is at most 50 | 🤔 assumed | `tier5_6.yaml` |
| 6.14 | Skunk removes a share of the health shown, temporary buffs included, leaving at least 1 | 🤔 assumed | `test_audit_tier3_4.py` |
| 6.15 | Shop abilities that stock food (Worm, Pigeon) do nothing once the shop has 7 slots, so from turn 9 they stock nothing | ❓ open (Q8) | `test_audit_tier1_2.py::TestLateGameStock` (expected failure) |

## Open questions

These need someone who plays the game to answer. The engine's current
behaviour is in brackets.

- **Q1. Game version.** Answered: the data in `data/turtle_pack.yaml`
  (checked 2026-11-11) is current.
- **Q2. Match rules.** Partly answered: 6 lives, a loss costs 1 life, a win
  or draw costs none. Still open: is the match won by reaching 10 wins, or
  only by being the last with lives left? (either, now)
- **Q15. More opponents.** Planned: matches against 1 to 8 opponents, each
  seen as it was in the last battle, with training picking how many at
  random. Open: who do you battle each turn when there are several
  opponents, and how are wins and the match winner decided?
- **Q3. Real-game cases.** Screenshots or recordings of tricky battles,
  especially ability order and summons, to turn 🤔 and 📖 rules into ✅.
- **Q4. Arena mode.** Answered: removed. Training, evaluation and the
  terminal game all play 1v1 matches.
- **Q5. Merge stats.** Merging a pet worth several copies: +1/+1 or +1/+1 per
  copy? (+1/+1)
- **Q6. Merge perks.** Which perk does a merged pet keep? (the higher-tier
  perk; on a tie the perk of the pet being kept)
- **Q7. Ability order ties.** With equal attack, is the order by health, or
  random? (health, then random)
- **Q8. Shop size.** Is the shop limited to 7 slots? From turn 9 the shop
  has 5 pets and 2 foods, so with that limit Worm's Apple and Pigeon's Bread
  Crumbs never appear. (7 slots; two tests mark this as an expected failure)
- **Q9. Rounding.** How do halves round (50% of 5 attack for Dodo, Badger,
  Rooster, Leopard; Crab's 25%; Skunk)? (down, minimum 1)
- **Q10. Same-moment faints.** When pets faint together, do Shark, Fly, Ant
  and Flamingo count or target the others that fainted at that moment?
  (handled in order of attack)
- **Q11. Hurt that knocks out.** Does a hit that makes a pet faint still set
  off its Hurt ability (Camel, Blowfish, Peacock)? Does Wolverine count it?
  (no)
- **Q12. Start of battle knock-outs.** Does a pet knocked out by a faster
  start-of-battle ability still use its own? (yes)
- **Q13. Per-turn limits.** Are uses "per turn" (Fly, Ox, Dragon, Cat)
  shared between the shop and that turn's battle? (yes)
- **Q14. Smaller cases.**
  - Whale's released pet stats (base, +2/+2 at level 2, +5/+5 at level 3).
  - Is Canned Food "food that gives attack or health" for Cat? (yes)
  - Do summoned pets count as tier 1 for Rhino? (no)
  - Does a perk's attack bonus count for ability order? (yes)
  - Does buying count as "summoned" for Horse? (yes) Does buying to merge? (no)
  - Do Worm's and Pigeon's foods stay after a roll? (no)
