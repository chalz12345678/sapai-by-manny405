"""
Start-of-battle order, as seen in the game: abilities go from the highest
attack down, a Tiger repeats the ability of the pet ahead straight after it
(not at the Tiger's own attack), a pet knocked out before its turn still
uses its ability, and a Parrot uses the ability it copied at end of turn.
"""

import unittest

import numpy as np

from gui import game
from sapai.scenarios import make_team

TEAM = ["Fish 1/50", "Leopard 12/4", "Tiger 1/30"]
ENEMY = ["Skunk 9/40", "Parrot 5/3", "Leopard 10/4", "Tiger 1/30"]


def start_lines(seed):
    np.random.seed(seed)
    frames, _ = game.replay_battle(
        make_team(TEAM), make_team(ENEMY), names=("You", "Opponent")
    )
    return frames[1]["lines"]


class TestStartOfBattleOrder(unittest.TestCase):
    def test_order_holds_whatever_the_targets(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                owners = [
                    line.split(" deals")[0].split(" removes")[0]
                    for line in start_lines(seed)
                ]
                self.assertEqual(
                    owners,
                    [
                        "You: Leopard",  # attack 12
                        "You: Leopard",  # Tiger (attack 1) repeats it right away
                        "Opponent: Leopard",  # attack 10, even if already knocked out
                        "Opponent: Leopard",  # their Tiger repeats it right away
                        "Opponent: Skunk",  # attack 9
                        "Opponent: Parrot",  # attack 5, Skunk's copied ability
                    ],
                )

    def test_skunk_and_copied_skunk_take_thirds_rounded_up(self):
        lines = start_lines(1)
        ### Fish 50 -> 45 after both Leopards' hits in this run, then a third
        ###   of 45 (15) and a third of 30 (10)
        self.assertIn("Opponent: Skunk removes health from Fish (now 1/30)", lines)
        self.assertIn("Opponent: Parrot removes health from Fish (now 1/20)", lines)


if __name__ == "__main__":
    unittest.main()
