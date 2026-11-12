# %%
import os
import random
import tempfile
import unittest

import numpy as np

try:
    import gymnasium  # noqa: F401
    import sb3_contrib  # noqa: F401
    import torch  # noqa: F401

    HAVE_RL = True
except ImportError:  # pragma: no cover
    HAVE_RL = False

if HAVE_RL:
    from sapai import Player, Team
    from sapai.rl.env import (
        ACTIONS,
        END_TURN,
        N_ACTIONS,
        OBS_SIZE,
        SAPEnv,
        action_mask,
        apply_action,
    )
    from sapai.rl.opponents import (
        GreedyBot,
        LibraryOpponent,
        RandomBot,
        TeamLibrary,
        harvest_teams,
    )


class FixedOpponent:
    """Always battles the same team"""

    def __init__(self, names):
        self.names = names

    def reset(self, rng=None):
        pass

    def team_for_turn(self, turn):
        return Team(self.names)

    def after_battle(self, result):
        pass


def random_states(n_games=8, seed=0):
    """Players at random points of random games"""
    env = SAPEnv(FixedOpponent(["ant"]))
    rng = np.random.default_rng(seed)
    states = []
    for game in range(n_games):
        env.reset(seed=seed + game)
        done = False
        while not done:
            if rng.random() < 0.2:
                states.append(env.player.state)
            action = rng.choice(np.flatnonzero(env.action_masks()))
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return states


@unittest.skipUnless(HAVE_RL, "gymnasium, sb3-contrib and torch are required")
class TestEnv(unittest.TestCase):
    def test_spaces(self):
        env = SAPEnv(FixedOpponent(["ant"]))
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, (OBS_SIZE,))
        self.assertEqual(env.action_space.n, N_ACTIONS)
        self.assertEqual(N_ACTIONS, 118)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(ACTIONS[END_TURN], ("end_turn", ()))

    def test_every_legal_action_applies(self):
        np.random.seed(0)
        for state in random_states():
            player = Player.from_state(state)
            for action in np.flatnonzero(action_mask(player)):
                if action == END_TURN:
                    continue
                copy = Player.from_state(state)
                apply_action(copy, int(action))

    def test_illegal_action_raises(self):
        env = SAPEnv(FixedOpponent(["ant"]))
        env.reset(seed=0)
        illegal = np.flatnonzero(~env.action_masks())
        self.assertGreater(len(illegal), 0)
        with self.assertRaises(ValueError):
            env.step(int(illegal[0]))

    def test_random_games_have_no_engine_errors(self):
        env = SAPEnv(FixedOpponent(["ant", "fish"]))
        rng = np.random.default_rng(1)
        for game in range(20):
            env.reset(seed=game)
            done = False
            while not done:
                obs, _, terminated, truncated, _ = env.step(
                    rng.choice(np.flatnonzero(env.action_masks()))
                )
                self.assertTrue(env.observation_space.contains(obs))
                done = terminated or truncated
        self.assertEqual(env.engine_errors, 0)

    def test_same_seed_same_game(self):
        def play(seed):
            env = SAPEnv(FixedOpponent(["ant", "fish"]))
            obs, _ = env.reset(seed=seed)
            rng = np.random.default_rng(7)
            history = [obs]
            for _ in range(60):
                action = rng.choice(np.flatnonzero(env.action_masks()))
                obs, reward, terminated, truncated, _ = env.step(action)
                history.append(np.append(obs, reward))
                if terminated or truncated:
                    break
            return np.concatenate(history)

        np.testing.assert_array_equal(play(3), play(3))
        self.assertFalse(np.array_equal(play(3), play(4)))

    def test_rewards(self):
        ### An empty opponent team loses to any team
        env = SAPEnv(FixedOpponent([]), wins_needed=2)
        env.reset(seed=0)
        buy = [
            a for a in np.flatnonzero(env.action_masks()) if ACTIONS[a][0] == "buy_pet"
        ]
        env.step(buy[0])
        _, reward, terminated, _, info = env.step(END_TURN)
        self.assertEqual((reward, terminated, info["result"]), (1.0, False, 0))
        _, reward, terminated, _, info = env.step(END_TURN)
        self.assertEqual(
            (reward, terminated, info["game_won"]), (1.0 + 5.0, True, True)
        )

        ### Losing costs reward and lives
        env = SAPEnv(FixedOpponent(["mammoth"]))
        env.reset(seed=0)
        _, reward, _, _, info = env.step(END_TURN)
        self.assertEqual((reward, info["lives"]), (-1.0, 9))

    def test_actions_per_turn_cap(self):
        env = SAPEnv(FixedOpponent(["ant"]), max_actions_per_turn=3)
        env.reset(seed=0)
        freeze = [
            a for a in np.flatnonzero(env.action_masks()) if ACTIONS[a][0] == "freeze"
        ]
        for _ in range(3):
            _, _, _, _, info = env.step(freeze[0])
        ### The third action ended the turn
        self.assertEqual(info["turn"], 2)

    def test_truncation(self):
        env = SAPEnv(FixedOpponent(["ant"]), max_turns=2, lives=100)
        env.reset(seed=0)
        env.step(END_TURN)
        _, _, terminated, truncated, _ = env.step(END_TURN)
        self.assertEqual((terminated, truncated), (False, True))


@unittest.skipUnless(HAVE_RL, "gymnasium, sb3-contrib and torch are required")
class TestOpponents(unittest.TestCase):
    def test_bots_and_library(self):
        library = TeamLibrary.from_bots(3, turns=5)
        self.assertEqual(len(library), 3 * 2 * 5)
        team = library.sample(3, random.Random(0))
        self.assertIsInstance(team, Team)
        ### Turns past the recorded ones use the last recorded turn
        self.assertIsInstance(library.sample(20, random.Random(0)), Team)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "library.pkl")
            library.save(path)
            self.assertEqual(len(TeamLibrary.load(path)), len(library))

        env = SAPEnv(LibraryOpponent(library))
        env.reset(seed=0)
        env.step(END_TURN)

    def test_greedy_bot_builds_a_team(self):
        player = Player()
        GreedyBot(random.Random(0)).shop(player)
        self.assertGreater(len(player.team), 0)
        player = Player()
        RandomBot(random.Random(0), end_prob=0).shop(player)


@unittest.skipUnless(HAVE_RL, "gymnasium, sb3-contrib and torch are required")
class TestPolicy(unittest.TestCase):
    def make_model(self, policy="token"):
        from sb3_contrib import MaskablePPO

        from sapai.rl.policy import TokenMaskablePolicy

        env = SAPEnv(FixedOpponent(["ant"]))
        kwargs = dict(n_steps=64, batch_size=32, n_epochs=1, device="cpu", seed=0)
        if policy == "token":
            return MaskablePPO(
                TokenMaskablePolicy,
                env,
                policy_kwargs={"d_model": 32, "n_heads": 2, "n_layers": 1},
                **kwargs,
            )
        return MaskablePPO("MlpPolicy", env, **kwargs)

    def test_token_logits_follow_action_order(self):
        import torch as th

        from sapai.rl.env import ACTION_START, MAX_TEAM, PAIRS
        from sapai.rl.policy import TokenNet

        th.manual_seed(0)
        net = TokenNet(d_model=32, n_heads=2, n_layers=1)
        env = SAPEnv(FixedOpponent(["ant"]))
        obs, _ = env.reset(seed=0)
        self.assertEqual(
            tuple(net.forward_actor(th.tensor(obs[None])).shape), (1, N_ACTIONS)
        )

        ### Each action is scored from the tokens it names
        g, t, s = th.randn(1, 32), th.randn(1, 5, 32), th.randn(1, 7, 32)
        logits = net.logits(g, t, s)[0]
        with th.no_grad():
            for i in range(7):
                self.assertAlmostEqual(
                    float(logits[ACTION_START["buy_pet"] + i]),
                    float(net.buy_pet(s[0, i])),
                    places=5,
                )
                for j in range(MAX_TEAM):
                    k = ACTION_START["buy_food"] + i * MAX_TEAM + j
                    self.assertAlmostEqual(
                        float(logits[k]), float(net.buy_food(s, t)[0, i, j]), places=5
                    )
            for j in range(MAX_TEAM):
                self.assertAlmostEqual(
                    float(logits[ACTION_START["sell"] + j]),
                    float(net.sell(t[0, j])),
                    places=5,
                )
            for k, (a, b) in enumerate(PAIRS):
                self.assertAlmostEqual(
                    float(logits[ACTION_START["combine"] + k]),
                    float(net.combine(t, t)[0, a, b]),
                    places=5,
                )

    def test_learn_predict_save_load(self):
        from sb3_contrib import MaskablePPO

        for policy in ["token", "mlp"]:
            model = self.make_model(policy)
            model.learn(128)
            env = SAPEnv(FixedOpponent(["ant"]))
            obs, _ = env.reset(seed=1)
            for _ in range(30):
                action, _ = model.predict(obs, action_masks=env.action_masks())
                self.assertTrue(env.action_masks()[action])
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            with tempfile.TemporaryDirectory() as d:
                model.save(os.path.join(d, "model"))
                loaded = MaskablePPO.load(os.path.join(d, "model"), device="cpu")
                obs, _ = env.reset(seed=2)
                a1, _ = model.predict(
                    obs, action_masks=env.action_masks(), deterministic=True
                )
                a2, _ = loaded.predict(
                    obs, action_masks=env.action_masks(), deterministic=True
                )
                self.assertEqual(int(a1), int(a2))

            ### Self-play: teams built by the model, recorded by turn
            teams = harvest_teams(model, SAPEnv(FixedOpponent(["ant"])), 2)
            self.assertIn(1, teams)
            library = TeamLibrary()
            library.add_all(teams)
            self.assertGreater(len(library), 0)

    def test_evaluate_and_train_scripts(self):
        from sapai.rl.evaluate import play_games
        from sapai.rl.train import main as train_main

        r = play_games(GreedyBot(random.Random(0)), 2)
        self.assertEqual(r["games"], 2)
        self.assertGreater(r["mean_wins"] + r["battle_loss_rate"], 0)

        with tempfile.TemporaryDirectory() as d:
            train_main(
                [
                    "--policy=token",
                    "--timesteps=128",
                    "--n-envs=1",
                    "--n-steps=64",
                    "--batch-size=32",
                    "--eval-every=64",
                    "--eval-games=1",
                    "--library-games=2",
                    "--selfplay-every=64",
                    f"--out={d}",
                ]
            )
            for name in ["best_model.zip", "final_model.zip", "progress.csv"]:
                self.assertTrue(os.path.exists(os.path.join(d, name)))

            ### Resume training against greedy bot teams only
            resumed = os.path.join(d, "resumed")
            train_main(
                [
                    f"--resume={os.path.join(d, 'final_model.zip')}",
                    "--timesteps=64",
                    "--n-envs=1",
                    "--eval-every=64",
                    "--eval-games=1",
                    "--library-games=2",
                    "--library-bots=greedy",
                    f"--out={resumed}",
                ]
            )
            self.assertTrue(os.path.exists(os.path.join(resumed, "final_model.zip")))

            ### Self-play starts once evaluations stop improving
            plateau = os.path.join(d, "plateau")
            train_main(
                [
                    "--policy=mlp",
                    "--timesteps=320",
                    "--n-envs=1",
                    "--n-steps=64",
                    "--batch-size=32",
                    "--eval-every=64",
                    "--eval-games=1",
                    "--library-games=2",
                    "--selfplay-every=64",
                    "--selfplay-on-plateau=1",
                    "--selfplay-games=1",
                    f"--out={plateau}",
                ]
            )
            for name in ["selfplay_start.zip", "latest_model.zip"]:
                self.assertTrue(os.path.exists(os.path.join(plateau, name)))
            with open(os.path.join(plateau, "progress.csv")) as f:
                selfplay = [row.strip().split(",")[-1] for row in f][1:]
            self.assertEqual(selfplay[0], "0")
            self.assertIn("1", selfplay)


# %%
