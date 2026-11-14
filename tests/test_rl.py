# %%
import json
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
        BASE_OBS_SIZE,
        END_TURN,
        N_ACTIONS,
        OBS_SIZE,
        OPPONENT_START,
        SAPEnv,
        action_mask,
        apply_action,
    )
    from sapai.rl.opponents import (
        GreedyBot,
        GreedyOpponent,
        ModelOpponent,
        PoolOpponent,
        RandomBot,
    )


class FixedOpponent:
    """Always battles with the same team"""

    def __init__(self, names):
        self.names = names

    def reset(self, rng=None):
        pass

    def shop(self, player, view=None):
        player.team = Team(self.names)


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
    def test_agent_sees_the_opponents_last_team(self):
        env = SAPEnv(FixedOpponent(["mammoth", "ant"]))
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (OBS_SIZE,))
        ### Before the first battle only the opponent's lives are known
        self.assertEqual(obs[OPPONENT_START:-3].sum(), 0)
        self.assertEqual(obs[-1], 0)
        obs, _, _, _, info = env.step(END_TURN)
        seen = env.match.opponent_last_team(0)
        self.assertEqual(
            [x.pet.name for x in seen if not x.empty], ["pet-mammoth", "pet-ant"]
        )
        self.assertGreater(obs[OPPONENT_START:].sum(), 0)
        self.assertEqual(info["opponent_lives"], env.match.players[1].lives)

    def test_the_opponent_can_lose_the_match(self):
        ### An opponent with no pets loses every battle and runs out of lives
        env = SAPEnv(FixedOpponent([]), wins_needed=99, lives=2)
        env.reset(seed=0)
        buy = [
            a for a in np.flatnonzero(env.action_masks()) if ACTIONS[a][0] == "buy_pet"
        ]
        env.step(buy[0])
        done, total = False, 0
        while not done:
            _, reward, terminated, truncated, info = env.step(END_TURN)
            total += reward
            done = terminated or truncated
        self.assertTrue(terminated)
        self.assertTrue(info["game_won"])
        self.assertLessEqual(info["opponent_lives"], 0)

    def test_old_models_play_with_part_of_the_observation(self):
        from sb3_contrib import MaskablePPO

        from sapai.rl.env import observation_for

        model = MaskablePPO.load(
            os.path.join(os.path.dirname(__file__), "..", "models", "mlp_8.2M.zip"),
            device="cpu",
        )
        self.assertEqual(model.observation_space.shape, (BASE_OBS_SIZE,))
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        self.assertEqual(observation_for(model, obs).shape, (BASE_OBS_SIZE,))
        env = SAPEnv(ModelOpponent(model))
        env.reset(seed=0)
        env.step(END_TURN)
        self.assertEqual(env.engine_errors, 0)

    def test_pool_opponent(self):
        from sb3_contrib import MaskablePPO

        from sapai.rl.policy import TokenMaskablePolicy

        with tempfile.TemporaryDirectory() as d:
            ### An empty pool plays the greedy bot
            pool = PoolOpponent(d, greedy_fraction=0)
            pool.reset(random.Random(0))
            self.assertEqual(pool.current_name, "greedy")
            model = MaskablePPO(
                TokenMaskablePolicy,
                SAPEnv(FixedOpponent(["ant"])),
                policy_kwargs={"d_model": 32, "n_heads": 2, "n_layers": 1},
                n_steps=64,
                device="cpu",
            )
            model.save(os.path.join(d, "a.zip"))
            pool.reset(random.Random(0))
            self.assertEqual(pool.current_name, "a.zip")
            env = SAPEnv(pool)
            env.reset(seed=0)
            for _ in range(3):
                env.step(END_TURN)
            self.assertEqual(env.engine_errors, 0)
            self.assertGreater(len(env.match.players[1].team), 0)
            ### greedy_fraction 1 always plays the greedy bot
            pool = PoolOpponent(d, greedy_fraction=1)
            pool.reset(random.Random(0))
            self.assertEqual(pool.current_name, "greedy")

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
                    "--timesteps=192",
                    "--n-envs=1",
                    "--n-steps=64",
                    "--batch-size=32",
                    "--eval-every=64",
                    "--eval-games=1",
                    "--pool-every=64",
                    "--pool-size=2",
                    "--greedy-fraction=0",
                    f"--out={d}",
                ]
            )
            for name in ["best_model.zip", "final_model.zip", "progress.csv"]:
                self.assertTrue(os.path.exists(os.path.join(d, name)))
            ### Checkpoints go into the self-play pool, newest kept
            pool = sorted(os.listdir(os.path.join(d, "pool")))
            self.assertEqual(len(pool), 2)
            with open(os.path.join(d, "args.json")) as f:
                self.assertEqual(json.load(f)["pool_size"], 2)

            ### Resuming a model from before matches were 1v1 upgrades it,
            ###   and --pool-models starts the pool with other models
            old = os.path.join(
                os.path.dirname(__file__), "..", "models", "mlp_8.2M.zip"
            )
            resumed = os.path.join(d, "resumed")
            train_main(
                [
                    f"--resume={old}",
                    "--timesteps=64",
                    "--n-envs=1",
                    "--n-steps=64",
                    "--batch-size=32",
                    "--eval-every=64",
                    "--eval-games=1",
                    f"--pool-models={old}",
                    f"--out={resumed}",
                ]
            )
            from sb3_contrib import MaskablePPO

            final = MaskablePPO.load(os.path.join(resumed, "final_model.zip"))
            self.assertEqual(final.observation_space.shape, (OBS_SIZE,))
            self.assertIn("mlp_8.2M.zip", os.listdir(os.path.join(resumed, "pool")))


@unittest.skipUnless(HAVE_RL, "gymnasium, sb3-contrib and torch are required")
class TestBenchmark(unittest.TestCase):
    def test_play_out_keeps_playing(self):
        env = SAPEnv(FixedOpponent(["mammoth"]), play_out=True, max_turns=12)
        env.reset(seed=0)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(END_TURN)
            self.assertFalse(terminated)
            self.assertGreaterEqual(info["lives"], 1)
            done = truncated
        self.assertEqual(info["losses"], 12)

    def test_model_opponent_and_benchmark(self):
        from sb3_contrib import MaskablePPO

        from sapai.rl import benchmark
        from sapai.rl.opponents import ModelOpponent

        model = MaskablePPO(
            "MlpPolicy",
            SAPEnv(FixedOpponent(["ant"])),
            n_steps=64,
            batch_size=32,
            device="cpu",
            seed=0,
        )
        ### A model opponent shops for its own side of the match
        from sapai.match import Match

        opponent = ModelOpponent(model, deterministic=False)
        opponent.reset()
        m = Match(seed=0)
        opponent.shop(m.players[1], m.view(1))
        m.battle()
        self.assertEqual(m.players[1].turn, 2)

        ### Greedy against greedy is close to even
        greedy = benchmark.Player("greedy", None)
        score, r = benchmark.match(greedy, greedy, games=20, turns=8)
        self.assertGreater(score, 0.35)
        self.assertLess(score, 0.65)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.zip")
            model.save(path)
            run = os.path.join(d, "run")
            os.makedirs(run)
            model.save(os.path.join(run, "latest_model"))
            with open(os.path.join(run, "progress.csv"), "w") as f:
                f.write("timesteps,mean_wins\n64,0\n")
            csv_path = os.path.join(d, "benchmark.csv")
            pool = [greedy, benchmark.Player("frozen", path)]
            benchmark.watch([run], pool, 2, 4, 0, 0, os.path.join(d, "bench"), csv_path)
            self.assertTrue(os.path.exists(os.path.join(d, "bench", "run_64.zip")))
            with open(csv_path) as f:
                rows = f.read().splitlines()
            ### Header, one pool pair, and the candidate against both
            self.assertEqual(len(rows), 4)
            self.assertTrue(rows[-1].split(",")[1] == "run")

            ### Watching again reuses the saved pool results
            from unittest import mock

            with mock.patch.object(benchmark, "round_robin") as rr:
                benchmark.watch(
                    [run], pool, 2, 4, 0, 0, os.path.join(d, "bench"), csv_path
                )
                rr.assert_not_called()
            with open(csv_path) as f:
                self.assertEqual(len(f.read().splitlines()), 6)

            ### A checkpoint rated earlier joins the pool with its results;
            ###   only the missing pair is played
            bigger = pool + [
                benchmark.Player("ckpt", os.path.join(d, "bench", "run_64.zip"))
            ]
            known = benchmark.previous_results(csv_path, bigger)
            self.assertIn(("ckpt", "greedy"), known)
            self.assertIn(("greedy", "ckpt"), known)
            results, played = benchmark.pool_round_robin(bigger, 2, 4, 0, csv_path)
            self.assertEqual(len(results), 3)
            self.assertEqual(played, [])

    def test_fit_ratings(self):
        from sapai.rl.benchmark import fit_ratings

        results = [("a", "greedy", 0.75), ("b", "greedy", 0.5), ("a", "b", 0.75)]
        ratings = fit_ratings(["greedy", "a", "b"], results)
        self.assertEqual(ratings["greedy"], 1000)
        self.assertGreater(ratings["a"], ratings["b"])
        self.assertAlmostEqual(ratings["b"], 1000, delta=60)
        ### A 0.75 score is about +190 Elo
        self.assertAlmostEqual(ratings["a"] - ratings["b"], 190, delta=60)


@unittest.skipUnless(HAVE_RL, "gymnasium, sb3-contrib and torch are required")
class TestResumeAll(unittest.TestCase):
    def test_resume_all_dry_run(self):
        import contextlib
        import io
        from unittest import mock

        from sapai.rl import resume_all

        with tempfile.TemporaryDirectory() as d:
            for name, policy in [("mlp2", "mlp"), ("mlp3", "mlp"), ("tok7", "token")]:
                os.makedirs(os.path.join(d, name))
                with open(os.path.join(d, name, "args.json"), "w") as f:
                    json.dump({"policy": policy, "out": name, "resume": None}, f)
                open(os.path.join(d, name, "latest_model.zip"), "w").close()
            os.makedirs(os.path.join(d, "mlp3", "pool"))
            open(os.path.join(d, "mlp3", "pool", "step_5.zip"), "w").close()
            csv_path = os.path.join(d, "benchmark.csv")
            with open(os.path.join(d, "benchmark_watch.json"), "w") as f:
                json.dump(
                    {
                        "watch": ["runs/mlp3", "runs/tok7"],
                        "pool": ["greedy", "a=a.zip"],
                        "games": 100,
                        "turns": 15,
                        "seed": 1,
                        "every_minutes": 45,
                        "out_dir": "bench",
                        "csv": csv_path,
                    },
                    f,
                )

            log = io.StringIO()
            with mock.patch.object(resume_all, "running_commands", return_value=[]):
                with contextlib.redirect_stdout(log):
                    resume_all.main(
                        ["--runs-dir", d, "--benchmark-csv", csv_path, "--dry-run"]
                    )
            lines = log.getvalue().splitlines()
            mlp = [x for x in lines if "--policy=mlp" in x][0]
            self.assertIn(f"--out={os.path.join(d, 'mlp4')}", mlp)
            self.assertIn(
                f"--resume={os.path.join(d, 'mlp3', 'model_resume.zip')}", mlp
            )
            ### The next run starts with this run's self-play pool
            self.assertIn(
                "--pool-models " + os.path.join(d, "mlp3", "pool", "step_5.zip"), mlp
            )
            tok = [x for x in lines if "--policy=token" in x][0]
            self.assertIn(f"--out={os.path.join(d, 'tok8')}", tok)
            self.assertNotIn("--pool-models", tok)
            watch = [x for x in lines if "sapai.rl.benchmark" in x][0]
            self.assertIn(os.path.join(d, "mlp4"), watch)
            self.assertIn(os.path.join(d, "tok8"), watch)
            self.assertIn("--pool greedy a=a.zip", watch)

            ### Runs that are still training are left alone
            log = io.StringIO()
            running = [f"python -m sapai.rl.train --out {os.path.join(d, 'mlp3')}"]
            with mock.patch.object(
                resume_all, "running_commands", return_value=running
            ):
                with contextlib.redirect_stdout(log):
                    resume_all.main(
                        ["--runs-dir", d, "--benchmark-csv", csv_path, "--dry-run"]
                    )
            self.assertIn("mlp3 is still training", log.getvalue())
            self.assertNotIn("--policy=mlp", log.getvalue())

            ### A benchmark watcher watching a folder is not training it
            log = io.StringIO()
            running = [
                f"python -m sapai.rl.benchmark --watch {os.path.join(d, 'mlp3')}"
            ]
            with mock.patch.object(
                resume_all, "running_commands", return_value=running
            ):
                with contextlib.redirect_stdout(log):
                    resume_all.main(
                        ["--runs-dir", d, "--benchmark-csv", csv_path, "--dry-run"]
                    )
            self.assertIn("--policy=mlp", log.getvalue())

            ### --only resumes one series
            log = io.StringIO()
            with mock.patch.object(resume_all, "running_commands", return_value=[]):
                with contextlib.redirect_stdout(log):
                    resume_all.main(
                        ["--runs-dir", d, "--benchmark-csv", csv_path, "--dry-run"]
                        + ["--only", "tok"]
                    )
            self.assertIn("--policy=token", log.getvalue())
            self.assertNotIn("--policy=mlp", log.getvalue())

            ### A stopped series is never resumed
            with mock.patch.object(resume_all.subprocess, "run"):
                with contextlib.redirect_stdout(io.StringIO()):
                    resume_all.main(["--runs-dir", d, "--stop", "mlp"])
            self.assertTrue(os.path.exists(os.path.join(d, "mlp3", "STOPPED")))
            log = io.StringIO()
            with mock.patch.object(resume_all, "running_commands", return_value=[]):
                with contextlib.redirect_stdout(log):
                    resume_all.main(
                        ["--runs-dir", d, "--benchmark-csv", csv_path, "--dry-run"]
                    )
            self.assertNotIn("--policy=mlp", log.getvalue())
            self.assertIn("--policy=token", log.getvalue())


@unittest.skipUnless(HAVE_RL, "gymnasium, sb3-contrib and torch are required")
class TestGrow(unittest.TestCase):
    def test_grown_model_plays_the_same(self):
        import torch as th
        from sb3_contrib import MaskablePPO

        from sapai.rl.grow import grow_model
        from sapai.rl.policy import TokenMaskablePolicy

        env = SAPEnv(FixedOpponent(["ant"]))
        model = MaskablePPO(
            TokenMaskablePolicy,
            env,
            policy_kwargs={"d_model": 32, "n_heads": 2, "n_layers": 1},
            n_steps=64,
            device="cpu",
            seed=0,
        )
        ### Make the trained layer non-trivial
        model.learn(64)
        grown = grow_model(model, 3)
        self.assertEqual(len(grown.policy.mlp_extractor.encoder.layers), 3)
        self.assertEqual(grown.num_timesteps, model.num_timesteps)
        obs = []
        o, _ = env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(40):
            obs.append(o)
            o, _, done, trunc, _ = env.step(
                rng.choice(np.flatnonzero(env.action_masks()))
            )
            if done or trunc:
                o, _ = env.reset()
        x = th.as_tensor(np.array(obs, dtype=np.float32))
        with th.no_grad():
            for f in [
                lambda m: m.policy.get_distribution(x).distribution.logits,
                lambda m: m.policy.predict_values(x),
            ]:
                self.assertTrue(th.allclose(f(model), f(grown), atol=1e-5))
        with self.assertRaises(ValueError):
            grow_model(grown, 2)

    def test_upgraded_models_play_the_same(self):
        import torch as th
        from sb3_contrib import MaskablePPO

        from sapai.rl.grow import upgrade_model

        models = os.path.join(os.path.dirname(__file__), "..", "models")
        env = SAPEnv(GreedyOpponent())
        obs = []
        o, _ = env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(60):
            obs.append(o)
            o, _, done, trunc, _ = env.step(
                rng.choice(np.flatnonzero(env.action_masks()))
            )
            if done or trunc:
                o, _ = env.reset()
        x = th.as_tensor(np.array(obs, dtype=np.float32))
        ### The opponent's team is in these observations
        self.assertGreater(float(x[:, OPPONENT_START:].abs().sum()), 0)
        for name in ["mlp_8.2M.zip", "token_3.5M.zip"]:
            old = MaskablePPO.load(os.path.join(models, name), device="cpu")
            new = upgrade_model(old)
            self.assertEqual(new.observation_space.shape, (OBS_SIZE,))
            with th.no_grad():
                for f, a, b in [
                    (
                        lambda m, o: m.policy.get_distribution(o).distribution.logits,
                        x[:, :BASE_OBS_SIZE],
                        x,
                    ),
                    (lambda m, o: m.policy.predict_values(o), x[:, :BASE_OBS_SIZE], x),
                ]:
                    self.assertTrue(th.allclose(f(old, a), f(new, b), atol=1e-5))
            with self.assertRaises(ValueError):
                upgrade_model(new)


# %%
