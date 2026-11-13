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

    def test_library_sources_and_pruning(self):
        library = TeamLibrary.from_bots(10, bots=("greedy",), turns=4)
        self.assertEqual(library.count("bot"), 40)
        self_teams = TeamLibrary.from_bots(3, bots=("random",), turns=4).teams
        library.add_all(self_teams, "self")
        self.assertEqual(library.count("self"), 12)

        library.prune("bot", 0.2, random.Random(0))
        ### 20% of the 10 bot teams for each of the 4 turns, self-play kept
        self.assertEqual(library.count("bot"), 8)
        self.assertEqual(library.count("self"), 12)
        for turn in library.teams:
            self.assertEqual(len(library.teams[turn]), len(library.sources[turn]))
            self.assertEqual(library.sources[turn].count("bot"), 2)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "library.pkl")
            library.save(path)
            loaded = TeamLibrary.load(path)
            self.assertEqual(loaded.count("self"), 12)
            self.assertEqual(loaded.count("bot"), 8)

            ### Libraries saved before sources existed load as bot teams
            import pickle

            with open(path, "wb") as f:
                pickle.dump(library.teams, f)
            self.assertEqual(TeamLibrary.load(path).count("bot"), 20)

        env = SAPEnv(LibraryOpponent(library))
        env.reset(seed=0)
        ### 3 self-play teams per turn, half rounds to 2 per turn
        self.assertEqual(env.prune_opponent_teams("self", 0.5), 8)

    def test_library_cap_removes_oldest_self_play_teams(self):
        library = TeamLibrary.from_bots(2, bots=("greedy",), turns=3)
        states = TeamLibrary.from_bots(4, bots=("random",), turns=3).teams
        old = {t: x[:2] for t, x in states.items()}
        new = {t: x[2:] for t, x in states.items()}
        library.add_all(old, "self", age=100)
        library.add_all(new, "self", age=200)
        self.assertEqual((library.count("bot"), library.count("self")), (6, 12))

        ### Only the oldest self-play teams go; bot teams are kept
        removed = library.cap(12)
        self.assertEqual(removed, 6)
        self.assertEqual((library.count("bot"), library.count("self")), (6, 6))
        for turn in library.teams:
            ages = [
                a
                for s, a in zip(library.sources[turn], library.ages[turn])
                if s == "self"
            ]
            self.assertEqual(ages, [200, 200])
        self.assertEqual(library.cap(100), 0)
        ### A limit of 0 removes every self-play team
        self.assertEqual(library.cap(0), 6)
        self.assertEqual(library.count("self"), 0)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "library.pkl")
            library.save(path)
            self.assertEqual(TeamLibrary.load(path).ages, library.ages)

            ### Pools saved before ages existed are aged by their order
            import pickle

            with open(path, "wb") as f:
                pickle.dump({"teams": library.teams, "sources": library.sources}, f)
            loaded = TeamLibrary.load(path)
            for turn, ages in loaded.ages.items():
                self.assertEqual(ages, sorted(ages))

        ### add_library keeps ages
        copy = TeamLibrary()
        copy.add_library(library, "self")
        self.assertEqual(copy.ages, library.ages)
        self.assertEqual(copy.count("self"), len(library))

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

            ### Most bot teams are removed once the self-play pool is big
            ###   enough, and the pool is saved for later runs
            import contextlib
            import io

            pruned = os.path.join(d, "pruned")
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                train_main(
                    [
                        "--policy=mlp",
                        "--timesteps=128",
                        "--n-envs=1",
                        "--n-steps=64",
                        "--batch-size=32",
                        "--eval-every=64",
                        "--eval-games=1",
                        "--library-games=4",
                        "--library-bots=greedy",
                        "--selfplay-every=64",
                        "--selfplay-games=1",
                        "--selfplay-initial-games=2",
                        "--prune-bots-at=1",
                        "--keep-bot-fraction=0.25",
                        f"--out={pruned}",
                    ]
                )
            self.assertIn("kept 25% of the 100 bot teams", log.getvalue())
            pool_path = os.path.join(pruned, "selfplay_teams.pkl")
            self.assertGreater(TeamLibrary.load(pool_path).count("self"), 0)

            ### A later run starts with the saved pool and prunes right away
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                train_main(
                    [
                        f"--resume={os.path.join(pruned, 'final_model.zip')}",
                        "--timesteps=64",
                        "--n-envs=1",
                        "--eval-every=64",
                        "--eval-games=1",
                        "--library-games=4",
                        "--library-bots=greedy",
                        "--selfplay-every=1000",
                        f"--selfplay-teams={pool_path}",
                        "--prune-bots-at=1",
                        f"--out={os.path.join(d, 'pruned2')}",
                    ]
                )
            self.assertIn("self-play teams", log.getvalue())
            self.assertIn("kept 20% of the 100 bot teams", log.getvalue())

            ### The opponent pool is capped by removing the oldest self-play
            ###   teams, and the run's settings are saved for resume_all
            capped = os.path.join(d, "capped")
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                train_main(
                    [
                        f"--resume={os.path.join(pruned, 'final_model.zip')}",
                        "--timesteps=64",
                        "--n-envs=1",
                        "--eval-every=64",
                        "--eval-games=1",
                        "--library-games=4",
                        "--library-bots=greedy",
                        "--selfplay-every=1000",
                        f"--selfplay-teams={pool_path}",
                        "--selfplay-games=1",
                        "--prune-bots-at=1",
                        "--max-opponent-teams=40",
                        f"--out={capped}",
                    ]
                )
            self.assertIn("opponent pool capped at 40 teams", log.getvalue())
            ### 100 bot teams over 25 turns, 20% rounds up to 1 per turn: 25
            ###   bot teams stay, so 15 self-play teams fit
            saved = TeamLibrary.load(os.path.join(capped, "selfplay_teams.pkl"))
            self.assertEqual(len(saved), 40 - 25)
            loaded = TeamLibrary.load(pool_path)
            newest = sorted(a for x in loaded.ages.values() for a in x)[-1]
            ### The newest harvest is kept (the order of removal is checked in
            ###   test_library_cap_removes_oldest_self_play_teams)
            self.assertGreaterEqual(
                max(a for x in saved.ages.values() for a in x), newest
            )
            with open(os.path.join(capped, "args.json")) as f:
                self.assertEqual(json.load(f)["max_opponent_teams"], 40)


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
        opponent = ModelOpponent(model)
        opponent.reset()
        team = opponent.team_for_turn(1)
        self.assertIsInstance(team, Team)
        opponent.after_battle(0)
        self.assertEqual((opponent.last_result, opponent.player.turn), (1, 2))

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
            open(os.path.join(d, "mlp3", "selfplay_teams.pkl"), "w").close()
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
            self.assertIn("--selfplay-teams=", mlp)
            tok = [x for x in lines if "--policy=token" in x][0]
            self.assertIn(f"--out={os.path.join(d, 'tok8')}", tok)
            self.assertNotIn("--selfplay-teams=", tok)
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

            ### Runs of other trainers continue from their run folder
            os.makedirs(os.path.join(d, "search1", "latest_agent"))
            with open(os.path.join(d, "search1", "args.json"), "w") as f:
                json.dump({"_module": "sapai.rl.search_train", "games": 5}, f)
            log = io.StringIO()
            with mock.patch.object(resume_all, "running_commands", return_value=[]):
                with contextlib.redirect_stdout(log):
                    resume_all.main(
                        ["--runs-dir", d, "--benchmark-csv", csv_path, "--dry-run"]
                        + ["--only", "search"]
                    )
            line = [x for x in log.getvalue().splitlines() if "search_train" in x][0]
            self.assertIn(f"--resume={os.path.join(d, 'search1')}", line)
            self.assertIn(f"--out={os.path.join(d, 'search2')}", line)
            self.assertIn("--games=5", line)
            self.assertNotIn("_module", line)


@unittest.skipUnless(HAVE_RL, "gymnasium, sb3-contrib and torch are required")
class TestSearch(unittest.TestCase):
    def make_agent(self, **kwargs):
        from sb3_contrib import MaskablePPO

        from sapai.rl.policy import TokenMaskablePolicy
        from sapai.rl.search import SearchAgent, ValueNet

        policy = MaskablePPO(
            TokenMaskablePolicy,
            SAPEnv(FixedOpponent(["ant"])),
            policy_kwargs={"d_model": 32, "n_heads": 2, "n_layers": 1},
            n_steps=64,
            device="cpu",
            seed=0,
        )
        return SearchAgent(policy, ValueNet(32, 2, 1), **kwargs)

    def test_value_net(self):
        from sapai.rl.search import ValueNet

        env = SAPEnv(FixedOpponent(["ant"]))
        obs, _ = env.reset(seed=0)
        values = ValueNet(32, 2, 1).predict(np.stack([obs, obs]))
        self.assertEqual(values.shape, (2,))

    def test_choose_and_shop(self):
        from sapai.rl.search import Context

        agent = self.make_agent(n_candidates=4, rollout_depth=3)
        env = SAPEnv(FixedOpponent(["ant"]))
        env.reset(seed=0)
        np.random.seed(0)
        state = env.player.state
        action, info = agent.choose(env.player, Context())
        self.assertTrue(action_mask(env.player)[action])
        self.assertIn(END_TURN, info["q"])
        ### Searching does not change the real player
        self.assertEqual(env.player.state, state)

        ### Playing whole games, ending turns before the action cap
        record = []
        for _ in range(3):
            agent.shop_env(env, record)
            self.assertLess(env.actions_this_turn, env.max_actions_per_turn)
            env.step(END_TURN)
        self.assertGreater(len(record), 0)
        self.assertTrue(all(m[a] for _, m, a in record))

    def test_prior_weight_follows_policy(self):
        from sapai.rl.search import Context

        ### With a huge prior weight the search picks the policy's favourite
        agent = self.make_agent(n_candidates=4, rollout_depth=1, prior_weight=1e6)
        env = SAPEnv(FixedOpponent(["ant"]))
        env.reset(seed=0)
        ctx = Context()
        mask = action_mask(env.player)
        probs = agent.policy_probs([ctx.encode(env.player)], [mask])[0]
        action, _ = agent.choose(env.player, ctx)
        self.assertEqual(action, int(np.argmax(np.where(mask, probs, -1))))

    def test_save_load_and_benchmark_player(self):
        from sapai.rl import benchmark
        from sapai.rl.search import SearchAgent, SearchOpponent, is_agent_folder

        agent = self.make_agent(n_candidates=3, rollout_depth=2, prior_weight=0.7)
        with tempfile.TemporaryDirectory() as d:
            folder = os.path.join(d, "agent")
            agent.save(folder)
            self.assertTrue(is_agent_folder(folder))
            loaded = SearchAgent.load(folder)
            self.assertEqual(loaded.settings(), agent.settings())

            player = benchmark.Player("search", folder)
            self.assertIsNotNone(player.agent)
            opponent = player.opponent()
            self.assertIsInstance(opponent, SearchOpponent)
            opponent.reset()
            self.assertIsInstance(opponent.team_for_turn(1), Team)
            opponent.after_battle(1)
            self.assertEqual(opponent.wins, 1)
            score, r = benchmark.match(player, benchmark.Player("greedy", None), 2, 3)
            self.assertGreaterEqual(score, 0)

    def test_search_train(self):
        from sapai.rl.search import is_agent_folder
        from sapai.rl.search_train import main as search_main
        from sapai.rl.policy import TokenMaskablePolicy
        from sb3_contrib import MaskablePPO

        with tempfile.TemporaryDirectory() as d:
            policy = MaskablePPO(
                TokenMaskablePolicy,
                SAPEnv(FixedOpponent(["ant"])),
                policy_kwargs={"d_model": 32, "n_heads": 2, "n_layers": 1},
                n_steps=64,
                device="cpu",
            )
            policy.save(os.path.join(d, "policy.zip"))
            out = os.path.join(d, "search1")
            common = [
                "--library-games=4",
                "--keep-bot-teams=20",
                "--opponent-pool",
                "--games-per-iter=2",
                "--value-steps=2",
                "--distill-every=1",
                "--distill-steps=2",
                "--eval-every=2",
                "--eval-games=1",
                "--n-candidates=3",
                "--rollout-depth=2",
                "--d-model=32",
                "--n-heads=2",
                "--n-layers=1",
            ]
            search_main(
                [
                    f"--policy={os.path.join(d, 'policy.zip')}",
                    "--bootstrap-games=2",
                    "--bootstrap-steps=2",
                    "--total-games=2",
                    f"--out={out}",
                ]
                + common
            )
            self.assertTrue(is_agent_folder(os.path.join(out, "latest_agent")))
            with open(os.path.join(out, "args.json")) as f:
                self.assertEqual(json.load(f)["_module"], "sapai.rl.search_train")
            ### Resuming continues the game count
            out2 = os.path.join(d, "search2")
            search_main(
                [f"--resume={out}", "--total-games=4", f"--out={out2}"] + common
            )
            with open(os.path.join(out2, "progress.csv")) as f:
                steps = [int(x.split(",")[0]) for x in f.read().splitlines()[1:]]
            self.assertGreaterEqual(min(steps), 2)
            self.assertGreaterEqual(max(steps), 4)


# %%
