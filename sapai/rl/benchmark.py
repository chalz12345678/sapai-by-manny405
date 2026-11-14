"""
Benchmark models against each other and against frozen earlier models.

Matches are head-to-head: both sides shop every turn and battle each other for
a fixed number of turns (lives never end the match). A side's score is
(battles won + half of battles drawn) / battles, so 0.5 means even. Ratings are
Elo-style numbers fitted to all scores, with the greedy bot fixed at 1000.

Round robin between any players ("greedy" is the greedy bot):

    python -m sapai.rl.benchmark --players greedy mlp=runs/mlp2/model_2300k.zip \\
        token=runs/token2/model_1500k.zip

Keep benchmarking new checkpoints of training runs against a frozen pool:

    python -m sapai.rl.benchmark --watch runs/mlp4 runs/token4 \\
        --pool greedy mlp_2.3M=runs/mlp2/model_2300k.zip --every-minutes 45

Watch mode copies each run's latest_model.zip to runs/benchmark/, plays it
against every pool member and appends the results to runs/benchmark.csv.
"""

import argparse
import csv
import itertools
import json
import os
import random
import shutil
import time

import numpy as np

from sapai.rl.evaluate import play_games
from sapai.rl.opponents import GreedyBot, GreedyOpponent, ModelOpponent

GREEDY_RATING = 1000.0


def parse_player(spec):
    """'greedy', 'path.zip' or 'name=path.zip' -> (name, path or None)"""
    if spec == "greedy":
        return "greedy", None
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name, path
    return os.path.splitext(os.path.basename(spec))[0], spec


def load_model(path):
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(path, device="cpu")


class Player:
    """A benchmark player: the greedy bot (path None) or a saved model (.zip)"""

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.model = load_model(path) if path else None

    def controller(self, seed):
        return self.model if self.model else GreedyBot(random.Random(seed))

    def opponent(self):
        return ModelOpponent(self.model) if self.model else GreedyOpponent()


def match(a, b, games=100, turns=15, seed=20_000):
    """Score of player a against player b, with the result counts"""
    r = play_games(
        a.controller(seed),
        games,
        seed=seed,
        env_kwargs={"play_out": True, "max_turns": turns},
        opponent=b.opponent(),
    )
    score = r["battle_win_rate"] + 0.5 * r["battle_draw_rate"]
    return score, r


def expected(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def fit_ratings(names, results, fixed=None, iters=5000, lr=20.0):
    """
    Elo-style ratings that best explain the pairwise scores.

    results: list of (name_a, name_b, score of a)
    fixed: {name: rating} for ratings that are not fitted
    """
    fixed = dict(fixed or {})
    if "greedy" in names and "greedy" not in fixed:
        fixed["greedy"] = GREEDY_RATING
    ratings = {n: fixed.get(n, GREEDY_RATING) for n in names}
    ### Scores of exactly 0 or 1 would push ratings to infinity
    clipped = [(a, b, min(max(s, 0.01), 0.99)) for a, b, s in results]
    for _ in range(iters):
        grad = {n: 0.0 for n in names}
        for a, b, s in clipped:
            e = expected(ratings[a], ratings[b])
            grad[a] += s - e
            grad[b] -= s - e
        for n in names:
            if n not in fixed:
                ratings[n] += lr * grad[n]
    return ratings


def round_robin(players, games, turns, seed, verbose=True):
    results = []
    for a, b in itertools.combinations(players, 2):
        score, r = match(a, b, games, turns, seed)
        results.append((a.name, b.name, score))
        if verbose:
            print(
                f"{a.name} vs {b.name}: score {score:.3f}"
                + f" (won/lost/drawn {r['battle_win_rate']:.1%}/"
                + f"{r['battle_loss_rate']:.1%}/{r['battle_draw_rate']:.1%})",
                flush=True,
            )
    return results


def print_ratings(ratings):
    for name, rating in sorted(ratings.items(), key=lambda x: -x[1]):
        print(f"  {name:<24} {rating:7.0f}")


def watch_args_path(csv_path):
    return os.path.join(os.path.dirname(csv_path) or ".", "benchmark_watch.json")


def latest_steps(run_dir):
    path = os.path.join(run_dir, "progress.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return int(rows[-1]["timesteps"]) if rows else None


def previous_results(csv_path, pool):
    """
    Known scores between pool members from benchmark.csv, as
    {(name_a, name_b): score of a}. Pool members are matched by name or by
    checkpoint file name, so a checkpoint rated earlier (for example
    runs/benchmark/mlp5_8200000.zip) brings its results against the pool.
    """
    if not os.path.exists(csv_path):
        return {}
    ids = {}
    for p in pool:
        ids[p.name] = p.name
        if p.path:
            ids[os.path.splitext(os.path.basename(p.path))[0]] = p.name
    results = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            model = row["model"]
            if row["steps"]:
                model = f"{model}_{row['steps']}"
            a, b = ids.get(model), ids.get(row["opponent"])
            if a and b and a != b:
                results[(a, b)] = float(row["score"])
                results[(b, a)] = 1 - float(row["score"])
    return results


def pool_round_robin(pool, games, turns, seed, csv_path):
    """Pool results, reusing known ones and only playing missing pairs"""
    known = previous_results(csv_path, pool)
    results = []
    played = []
    for a, b in itertools.combinations(pool, 2):
        if (a.name, b.name) in known:
            results.append((a.name, b.name, known[(a.name, b.name)]))
        else:
            score, r = match(a, b, games, turns, seed)
            results.append((a.name, b.name, score))
            played.append((a.name, b.name, score))
            print(f"{a.name} vs {b.name}: score {score:.3f}", flush=True)
    print(
        f"Pool: reused {len(results) - len(played)} results, played"
        + f" {len(played)} new matches",
        flush=True,
    )
    return results, played


def watch(run_dirs, pool, games, turns, seed, every_minutes, out_dir, csv_path):
    os.makedirs(out_dir, exist_ok=True)
    names = [p.name for p in pool]
    ### Reuse known results between pool members, so a restart or a bigger
    ###   pool only plays the missing pairs
    print("Rating the benchmark pool", flush=True)
    pool_results, played = pool_round_robin(pool, games, turns, seed, csv_path)
    pool_ratings = fit_ratings(names, pool_results)
    print_ratings(pool_ratings)

    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                [
                    "time",
                    "model",
                    "steps",
                    "opponent",
                    "score",
                    "won",
                    "lost",
                    "drawn",
                    "rating",
                ]
            )
        if played:
            for a, b, s in played:
                writer.writerow(
                    [
                        time.strftime("%Y-%m-%d %H:%M"),
                        a,
                        "",
                        b,
                        round(s, 4),
                        "",
                        "",
                        "",
                        round(pool_ratings[a], 1),
                    ]
                )

    done = {}
    while True:
        for run_dir in run_dirs:
            run = os.path.basename(os.path.normpath(run_dir))
            steps = latest_steps(run_dir)
            latest = os.path.join(run_dir, "latest_model.zip")
            if not os.path.exists(latest):
                continue
            if steps is None or done.get(run) == steps:
                continue
            done[run] = steps
            name = f"{run}_{steps}"
            path = os.path.join(out_dir, name + ".zip")
            shutil.copy(latest, path)
            candidate = Player(name, path)
            results = []
            rows = []
            for opponent in pool:
                score, r = match(candidate, opponent, games, turns, seed)
                results.append((name, opponent.name, score))
                rows.append((opponent.name, score, r))
            rating = fit_ratings(
                [name] + [p.name for p in pool], results, fixed=pool_ratings
            )[name]
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for opponent_name, score, r in rows:
                    writer.writerow(
                        [
                            time.strftime("%Y-%m-%d %H:%M"),
                            run,
                            steps,
                            opponent_name,
                            round(score, 4),
                            round(r["battle_win_rate"], 4),
                            round(r["battle_loss_rate"], 4),
                            round(r["battle_draw_rate"], 4),
                            round(rating, 1),
                        ]
                    )
            scores = ", ".join(f"{o} {s:.3f}" for o, s, _ in rows)
            print(f"{name}: rating {rating:.0f} ({scores})", flush=True)
        if every_minutes <= 0:
            return
        time.sleep(every_minutes * 60)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--players", nargs="+", help="round robin between these")
    parser.add_argument("--watch", nargs="+", help="training run directories")
    parser.add_argument("--pool", nargs="+", help="frozen benchmark players")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--turns", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--every-minutes", type=float, default=45)
    parser.add_argument("--out-dir", default="runs/benchmark")
    parser.add_argument("--csv", default="runs/benchmark.csv")
    args = parser.parse_args(argv)

    if args.players:
        players = [Player(*parse_player(x)) for x in args.players]
        results = round_robin(players, args.games, args.turns, args.seed)
        print("Ratings:")
        print_ratings(fit_ratings([p.name for p in players], results))
    elif args.watch:
        if not args.pool:
            parser.error("--watch needs --pool")
        ### Settings of the watcher, used by sapai.rl.resume_all
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(watch_args_path(args.csv), "w") as f:
            json.dump(vars(args), f, indent=2)
        pool = [Player(*parse_player(x)) for x in args.pool]
        watch(
            args.watch,
            pool,
            args.games,
            args.turns,
            args.seed,
            args.every_minutes,
            args.out_dir,
            args.csv,
        )
    else:
        parser.error("use --players or --watch")


if __name__ == "__main__":
    main()
