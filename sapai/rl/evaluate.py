"""
Evaluate a trained model or a bot in 1v1 matches against the greedy bot.

    python -m sapai.rl.evaluate --model runs/token/best_model.zip --games 500
    python -m sapai.rl.evaluate --bot greedy --games 500
    python -m sapai.rl.evaluate --bot random --games 500

Each game is a 1v1 match with the normal rules: the last with lives left wins. The greedy opponent shops every turn alongside the agent.
"""

import argparse
import random

import numpy as np

from sapai.rl.env import END_TURN, SAPEnv, observation_for
from sapai.rl.opponents import GreedyBot, GreedyOpponent, RandomBot


def play_games(
    controller,
    n_games,
    seed=0,
    deterministic=True,
    env_kwargs=None,
    opponent=None,
):
    """
    controller: a model with predict(obs, action_masks=..., deterministic=...)
        or a bot with shop(player)
    opponent: the opponent, by default the live greedy bot

    Returns a dict of results against the opponent.
    """
    env = SAPEnv(opponent or GreedyOpponent(), **(env_kwargs or {}))
    is_bot = hasattr(controller, "shop")
    battles = {"win": 0, "loss": 0, "draw": 0}
    games_won = 0
    turns = []
    wins = []
    for game in range(n_games):
        obs, info = env.reset(seed=seed + game)
        done = False
        while not done:
            if is_bot:
                controller.shop(env.player)
                action = END_TURN
            else:
                ### Models trained before matches were 1v1 see only part of it
                action, _ = controller.predict(
                    observation_for(controller, obs),
                    action_masks=env.action_masks(),
                    deterministic=deterministic,
                )
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        battles["win"] += info["wins"]
        battles["loss"] += info["losses"]
        battles["draw"] += info["draws"]
        games_won += int(info["game_won"])
        turns.append(info["turn"])
        wins.append(info["wins"])
    n_battles = sum(battles.values())
    return {
        "games": n_games,
        "game_win_rate": games_won / n_games,
        "battle_win_rate": battles["win"] / max(n_battles, 1),
        "battle_loss_rate": battles["loss"] / max(n_battles, 1),
        "battle_draw_rate": battles["draw"] / max(n_battles, 1),
        "mean_wins": float(np.mean(wins)),
        "mean_turns": float(np.mean(turns)),
        "engine_errors": env.engine_errors,
    }


def format_results(name, r):
    return (
        f"{name}: game win rate {r['game_win_rate']:.1%}, "
        f"battles won/lost/drawn {r['battle_win_rate']:.1%}/"
        f"{r['battle_loss_rate']:.1%}/{r['battle_draw_rate']:.1%}, "
        f"mean wins {r['mean_wins']:.2f}, mean turns {r['mean_turns']:.1f} "
        f"({r['games']} games)"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--model", help="path to a saved MaskablePPO model")
    parser.add_argument("--bot", choices=["greedy", "random"])
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args(argv)

    if args.model:
        from sb3_contrib import MaskablePPO

        controller = MaskablePPO.load(args.model, device="cpu")
        name = args.model
    elif args.bot == "greedy":
        controller = GreedyBot(random.Random(args.seed))
        name = "greedy bot"
    else:
        controller = RandomBot(random.Random(args.seed))
        name = "random bot"
    r = play_games(controller, args.games, args.seed, not args.stochastic)
    print(format_results(name, r))


if __name__ == "__main__":
    main()
