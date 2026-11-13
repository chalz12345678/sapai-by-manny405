# Trained models

MaskablePPO agents (sb3-contrib) trained on the Turtle Pack with
`sapai.rl.train`, first against greedy bot teams and then with self-play.

| File | Policy | Training steps | Benchmark rating |
|---|---|---|---|
| **`token_13.5M.zip`** | token network (`TokenMaskablePolicy`) | 13.5M | **1459** (best) |
| `token_11.8M.zip` | token network (`TokenMaskablePolicy`) | 11.8M | 1440 |
| `token_6.3M.zip` | token network (`TokenMaskablePolicy`) | 6.3M | 1427 (first plateau) |
| `token_4.3M.zip` | token network (`TokenMaskablePolicy`) | 4.3M | 1426 |
| `token_3.5M.zip` | token network (`TokenMaskablePolicy`) | 3.5M | 1393 |
| **`mlp_19.3M.zip`** | MLP (`MlpPolicy`, 256x256) | 19.3M | **1420** (best MLP, plateau) |
| `mlp_10.9M.zip` | MLP (`MlpPolicy`, 256x256) | 10.9M | 1332 |
| `mlp_8.2M.zip` | MLP (`MlpPolicy`, 256x256) | 8.2M | 1312 |

Ratings are Elo-style from `sapai.rl.benchmark`: head-to-head matches of 100
games x 15 turns against a pool of frozen models, with the greedy bot fixed
at 1000. For reference, the MLP at 2.3M steps (the first model confirmed to
beat the greedy bot) rated 1089. Scores are battles won plus half of draws,
so 0.5 is even:

| Model | greedy | mlp_8.2M | token_3.5M |
|---|---|---|---|
| `token_13.5M` | 0.871 | 0.739 | 0.661 |
| `token_11.8M` | 0.857 | 0.700 | 0.610 |
| `token_6.3M` | 0.862 | 0.734 | 0.548 |
| `token_4.3M` | 0.863 | 0.713 | 0.560 |
| `mlp_19.3M` | 0.885 | 0.694 | 0.479 |
| `mlp_10.9M` | 0.849 | 0.543 | 0.386 |

Both networks plateaued under self-play. The token network's ratings stayed
between 1417 and 1434 from 4.3M to 10.2M steps (`token_6.3M` was its plateau
best), then rose to 1440 at 11.1M and 11.8M steps after the opponent pool was
capped to the newest 25,000 teams, and to 1459 at 13.5M steps after a
further cap to 15,000 teams (13,000 self-play and 2,000 greedy bot teams);
`token_13.5M` is its best so far. The MLP needed about three times as many steps to get there: its
ratings stayed between 1407 and 1420 from 18.2M to 23.6M steps, so
`mlp_19.3M` is its plateau best, close to the token network's level.

In normal games against the greedy bot (10 wins to win a game) the token
models win about 95-99% of games and `mlp_8.2M` about 87-90%; the greedy bot
playing the same games wins 25%.

## Using a model

The models need `torch`, `gymnasium` and `sb3-contrib`.

```bash
python -m sapai.rl.evaluate --model models/token_13.5M.zip --games 100
python -m sapai.rl.benchmark --players greedy models/token_13.5M.zip models/mlp_8.2M.zip
```

```python
from sb3_contrib import MaskablePPO
from sapai.rl.env import SAPEnv

model = MaskablePPO.load("models/token_13.5M.zip", device="cpu")
env = SAPEnv()  # plays against the live greedy bot
obs, info = env.reset(seed=0)
done = False
while not done:
    action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
print(info)
```

To continue training a model, pass it to `python -m sapai.rl.train --resume`.
The models depend on the observation and action layout in `sapai/rl/env.py`
and on the pets and foods in `sapai/data.py`; changing either means
retraining.

## Search experiments

`sapai.rl.search` adds a search on top of a policy: for each shop decision
the policy's top moves are played forward to the end of the turn and scored,
and the best score (plus a bonus for the policy's own preference) is taken.
Starting from `token_11.8M`, no scoring tried beat the plain policy in
head-to-head play (60 games x 15 turns, 0.5 is even):

| Scoring of end of turn teams | Policy weight | Score vs `token_11.8M` |
|---|---|---|
| Learned value network (`runs/search1`, 2,400 games of training) | 3.0 | 0.44-0.51 |
| 16 simulated battles against pool teams | 3.0 | 0.513 |
| 16 simulated battles + value network | 3.0 | 0.468 |
| 16 simulated battles | 1.0 | 0.476 |
| 16 simulated battles | 0.5 | 0.499 |

The benchmark agreed (search agent 1441-1445, `token_11.8M` 1440). The
opponent is unknown while shopping and most of a game depends on future
shops, so an end of turn team is hard to score better than the policy
already does. Self-play PPO stayed the stronger route.
