# Trained models

MaskablePPO agents (sb3-contrib) trained on the Turtle Pack with
`sapai.rl.train`, first against greedy bot teams and then with self-play.

| File | Policy | Training steps | Benchmark rating |
|---|---|---|---|
| `token_3.6M.zip` | token network (`TokenMaskablePolicy`) | 3.6M | 1395 |
| `token_3.5M.zip` | token network (`TokenMaskablePolicy`) | 3.5M | 1393 |
| `mlp_8.2M.zip` | MLP (`MlpPolicy`, 256x256) | 8.2M | 1312 |

Ratings are Elo-style from `sapai.rl.benchmark`: head-to-head matches of 100
games x 15 turns against a pool of frozen models, with the greedy bot fixed
at 1000. For reference, the MLP at 2.3M steps (the first model confirmed to
beat the greedy bot) rated 1089. In head-to-head battles the token models
score about 0.66 against `mlp_8.2M` and about 0.85 against the greedy bot
(battles won plus half of draws).

In normal games against the greedy bot (10 wins to win a game) the token
models win about 95-99% of games and `mlp_8.2M` about 87-90%; the greedy bot
playing the same games wins 25%.

## Using a model

The models need `torch`, `gymnasium` and `sb3-contrib`.

```bash
python -m sapai.rl.evaluate --model models/token_3.6M.zip --games 100
python -m sapai.rl.benchmark --players greedy models/token_3.6M.zip models/mlp_8.2M.zip
```

```python
from sb3_contrib import MaskablePPO
from sapai.rl.env import SAPEnv

model = MaskablePPO.load("models/token_3.6M.zip", device="cpu")
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
