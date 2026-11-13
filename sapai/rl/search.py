"""
Value network plus search for shopping.

For every shop decision the policy proposes its most likely actions. Each
candidate is applied to a copy of the player and played forward with the
policy to the end of the turn, then a value network scores the end of turn
position: the expected reward for the rest of the game (battle wins and
losses plus the game win bonus). The candidate with the best score is taken.
Rolls are random, so they are scored by averaging a few sampled rolls.

A SearchAgent is saved as a folder with policy.zip (a MaskablePPO model),
value.pt (the ValueNet weights) and agent.json (its settings).
"""

import copy
import json
import os

import numpy as np
import torch as th
from torch import nn

from sapai.data import data
from sapai.player import Player
from sapai.rl.env import (
    ACTION_START,
    ACTIONS,
    END_TURN,
    N_ACTIONS,
    action_mask,
    apply_action,
    encode,
)
from sapai.rl.policy import TokenNet

ROLL = ACTION_START["roll"]
### Values are predicted in units of VALUE_SCALE reward
VALUE_SCALE = 10.0
### Actions not used when playing a candidate forward: rolls are random and
###   freezing back and forth does not change the team
ROLLOUT_SKIP = np.array([name in ["roll", "freeze"] for name, _ in ACTIONS])


class ValueNet(nn.Module):
    """Expected reward for the rest of the game from an end of turn position"""

    def __init__(self, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.kwargs = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        self.net = TokenNet(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, separate_critic=False
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, obs):
        g, t, s, present = self.net.encode(obs)
        return self.head(self.net.value_latent(g, t, present)).squeeze(-1)

    def predict(self, obs):
        """Values in reward units for a batch of observations (numpy)"""
        with th.no_grad():
            return self(th.as_tensor(np.asarray(obs, dtype=np.float32))).numpy() * (
                VALUE_SCALE
            )


class Context:
    """What the observation needs besides the player"""

    def __init__(self, actions=0, last_result=None, wins=0, max_turns=25):
        self.actions = actions
        self.last_result = last_result
        self.wins = wins
        self.max_turns = max_turns

    def encode(self, player, extra_actions=0):
        return encode(
            player,
            self.actions + extra_actions,
            self.last_result,
            self.max_turns,
            self.wins,
        )


class SearchAgent:
    def __init__(
        self,
        policy_model,
        value_net,
        n_candidates=6,
        rollout_depth=6,
        roll_samples=2,
        temperature=0.0,
        max_actions_per_turn=30,
        prior_weight=0.5,
    ):
        self.policy_model = policy_model
        ### score = value + prior_weight * log(policy probability), so the
        ###   search only overrides the policy when the value is clearly better
        self.prior_weight = prior_weight
        self.value_net = value_net
        self.n_candidates = n_candidates
        self.rollout_depth = rollout_depth
        self.roll_samples = roll_samples
        self.temperature = temperature
        self.max_actions_per_turn = max_actions_per_turn
        self.rng = np.random.default_rng()

    ############################################################################
    #### Networks
    ############################################################################

    def policy_probs(self, obs, masks):
        with th.no_grad():
            dist = self.policy_model.policy.get_distribution(
                th.as_tensor(np.asarray(obs, dtype=np.float32)),
                action_masks=np.asarray(masks),
            )
            return dist.distribution.probs.numpy()

    ############################################################################
    #### Search
    ############################################################################

    def choose(self, player, ctx):
        """
        Best shop action for the player. Returns the action and a dict with
        the candidates and their values.
        """
        mask = action_mask(player)
        if ctx.actions >= self.max_actions_per_turn - 1:
            return END_TURN, {}
        probs = self.policy_probs([ctx.encode(player)], [mask])[0]
        legal = np.flatnonzero(mask)
        ranked = legal[np.argsort(-probs[legal], kind="stable")]
        candidates = [int(a) for a in ranked[: self.n_candidates]]
        if END_TURN not in candidates:
            candidates.append(END_TURN)

        ### Leaves: [candidate, player copy, actions taken, still playing]
        leaves = []
        for a in candidates:
            if a == END_TURN:
                leaves.append([a, player, 0, False])
                continue
            for _ in range(self.roll_samples if a == ROLL else 1):
                leaf = copy.deepcopy(player)
                try:
                    apply_action(leaf, a)
                except Exception:
                    continue
                leaves.append([a, leaf, 1, True])

        ### Play each candidate forward with the policy to the end of the turn
        for _ in range(self.rollout_depth):
            active = [x for x in leaves if x[3]]
            if not active:
                break
            obs = [ctx.encode(x[1], x[2]) for x in active]
            masks = [action_mask(x[1]) for x in active]
            step_probs = self.policy_probs(obs, masks)
            for leaf, p, m in zip(active, step_probs, masks):
                p = np.where(m & ~ROLLOUT_SKIP, p, -1.0)
                p[END_TURN] = max(p[END_TURN], 0.0)
                a = int(np.argmax(p))
                if (
                    a == END_TURN
                    or ctx.actions + leaf[2] >= self.max_actions_per_turn - 1
                ):
                    leaf[3] = False
                    continue
                try:
                    apply_action(leaf[1], a)
                except Exception:
                    leaf[3] = False
                    continue
                leaf[2] += 1

        values = self.value_net.predict([ctx.encode(x[1], x[2]) for x in leaves])
        q = {}
        for leaf, v in zip(leaves, values):
            q.setdefault(leaf[0], []).append(float(v))
        q = {a: float(np.mean(v)) for a, v in q.items()}

        actions = list(q)
        scores = np.array(
            [q[a] + self.prior_weight * np.log(max(probs[a], 1e-6)) for a in actions]
        )
        if self.temperature > 0:
            p = np.exp((scores - scores.max()) / (self.temperature * VALUE_SCALE))
            choice = actions[self.rng.choice(len(actions), p=p / p.sum())]
        else:
            ### Ties go to the action the policy prefers
            score = dict(zip(actions, scores))
            best = scores.max()
            choice = next(
                a for a in candidates if a in score and score[a] >= best - 1e-9
            )
        return choice, {"q": q, "prior": {a: float(probs[a]) for a in candidates}}

    def shop_env(self, env, record=None):
        """
        Shop for env.player until the agent ends the turn. The caller ends
        the turn with env.step(END_TURN). record, if given, collects
        (obs, mask, action) of each decision.
        """
        while True:
            ctx = Context(
                env.actions_this_turn, env.last_result, env.wins, env.max_turns
            )
            if env.actions_this_turn >= env.max_actions_per_turn - 1:
                return
            action, _ = self.choose(env.player, ctx)
            if record is not None:
                record.append((ctx.encode(env.player), action_mask(env.player), action))
            if action == END_TURN:
                return
            env.step(action)

    ############################################################################
    #### Saving
    ############################################################################

    def settings(self):
        return dict(
            n_candidates=self.n_candidates,
            rollout_depth=self.rollout_depth,
            roll_samples=self.roll_samples,
            max_actions_per_turn=self.max_actions_per_turn,
            prior_weight=self.prior_weight,
            value_net=self.value_net.kwargs,
        )

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        self.policy_model.save(os.path.join(folder, "policy.zip"))
        th.save(self.value_net.state_dict(), os.path.join(folder, "value.pt"))
        with open(os.path.join(folder, "agent.json"), "w") as f:
            json.dump(self.settings(), f, indent=2)

    @classmethod
    def load(cls, folder, **overrides):
        from sb3_contrib import MaskablePPO

        with open(os.path.join(folder, "agent.json")) as f:
            settings = json.load(f)
        value_net = ValueNet(**settings.pop("value_net"))
        value_net.load_state_dict(
            th.load(os.path.join(folder, "value.pt"), map_location="cpu")
        )
        value_net.eval()
        policy = MaskablePPO.load(os.path.join(folder, "policy.zip"), device="cpu")
        settings.update(overrides)
        return cls(policy, value_net, **settings)


def is_agent_folder(path):
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "agent.json"))


class SearchOpponent:
    """A SearchAgent as a live opponent, like ModelOpponent"""

    def __init__(self, agent, max_turns=25):
        self.agent = agent
        self.max_turns = max_turns
        self.player = None

    def reset(self, rng=None):
        self.player = Player()
        self.wins = 0
        self.last_result = None

    def shop(self):
        p = self.player
        for actions in range(self.agent.max_actions_per_turn):
            ctx = Context(actions, self.last_result, self.wins, self.max_turns)
            action, _ = self.agent.choose(p, ctx)
            if action == END_TURN:
                break
            try:
                apply_action(p, action)
            except Exception:
                pass
        p.team.move_forward()

    def team_for_turn(self, turn):
        self.shop()
        self.player.end_turn()
        self.player.team.move_forward()
        return self.player.team

    def record_result(self, result):
        own_result = {0: 1, 1: 0, 2: 2}[result]
        if own_result == 0:
            self.wins += 1
        elif own_result == 1:
            turn = min(self.player.turn, 11)
            self.player.lives = max(
                self.player.lives - data["turns"][f"turn-{turn}"]["livesLost"], 1
            )
        self.last_result = own_result

    def after_battle(self, result):
        self.record_result(result)
        self.player.start_turn(winner=(self.last_result != 1))
