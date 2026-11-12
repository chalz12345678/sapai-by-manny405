"""
Token based policy for MaskablePPO.

Each team slot, shop slot and the global state becomes a token: the one-hot
pet/food/perk ids are embedded by a linear layer and combined with the slot's
numbers. Team and shop tokens get position embeddings. A small transformer lets
the tokens attend to each other, then each action is scored from the tokens it
involves, for example "merge shop pet i into team pet j" is scored from shop
token i and team token j. What the network learns about a pet therefore carries
over to every slot.

Use with:
    MaskablePPO(TokenMaskablePolicy, env, policy_kwargs={"d_model": 128})
"""

import torch as th
from torch import nn

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from sapai.rl.env import (
    GLOBAL_TOKEN,
    MAX_SHOP,
    MAX_TEAM,
    N_ACTIONS,
    N_PETS,
    N_STATUSES,
    OBS_SIZE,
    PAIRS,
    SHOP_KINDS,
    SHOP_TOKEN,
    TEAM_TOKEN,
)

TEAM_PRESENT = N_PETS + N_STATUSES


class PairHead(nn.Module):
    """Scores every pair (x_i, y_j) from the two tokens"""

    def __init__(self, d_model, hidden=64):
        super().__init__()
        self.x = nn.Linear(d_model, hidden)
        self.y = nn.Linear(d_model, hidden, bias=False)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, y):
        h = th.relu(self.x(x)[:, :, None, :] + self.y(y)[:, None, :, :])
        return self.out(h).squeeze(-1)  # (B, n_x, n_y)


class TokenNet(nn.Module):
    """
    Used as the policy's mlp_extractor. forward_actor returns one logit per
    action, forward_critic returns a latent vector for the value head.
    """

    def __init__(
        self,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dim_feedforward=256,
        separate_critic=True,
        critic_hidden=256,
    ):
        super().__init__()
        self.latent_dim_pi = N_ACTIONS
        self.separate_critic = separate_critic
        self.latent_dim_vf = critic_hidden if separate_critic else d_model

        self.team_in = nn.Linear(TEAM_TOKEN, d_model)
        self.shop_in = nn.Linear(SHOP_TOKEN, d_model)
        self.global_in = nn.Linear(GLOBAL_TOKEN, d_model)
        self.team_pos = nn.Parameter(th.zeros(MAX_TEAM, d_model))
        self.shop_pos = nn.Parameter(th.zeros(MAX_SHOP, d_model))
        nn.init.normal_(self.team_pos, std=0.02)
        nn.init.normal_(self.shop_pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

        self.global_head = nn.Linear(d_model, 2)  # end turn, roll
        self.buy_pet = nn.Linear(d_model, 1)
        self.buy_combine = PairHead(d_model)
        self.buy_food = PairHead(d_model)
        self.buy_food_untargeted = nn.Linear(d_model, 1)
        self.sell = nn.Linear(d_model, 1)
        self.freeze = nn.Linear(d_model, 1)
        self.swap = PairHead(d_model)
        self.combine = PairHead(d_model)
        self.register_buffer("pair_a", th.tensor([a for a, b in PAIRS]))
        self.register_buffer("pair_b", th.tensor([b for a, b in PAIRS]))

        if separate_critic:
            ### The critic is its own MLP on the raw observation, so value
            ###   learning does not reshape the actor's transformer
            self.value = nn.Sequential(
                nn.Linear(OBS_SIZE, critic_hidden),
                nn.Tanh(),
                nn.Linear(critic_hidden, critic_hidden),
                nn.Tanh(),
            )
        else:
            self.value = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
            )

        ### Small initial action scores so the starting policy is close to
        ###   uniform over legal actions, like SB3 does for its action layer
        for head in [
            self.global_head,
            self.buy_pet,
            self.buy_food_untargeted,
            self.sell,
            self.freeze,
        ] + [x.out for x in [self.buy_combine, self.buy_food, self.swap, self.combine]]:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def encode(self, obs):
        b = obs.shape[0]
        n_team = MAX_TEAM * TEAM_TOKEN
        team = obs[:, :n_team].reshape(b, MAX_TEAM, TEAM_TOKEN)
        shop = obs[:, n_team : n_team + MAX_SHOP * SHOP_TOKEN].reshape(
            b, MAX_SHOP, SHOP_TOKEN
        )
        glob = obs[:, -GLOBAL_TOKEN:]

        team_present = team[:, :, TEAM_PRESENT] > 0.5
        shop_present = shop[:, :, : len(SHOP_KINDS)].sum(-1) > 0.5
        tokens = th.cat(
            [
                self.global_in(glob)[:, None, :],
                self.team_in(team) + self.team_pos,
                self.shop_in(shop) + self.shop_pos,
            ],
            dim=1,
        )
        ### Empty slots are ignored by attention, the global token never is
        padding = th.cat(
            [th.zeros_like(team_present[:, :1]), ~team_present, ~shop_present], dim=1
        )
        h = self.norm(self.encoder(tokens, src_key_padding_mask=padding))
        g = h[:, 0]
        t = h[:, 1 : 1 + MAX_TEAM]
        s = h[:, 1 + MAX_TEAM :]
        return g, t, s, team_present

    def logits(self, g, t, s):
        b = g.shape[0]
        swap = self.swap(t, t)[:, self.pair_a, self.pair_b]
        combine = self.combine(t, t)[:, self.pair_a, self.pair_b]
        ### Same order as sapai.rl.env.ACTIONS
        return th.cat(
            [
                self.global_head(g),
                self.buy_pet(s).squeeze(-1),
                self.buy_combine(s, t).reshape(b, -1),
                self.buy_food(s, t).reshape(b, -1),
                self.buy_food_untargeted(s).squeeze(-1),
                self.sell(t).squeeze(-1),
                self.freeze(s).squeeze(-1),
                swap,
                combine,
            ],
            dim=1,
        )

    def value_latent(self, g, t, team_present):
        weights = team_present.float()[:, :, None]
        team_mean = (t * weights).sum(1) / weights.sum(1).clamp(min=1)
        return self.value(th.cat([g, team_mean], dim=1))

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        g, t, s, present = self.encode(features)
        return self.logits(g, t, s)

    def forward_critic(self, features):
        if self.separate_critic:
            return self.value(features)
        g, t, s, present = self.encode(features)
        return self.value_latent(g, t, present)


class TokenMaskablePolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy that uses TokenNet to score each action directly"""

    def __init__(
        self,
        *args,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dim_feedforward=256,
        separate_critic=True,
        critic_hidden=256,
        **kwargs,
    ):
        object.__setattr__(
            self,
            "token_kwargs",
            dict(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dim_feedforward=dim_feedforward,
                separate_critic=separate_critic,
                critic_hidden=critic_hidden,
            ),
        )
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = TokenNet(**self.token_kwargs)

    def _build(self, lr_schedule):
        self._build_mlp_extractor()
        ### TokenNet already outputs one logit per action
        self.action_net = nn.Identity()
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(self.token_kwargs)
        return data
