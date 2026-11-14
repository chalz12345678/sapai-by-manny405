"""
Run the trained MaskablePPO policies in models/ with numpy only.

The browser GUI cannot use torch, so ``gui/export_models.py`` turns each
checkpoint's actor weights into an ``.npz`` file and this module reproduces the
actor's forward pass: ``TokenNet`` from ``sapai/rl/policy.py`` (a small pre-norm
transformer with per-action heads) or SB3's ``MlpPolicy`` (two tanh layers).
``predict`` matches ``MaskablePPO.predict(..., deterministic=True)``: the
highest-scoring legal action.
"""

import numpy as np

P = "mlp_extractor."


def _lin(w, b, x):
    y = x @ w.T
    return y if b is None else y + b


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


class NumpyPolicy:
    def __init__(self, weights, n_heads=4, layout=None):
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.kind = "token" if P + "team_in.weight" in self.w else "mlp"
        self.n_heads = n_heads
        self.layout = layout  # dict of env constants, needed for token nets
        if self.kind == "token":
            self.n_layers = len(
                {k.split(".")[3] for k in self.w if ".encoder.layers." in k}
            )

    # ---------------------------------------------------------------- MLP
    def _mlp_logits(self, obs):
        w = self.w
        h = np.tanh(_lin(w[P + "policy_net.0.weight"], w[P + "policy_net.0.bias"], obs))
        h = np.tanh(_lin(w[P + "policy_net.2.weight"], w[P + "policy_net.2.bias"], h))
        return _lin(w["action_net.weight"], w["action_net.bias"], h)

    # -------------------------------------------------------------- token
    def g(self, name):
        return self.w[P + name]

    def L(self, name, x):
        return _lin(self.g(name + ".weight"), self.w.get(P + name + ".bias"), x)

    def _attn(self, x, pad, i):
        pre = f"encoder.layers.{i}.self_attn."
        n, d = x.shape
        h = self.n_heads
        qkv = _lin(self.g(pre + "in_proj_weight"), self.g(pre + "in_proj_bias"), x)
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(n, h, d // h).transpose(1, 0, 2)
        k = k.reshape(n, h, d // h).transpose(1, 0, 2)
        v = v.reshape(n, h, d // h).transpose(1, 0, 2)
        s = q @ k.transpose(0, 2, 1) / np.sqrt(d // h)
        s = np.where(pad[None, None, :], -np.inf, s)
        s = s - s.max(-1, keepdims=True)
        a = np.exp(s)
        a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(n, d)
        return self.L(pre + "out_proj", o)

    def _pair(self, name, x, y):
        hx = self.L(name + ".x", x)
        hy = self.L(name + ".y", y)
        hh = np.maximum(hx[:, None, :] + hy[None, :, :], 0)
        return self.L(name + ".out", hh)[..., 0]

    def _token_logits(self, obs):
        c = self.layout
        MT, MS, TT, ST, GT = (
            c["MAX_TEAM"],
            c["MAX_SHOP"],
            c["TEAM_TOKEN"],
            c["SHOP_TOKEN"],
            c["GLOBAL_TOKEN"],
        )
        team = obs[: MT * TT].reshape(MT, TT)
        shop = obs[MT * TT : MT * TT + MS * ST].reshape(MS, ST)
        glob = obs[-GT:]
        team_present = team[:, c["TEAM_PRESENT"]] > 0.5
        shop_present = shop[:, : c["N_SHOP_KINDS"]].sum(-1) > 0.5
        x = np.concatenate(
            [
                self.L("global_in", glob)[None, :],
                self.L("team_in", team) + self.g("team_pos"),
                self.L("shop_in", shop) + self.g("shop_pos"),
            ]
        )
        pad = np.concatenate([[False], ~team_present, ~shop_present])
        for i in range(self.n_layers):
            pre = f"encoder.layers.{i}."
            x = x + self._attn(
                _ln(x, self.g(pre + "norm1.weight"), self.g(pre + "norm1.bias")), pad, i
            )
            hdn = _ln(x, self.g(pre + "norm2.weight"), self.g(pre + "norm2.bias"))
            x = x + self.L(pre + "linear2", np.maximum(self.L(pre + "linear1", hdn), 0))
        x = _ln(x, self.g("norm.weight"), self.g("norm.bias"))
        g, t, s = x[0], x[1 : 1 + MT], x[1 + MT :]
        pa, pb = np.array(c["PAIR_A"]), np.array(c["PAIR_B"])
        return np.concatenate(
            [
                self.L("global_head", g),
                self.L("buy_pet", s)[:, 0],
                self._pair("buy_combine", s, t).reshape(-1),
                self._pair("buy_food", s, t).reshape(-1),
                self.L("buy_food_untargeted", s)[:, 0],
                self.L("sell", t)[:, 0],
                self.L("freeze", s)[:, 0],
                self._pair("swap", t, t)[pa, pb],
                self._pair("combine", t, t)[pa, pb],
            ]
        )

    # -------------------------------------------------------------- public
    def logits(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return (
            self._token_logits(obs) if self.kind == "token" else self._mlp_logits(obs)
        )

    def value(self, obs):
        """The critic's estimate of future reward from this position: roughly
        battles won minus battles lost from here on, plus a bonus for winning
        the game. None if the export has no critic weights."""
        obs = np.asarray(obs, dtype=np.float32)
        w = self.w
        try:
            if self.kind == "token":
                h = np.tanh(_lin(w[P + "value.0.weight"], w[P + "value.0.bias"], obs))
                h = np.tanh(_lin(w[P + "value.2.weight"], w[P + "value.2.bias"], h))
            else:
                h = np.tanh(
                    _lin(w[P + "value_net.0.weight"], w[P + "value_net.0.bias"], obs)
                )
                h = np.tanh(
                    _lin(w[P + "value_net.2.weight"], w[P + "value_net.2.bias"], h)
                )
            return float(_lin(w["value_net.weight"], w["value_net.bias"], h)[0])
        except KeyError:
            return None

    def predict(self, obs, action_masks):
        z = self.logits(obs).astype(np.float64)
        z[~np.asarray(action_masks, dtype=bool)] = -np.inf
        return int(np.argmax(z)), z
