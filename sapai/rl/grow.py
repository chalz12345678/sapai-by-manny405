"""
Make a trained token model deeper without changing how it plays.

    python -m sapai.rl.grow models/token_16.8M.zip runs/deep_start.zip --n-layers 4

The new transformer layers are added after the trained ones with their output
projections set to zero. The layers are pre-norm residual blocks
(x + attention(norm(x)), then x + feedforward(norm(x))), so a block whose
outputs are zero passes its input through unchanged and the grown model
gives exactly the same actions and values as the original. Training then
switches the new layers on as they become useful. Continue training with
python -m sapai.rl.train --resume <grown model>.
"""

import argparse
import copy

import torch as th


def grow_model(model, n_layers):
    """A copy of a MaskablePPO token model with n_layers transformer layers"""
    from sb3_contrib import MaskablePPO

    old_net = model.policy.mlp_extractor
    old_layers = len(old_net.encoder.layers)
    if n_layers <= old_layers:
        raise ValueError(f"the model already has {old_layers} layers")

    policy_kwargs = dict(model.policy_kwargs)
    policy_kwargs["n_layers"] = n_layers
    if model.env is None:
        raise ValueError("load the model with an env to grow it")
    new = MaskablePPO(
        type(model.policy),
        model.env,
        policy_kwargs=policy_kwargs,
        learning_rate=model.learning_rate,
        n_steps=model.n_steps,
        batch_size=model.batch_size,
        n_epochs=model.n_epochs,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        clip_range=model.clip_range,
        ent_coef=model.ent_coef,
        vf_coef=model.vf_coef,
        max_grad_norm=model.max_grad_norm,
        target_kl=model.target_kl,
        device="cpu",
    )

    missing, unexpected = new.policy.load_state_dict(
        model.policy.state_dict(), strict=False
    )
    assert not unexpected, unexpected
    assert all(".encoder.layers." in k for k in missing), missing

    new_net = new.policy.mlp_extractor
    for layer in new_net.encoder.layers[old_layers:]:
        ### Start as a copy of the last trained layer, with zero output
        layer.load_state_dict(copy.deepcopy(old_net.encoder.layers[-1].state_dict()))
        with th.no_grad():
            layer.self_attn.out_proj.weight.zero_()
            layer.self_attn.out_proj.bias.zero_()
            layer.linear2.weight.zero_()
            layer.linear2.bias.zero_()
    new.num_timesteps = model.num_timesteps
    return new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--n-layers", type=int, default=4)
    args = parser.parse_args(argv)

    from sb3_contrib import MaskablePPO

    from sapai.rl.env import SAPEnv

    model = MaskablePPO.load(args.model, env=SAPEnv(), device="cpu")
    new = grow_model(model, args.n_layers)
    new.save(args.out)
    n = sum(p.numel() for p in new.policy.parameters())
    print(f"Saved {args.out}: {args.n_layers} layers, {n:,} parameters")


if __name__ == "__main__":
    main()
