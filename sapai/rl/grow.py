"""
Change a trained model's network without changing how it plays.

    python -m sapai.rl.grow models/token_22.5M.zip runs/start.zip --upgrade
    python -m sapai.rl.grow models/token_16.8M.zip runs/deep.zip --n-layers 4

--upgrade gives a model trained before matches were 1v1 the full observation,
which includes the opponent's last team, lives and wins. The new inputs start
with zero weights (the token network's opponent summary starts at zero, and
the extra input columns of the critic and of MLP layers are zero), so the
upgraded model gives exactly the same actions and values until training
gives them a use.

--n-layers makes a token model deeper. The new transformer layers are added
after the trained ones with their output projections set to zero. The layers
are pre-norm residual blocks (x + attention(norm(x)), then x +
feedforward(norm(x))), so a block whose outputs are zero passes its input
through unchanged.

Both can be combined. Continue training with
python -m sapai.rl.train --resume <new model>.
"""

import argparse
import copy

import torch as th


def rebuild(model, observation_space=None, **policy_kwargs):
    """
    The model with a new policy network: new observation space and/or policy
    settings. Weights are copied where shapes match; a weight whose input
    grew (more observation values) keeps its trained columns and gets zeros
    for the new ones. Anything new keeps the network's own starting values.
    Returns (model, names of weights that were not copied).
    """
    new = copy.copy(model)
    if observation_space is not None:
        new.observation_space = observation_space
    new.policy_kwargs = dict(model.policy_kwargs, **policy_kwargs)
    new.policy = model.policy_class(
        new.observation_space,
        model.action_space,
        model.lr_schedule,
        **new.policy_kwargs,
    )
    old_state = model.policy.state_dict()
    state = new.policy.state_dict()
    fresh = []
    for name, value in state.items():
        old = old_state.get(name)
        if old is None:
            fresh.append(name)
        elif old.shape == value.shape:
            state[name] = old.clone()
        elif (
            old.dim() == 2
            and old.shape[0] == value.shape[0]
            and old.shape[1] < value.shape[1]
        ):
            ### Input grew: keep the trained columns, new inputs start at zero
            grown = th.zeros_like(value)
            grown[:, : old.shape[1]] = old
            state[name] = grown
        else:
            fresh.append(name)
    new.policy.load_state_dict(state)
    new.policy.optimizer = new.policy.optimizer_class(
        new.policy.parameters(), lr=model.lr_schedule(1), **new.policy.optimizer_kwargs
    )
    return new, fresh


def upgrade_model(model):
    """A model trained before matches were 1v1, given the full observation"""
    from sapai.rl.env import SAPEnv

    space = SAPEnv().observation_space
    if model.observation_space.shape == space.shape:
        raise ValueError("the model already sees the opponent")
    new, fresh = rebuild(model, observation_space=space)
    ### Only the opponent summary may start fresh (it starts at zero)
    assert all(".opp_" in n for n in fresh), fresh
    return new


def grow_model(model, n_layers):
    """A copy of a token model with n_layers transformer layers"""
    old_net = model.policy.mlp_extractor
    old_layers = len(old_net.encoder.layers)
    if n_layers <= old_layers:
        raise ValueError(f"the model already has {old_layers} layers")
    new, fresh = rebuild(model, n_layers=n_layers)
    assert all(".encoder.layers." in n for n in fresh), fresh

    new_net = new.policy.mlp_extractor
    for layer in new_net.encoder.layers[old_layers:]:
        ### Start as a copy of the last trained layer, with zero output
        layer.load_state_dict(copy.deepcopy(old_net.encoder.layers[-1].state_dict()))
        with th.no_grad():
            layer.self_attn.out_proj.weight.zero_()
            layer.self_attn.out_proj.bias.zero_()
            layer.linear2.weight.zero_()
            layer.linear2.bias.zero_()
    return new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument(
        "--upgrade", action="store_true", help="give the model the full observation"
    )
    parser.add_argument("--n-layers", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.upgrade and args.n_layers is None:
        parser.error("give --upgrade and/or --n-layers")

    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(args.model, device="cpu")
    if args.upgrade:
        model = upgrade_model(model)
    if args.n_layers is not None:
        model = grow_model(model, args.n_layers)
    model.save(args.out)
    n = sum(p.numel() for p in model.policy.parameters())
    print(
        f"Saved {args.out}: observation {model.observation_space.shape[0]},"
        f" {n:,} parameters"
    )


if __name__ == "__main__":
    main()
