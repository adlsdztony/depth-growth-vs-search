"""Net2Deeper actor growth for the pinned, competent multi-task PPO learner."""
import hashlib
import torch
from torch import nn


def tensor_digest(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(key.encode())
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode())
            digest.update(value.numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def optimizer_digest(optimizer, params):
    flat = {}
    for i, parameter in enumerate(params):
        for key, value in optimizer.state.get(parameter, {}).items():
            flat[f'{i}:{key}'] = value
    return tensor_digest(flat)


def append_layer(algorithm, kind='identity', seed=0, probe=None):
    """Add a bias-free hidden layer; preserve survivor tensors, Adam and RNG.

    The optimizer's clipping module retains the same actor object, so its
    dynamic parameter iterator must include the newly registered module.
    """
    assert kind in {'identity', 'random'}
    assert algorithm._actor_method == algorithm._critic_method == 'default'
    actor = algorithm.policy.actor
    assert isinstance(actor.actor_body, nn.Sequential)
    assert isinstance(list(actor.actor_body.children())[-1], (nn.ReLU, nn.Sequential))
    linear = [m for m in actor.actor_body.modules() if isinstance(m, nn.Linear)][-1]
    width = linear.out_features
    device, dtype = linear.weight.device, linear.weight.dtype
    optimizer = algorithm._get_raw_optimizer()
    survivors = list(algorithm.parameters())
    before_optimizer = optimizer_digest(optimizer, survivors)
    before_critic = tensor_digest(algorithm.critic.state_dict())
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else []
    if probe is not None:
        x = torch.as_tensor(probe, device=device, dtype=dtype)
        with torch.no_grad():
            before_actor = [v.detach().clone() for v in actor(x)[0]]
            before_value = algorithm.critic(x).detach().clone()
    else:
        before_actor = before_value = None
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        layer = nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        if kind == 'identity':
            with torch.no_grad():
                layer.weight.copy_(torch.eye(width, device=device, dtype=dtype))
    index = sum(isinstance(m, nn.Linear) for m in actor.actor_body.modules())
    actor.actor_body.add_module(f'growth_{index}', nn.Sequential(layer, nn.ReLU()))
    optimizer.add_param_group({'params': list(layer.parameters()),
                              'lr': optimizer.param_groups[0]['lr'], 'growth_index': index})
    all_parameters = list(algorithm.parameters())
    opt_parameters = [p for group in optimizer.param_groups for p in group['params']]
    assert len(opt_parameters) == len({id(p) for p in opt_parameters})
    assert {id(p) for p in all_parameters} == {id(p) for p in opt_parameters}
    clipping_parameters = list(algorithm.optim._module.parameters())
    assert {id(p) for p in layer.parameters()} <= {id(p) for p in clipping_parameters}
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else []))
    assert before_optimizer == optimizer_digest(optimizer, survivors)
    assert before_critic == tensor_digest(algorithm.critic.state_dict())
    result = dict(kind=kind, added_parameters=sum(p.numel() for p in layer.parameters()),
                  actor_parameters=sum(p.numel() for p in actor.parameters()),
                  survivor_adam_preserved=True, critic_tensors_preserved=True,
                  global_rng_preserved=True, new_parameters_clipped=True,
                  hidden_depth=index + 1)
    if probe is not None:
        with torch.no_grad():
            after_actor = actor(x)[0]
            after_value = algorithm.critic(x)
        result['mean_max_error'] = float((before_actor[0] - after_actor[0]).abs().max())
        result['std_max_error'] = float((before_actor[1] - after_actor[1]).abs().max())
        result['critic_max_error'] = float((before_value - after_value).abs().max())
        if kind == 'identity':
            assert torch.allclose(before_actor[0], after_actor[0], atol=1e-6, rtol=1e-6), result
            assert torch.equal(before_actor[1], after_actor[1]), result
        assert torch.equal(before_value, after_value), result
    return result


def block_diagnostics(algorithm):
    actor = algorithm.policy.actor
    result = []
    for name, block in actor.actor_body.named_children():
        if not name.startswith('growth_'):
            continue
        weight = block[0].weight.detach()
        identity = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
        parameter = block[0].weight
        state = algorithm._get_raw_optimizer().state.get(parameter, {})
        result.append(dict(name=name, identity_delta=float(torch.linalg.vector_norm(weight - identity)),
                           adam_steps=int(state.get('step', 0)), parameters=weight.numel()))
    return result
