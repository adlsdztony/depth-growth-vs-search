"""CPU-tested component for incumbent-led sharing, not a GKP controller.

The incumbent owns and trains the shared prefix. A candidate owns its added
layer(s), copied output head/std and copied critic. Candidate gradients never
reach the incumbent. Keep updates still move the candidate's input features;
this is neither an independent branch nor a rollback guarantee. On-policy
collectors, path RNG isolation, decision calibration and commit/prune remain
separate integration work.
"""
import copy
from contextlib import contextmanager
import random
import weakref

import numpy as np
import torch
from torch import nn

from growth import optimizer_digest, tensor_digest


@contextmanager
def private_rng(device, seed):
    numpy_state=np.random.get_state();python_state=random.getstate()
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed);np.random.seed(seed % 2**32);random.seed(seed)
            yield
    finally:
        np.random.set_state(numpy_state);random.setstate(python_state)


def clone_critic(critic):
    # Some model wrappers cache a non-leaf latent. The pinned critic normally
    # does not; support that optional cache without adding an incumbent field.
    if not hasattr(critic,'_last_shared_latent'):
        return copy.deepcopy(critic)
    latent=critic._last_shared_latent
    try:
        critic._last_shared_latent=None
        return copy.deepcopy(critic)
    finally:
        critic._last_shared_latent=latent


class ProtectedActor(nn.Module):
    def __init__(self, incumbent, additions=1):
        super().__init__()
        assert type(additions) is int and additions in (0,1) and not incumbent._actor_head_hidden_sizes
        assert not incumbent._conditioned_sigma
        assert isinstance(list(incumbent.actor_body.modules())[-1], nn.ReLU)
        # No stochastic or stateful-normalization prefix is supported here.
        assert all(not isinstance(m,(nn.modules.dropout._DropoutNd,nn.modules.batchnorm._BatchNorm))
                   for m in incumbent.modules())
        self._incumbent_ref=weakref.ref(incumbent)
        self.actor_output=copy.deepcopy(incumbent.actor_output)
        self.sigma_param=nn.Parameter(incumbent.sigma_param.detach().clone())
        self._num_tasks=incumbent._num_tasks
        self._action_dim=incumbent._action_dim
        self._unbounded=incumbent._unbounded
        self._max_action=incumbent._max_action
        self._last_shared_latent=None
        width=self.actor_output.in_features
        self.candidate_body=nn.Sequential()
        for i in range(additions):
            layer=nn.Linear(width,width,bias=False,device=self.actor_output.weight.device,
                            dtype=self.actor_output.weight.dtype)
            with torch.no_grad():layer.weight.copy_(torch.eye(width,device=layer.weight.device,dtype=layer.weight.dtype))
            self.candidate_body.add_module(f'addition_{i}',nn.Sequential(layer,nn.ReLU()))

    @property
    def incumbent(self):
        value=self._incumbent_ref()
        assert value is not None,'Keep the owning ProtectedTrial alive'
        return value

    @property
    def last_shared_latent(self):return self._last_shared_latent

    def get_output_dim(self):return self._action_dim

    def get_preprocess_net(self):return self.incumbent.preprocess

    def forward(self,obs,state=None,info=None):
        x=torch.as_tensor(obs,device=self.actor_output.weight.device,dtype=self.actor_output.weight.dtype)
        if x.ndim==1:x=x.unsqueeze(0)
        with torch.no_grad():
            features,state=self.incumbent.preprocess(x,state,info)
            features=self.incumbent.actor_body(features)
        features=self.candidate_body(features.detach())
        self._last_shared_latent=features
        mu=self.actor_output(features)
        sigma=self.sigma_param[x[:,-self._num_tasks:].argmax(-1)].clamp(-20,2).exp()
        if not self._unbounded:mu=self._max_action*torch.tanh(mu)
        return (mu,sigma),state


class ProtectedTrial(nn.Module):
    """Own both models; shared actor prefix is serialized only with incumbent."""
    def __init__(self,incumbent,candidate):
        super().__init__();self.incumbent=incumbent;self.candidate=candidate

    def checkpoint(self):
        # Tianshou adds optimizer metadata through its custom state_dict/load
        # methods; a parent nn.Module.load_state_dict bypasses that load hook.
        return dict(model=dict(incumbent=self.incumbent.state_dict(),candidate=self.candidate.state_dict()),incumbent_optimizer=self.incumbent._get_raw_optimizer().state_dict(),
                    candidate_optimizer=self.candidate._get_raw_optimizer().state_dict(),
                    scope='Models and Adam only; no simulator or per-path RNG continuation guarantee.')

    def restore(self,saved):
        self.incumbent.load_state_dict(saved['model']['incumbent'])
        self.candidate.load_state_dict(saved['model']['candidate'])
        self.incumbent._get_raw_optimizer().load_state_dict(saved['incumbent_optimizer'])
        self.candidate._get_raw_optimizer().load_state_dict(saved['candidate_optimizer'])


def make_protected_trial(incumbent,builder,additions=1,seed=0,probe=None):
    """builder(actor,critic) must return a fresh pinned PPO with default gradients.

    It must build the optimizer AFTER receiving these supplied private modules.
    It must not collect data or mutate the incumbent. Inherited head/std and
    critic Adam histories are copied; new identity layers start with fresh Adam.
    """
    assert incumbent._actor_method==incumbent._critic_method=='default'
    source=incumbent.policy.actor;device=next(source.parameters()).device
    before_actor=tensor_digest(source.state_dict());before_critic=tensor_digest(incumbent.critic.state_dict())
    old_optimizer=incumbent._get_raw_optimizer()
    before_adam=optimizer_digest(old_optimizer,list(incumbent.parameters()))
    with private_rng(device,seed):
        actor=ProtectedActor(source,additions);critic=clone_critic(incumbent.critic)
        candidate=builder(actor,critic)
    assert candidate.policy.actor is actor and candidate.critic is critic
    assert candidate._actor_method==candidate._critic_method=='default'
    new_optimizer=candidate._get_raw_optimizer()
    assert new_optimizer.param_groups[0]['lr']==old_optimizer.param_groups[0]['lr']
    old_ids={id(p) for p in incumbent.parameters()}
    new_ids={id(p) for p in candidate.parameters()}
    assert not old_ids & new_ids,'Shared prefix must not be registered as a candidate child'
    assert new_ids=={id(p) for g in new_optimizer.param_groups for p in g['params']}
    for old_module,new_module in [(source,actor),(incumbent.critic,critic)]:
        old=dict(old_module.named_parameters())
        for name,parameter in new_module.named_parameters():
            if name in old and old[name] in old_optimizer.state:
                assert old[name].shape==parameter.shape
                new_optimizer.state[parameter]=copy.deepcopy(old_optimizer.state[old[name]])
    assert before_actor==tensor_digest(source.state_dict())
    assert before_critic==tensor_digest(incumbent.critic.state_dict())
    assert before_adam==optimizer_digest(old_optimizer,list(incumbent.parameters()))
    if probe is not None:
        with torch.no_grad():
            a=source(probe)[0];b=actor(probe)[0]
            assert torch.allclose(a[0],b[0],atol=1e-6,rtol=1e-6) and torch.equal(a[1],b[1])
            assert torch.equal(incumbent.critic(probe),critic(probe))
    return ProtectedTrial(incumbent,candidate)


def unique_tensor_bytes(module):
    storages={}
    for tensor in list(module.parameters())+list(module.buffers()):
        storage=tensor.untyped_storage()
        storages[(str(tensor.device),storage.data_ptr())]=storage.nbytes()
    return sum(storages.values())
