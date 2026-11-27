"""Native depth2 checkpoint load, one-block removal and paid mean-head imitation.

This module performs no environment creation and no parent selection. It never
claims that actor removal preserves behavior or that a simulator was resumed.
"""
import copy
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from growth import tensor_digest
from protected import private_rng


def equal(left,right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype==right.dtype and torch.equal(left.detach().cpu(),right.detach().cpu())
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return isinstance(right,(list,tuple)) and len(left)==len(right) and all(equal(a,b) for a,b in zip(left,right))
    return left==right


def named_adam(algorithm):
    optimizer=algorithm._get_raw_optimizer()
    return {name:copy.deepcopy(optimizer.state[p]) for name,p in algorithm.named_parameters() if p in optimizer.state}


def named_steps(algorithm):
    return {name:int(value['step'].item()) for name,value in named_adam(algorithm).items() if 'step' in value}


def model_digest(algorithm):
    return dict(actor=tensor_digest(algorithm.policy.actor.state_dict()),critic=tensor_digest(algorithm.critic.state_dict()))


def construct_depth2(factory, private_preprocess=False):
    """Construct adaptive_1 BEFORE native PPO creates its single optimizer."""
    original=factory.train.MultiTaskActor
    original_critic=factory.train.MultiTaskCritic
    def actor_constructor(**kwargs):
        actor=original(**kwargs)
        assert list(actor.actor_body._modules)==['0','1']
        width=actor.actor_output.in_features
        layer=nn.Linear(width,width,bias=False,device=actor.actor_output.weight.device,dtype=actor.actor_output.weight.dtype)
        with torch.no_grad():layer.weight.copy_(torch.eye(width,device=layer.weight.device,dtype=layer.weight.dtype))
        actor.actor_body.add_module('adaptive_1',nn.Sequential(layer,nn.ReLU()))
        return actor
    def critic_constructor(**kwargs):
        if private_preprocess:kwargs['preprocess_net']=copy.deepcopy(kwargs['preprocess_net'])
        return original_critic(**kwargs)
    with patch.object(factory.train,'MultiTaskActor',actor_constructor),patch.object(factory.train,'MultiTaskCritic',critic_constructor):algorithm=factory()
    raw=algorithm._get_raw_optimizer()
    assert len(raw.param_groups)==1
    assert [id(p) for p in raw.param_groups[0]['params']]==[id(p) for p in algorithm.parameters()]
    return algorithm


def load_parent(factory,path,device='cpu'):
    path=Path(path);saved=torch.load(path,map_location=device,weights_only=False)
    expected=saved['optimizer'];assert len(expected['param_groups'])==1
    parameter_count=len(expected['param_groups'][0]['params'])
    assert parameter_count in (20,21),'Only native single-group depth2 checkpoints supported'
    private_preprocess=parameter_count==21
    algorithm=construct_depth2(factory,private_preprocess)
    names=[name for name,_ in algorithm.named_parameters()]
    assert [name for name in saved['algorithm'] if name in set(names)]==names,'Native parameter ordering must match checkpoint'
    assert len(expected['param_groups'][0]['params'])==len(names)==parameter_count
    # The native hook handles _optimizers and its persistent learner metadata.
    algorithm.load_state_dict(saved['algorithm'])
    algorithm._get_raw_optimizer().load_state_dict(expected)
    assert equal(expected,algorithm._get_raw_optimizer().state_dict()),'All Adam states/options must survive native load'
    for name,p in algorithm.named_parameters():assert equal(p,saved['algorithm'][name])
    for prefix,module in [('policy.actor.',algorithm.policy.actor),('critic.',algorithm.critic)]:
        state={name[len(prefix):]:value for name,value in saved['algorithm'].items() if name.startswith(prefix)}
        assert equal(state,module.state_dict()),'All model and PopArt buffers must match parent'
    assert sum(p.numel() for p in algorithm.policy.actor.parameters())==181644
    assert sum(p.numel() for p in algorithm.critic.parameters())==342421
    history=saved.get('state',{})
    metadata=dict(parent_checkpoint=str(path),parent_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parent_training_steps=history.get('training_steps'),parent_recorded_physical_steps=history.get('total_interactions'),
        loaded_parent_fingerprints=model_digest(algorithm),loaded_parent_named_adam_steps=named_steps(algorithm),
        optimizer_groups=1,optimizer_parameter_names_in_order=names,private_actor_critic_preprocess=private_preprocess,
        named_parameters_without_adam=[name for name,p in algorithm.named_parameters() if p not in algorithm._get_raw_optimizer().state],
        simulator_resume=False,parent_rng_restored=False)
    return algorithm,metadata


def clone_teacher(actor):
    latent=actor._last_shared_latent
    try:actor._last_shared_latent=None;teacher=copy.deepcopy(actor)
    finally:actor._last_shared_latent=latent
    teacher.eval()
    for p in teacher.parameters():p.requires_grad_(False)
    return teacher


def remove_optional_block(algorithm,probe):
    actor=algorithm.policy.actor;raw=algorithm._get_raw_optimizer()
    assert list(actor.actor_body._modules)==['0','1','adaptive_1']
    block=actor.actor_body.adaptive_1
    assert isinstance(block,nn.Sequential) and isinstance(block[0],nn.Linear) and block[0].bias is None
    assert tuple(block[0].weight.shape)==(400,400) and isinstance(block[1],nn.ReLU)
    old_model={name:p.detach().clone() for name,p in algorithm.named_parameters()};old_adam=named_adam(algorithm)
    old_critic=copy.deepcopy(algorithm.critic.state_dict())
    device=next(algorithm.parameters()).device;x=torch.as_tensor(probe,device=device,dtype=torch.float32)
    with torch.no_grad():before_value=algorithm.critic(x).detach().clone()
    removed_names=[name for name in old_model if name.startswith('policy.actor.actor_body.adaptive_1.')]
    assert removed_names==['policy.actor.actor_body.adaptive_1.0.weight']
    del actor.actor_body._modules['adaptive_1']
    retained={id(p) for p in algorithm.parameters()}
    for group in raw.param_groups:group['params']=[p for p in group['params'] if id(p) in retained]
    for p in list(raw.state):
        if id(p) not in retained:del raw.state[p]
    assert len(raw.param_groups)==1 and len(raw.param_groups[0]['params'])==len(list(algorithm.parameters()))==len(old_model)-1
    assert [id(p) for p in raw.param_groups[0]['params']]==[id(p) for p in algorithm.parameters()]
    assert {id(p) for p in algorithm.optim._module.parameters()}==retained
    for name,p in algorithm.named_parameters():assert torch.equal(p,old_model[name])
    expected={name:value for name,value in old_adam.items() if name not in removed_names}
    assert equal(expected,named_adam(algorithm)) and equal(old_critic,algorithm.critic.state_dict())
    with torch.no_grad():after_value=algorithm.critic(x)
    assert torch.equal(before_value,after_value),'Actor pruning must never jump the critic function'
    return dict(removed_parameters=160000,removed_names=removed_names,surviving_adam_exact=True,
        critic_function_max_error=float((before_value-after_value).abs().max()),
        actor_parameters=sum(p.numel() for p in actor.parameters()),critic_parameters=sum(p.numel() for p in algorithm.critic.parameters()),
        named_adam_steps_after_removal=named_steps(algorithm))


def distill_mean_head(algorithm,teacher,observations,steps=64,batch_size=2048,seed=9492,check=lambda:None):
    """Inherited native Adam is retained; only output mean-head state advances."""
    assert steps>=0 and len(observations)>0
    actor=algorithm.policy.actor;raw=algorithm._get_raw_optimizer();device=next(actor.parameters()).device
    teacher_before=tensor_digest(teacher.state_dict());critic_before=copy.deepcopy(algorithm.critic.state_dict())
    before_adam=named_adam(algorithm);before_model={name:p.detach().clone() for name,p in algorithm.named_parameters()}
    flags={id(p):p.requires_grad for p in algorithm.parameters()}
    head_names={name for name,_ in algorithm.named_parameters() if name.startswith('policy.actor.actor_output.')}
    head=list(actor.actor_output.parameters())
    for name,p in algorithm.named_parameters():p.requires_grad_(name in head_names)
    obs=np.asarray(observations);task_ids=obs[:,-10:].argmax(-1)
    probe_indices=np.concatenate([np.flatnonzero(task_ids==i)[:200] for i in range(10)])
    assert len(probe_indices)>0
    probe=torch.as_tensor(obs[probe_indices],device=device,dtype=torch.float32)
    def loss_on_probe():
        with torch.no_grad():return float(torch.mean((actor(probe)[0][0]-teacher(probe)[0][0])**2))
    result=dict(optimizer_steps=0,mean_loss_before=loss_on_probe(),losses=[],head_adam_policy='inherited PPO Adam retained; only head mean updated')
    try:
        with private_rng(device,seed):
            for _ in range(steps):
                check();indices=np.random.randint(0,len(observations),size=batch_size)
                x=torch.as_tensor(np.asarray(observations)[indices],device=device,dtype=torch.float32)
                with torch.no_grad():target=teacher(x)[0][0]
                raw.zero_grad(set_to_none=True);prediction=actor(x)[0][0];loss=torch.mean((prediction-target)**2)
                assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(head,1.0);raw.step()
                result['optimizer_steps']+=1;result['losses'].append(float(loss.detach()))
    finally:
        raw.zero_grad(set_to_none=True)
        for p in algorithm.parameters():p.requires_grad_(flags[id(p)])
    result['mean_loss_after']=loss_on_probe()
    assert teacher_before==tensor_digest(teacher.state_dict()) and equal(critic_before,algorithm.critic.state_dict())
    after_adam=named_adam(algorithm)
    for name,p in algorithm.named_parameters():
        if name not in head_names:
            assert torch.equal(p,before_model[name]) and equal(before_adam.get(name),after_adam.get(name))
    result.update(teacher_unchanged=True,critic_prefix_std_weights_and_adam_unchanged=True,
                  named_adam_steps_after_distillation=named_steps(algorithm))
    return result
