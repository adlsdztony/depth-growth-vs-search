"""Read-only named-Adam audit via native module/group reconstruction.
Saved IDs are never paired with state_dict tensor order. Unsupported layouts fail.
"""
import copy
from pathlib import Path
import re
from unittest.mock import patch
import torch
from torch import nn
from backend import NativeFactory
from growth import append_layer
from prune_ops import equal,named_steps,model_digest


def load_native_checkpoint(path,author_root,device='cpu',task_names=None,factory=None):
    path=Path(path);saved=torch.load(path,map_location=device,weights_only=False)
    state=saved['algorithm'];expected=saved['optimizer']
    if task_names is None:task_names=saved.get('state',{}).get('task_names')
    if factory is None:factory=NativeFactory(author_root,device=device,task_names=task_names)
    assert factory is not None
    assert not any('.growth_' in k for k in state), 'fresh common parent must be shallow D2, not a prior grown model'
    groups=expected['param_groups']; assert len(groups)==1
    algorithm=factory()
    layout='same fresh native D2 factory; exact sole common Adam group'
    raw=algorithm._get_raw_optimizer();by_id={id(p):name for name,p in algorithm.named_parameters()}
    group_names=[[by_id[id(p)] for p in group['params']] for group in raw.param_groups]
    assert [len(g) for g in group_names]==[len(g['params']) for g in groups]
    assert {n for g in group_names for n in g}==set(by_id.values())
    algorithm.load_state_dict(state);raw.load_state_dict(expected)
    assert equal(raw.state_dict(),expected)
    for name,p in algorithm.named_parameters():assert equal(p,state[name]),name
    for prefix,module in [('policy.actor.',algorithm.policy.actor),('critic.',algorithm.critic)]:
        assert equal(module.state_dict(),{n[len(prefix):]:v for n,v in state.items() if n.startswith(prefix)})
    report=dict(layout=layout,optimizer_parameter_names_by_group=group_names,named_adam_steps=named_steps(algorithm),
        parameters_without_initialized_Adam=[n for n,p in algorithm.named_parameters() if p not in raw.state],
        actor_parameters=sum(p.numel() for p in algorithm.policy.actor.parameters()),critic_parameters=sum(p.numel() for p in algorithm.critic.parameters()),
        loaded_model_fingerprints=model_digest(algorithm),all_raw_Adam_tensors_and_options_exact=True,
        ownership_basis='Native construction and named_parameters, never state_dict tensor order',scope='Read-only engineering audit, no simulator resume')
    return algorithm,report
