"""Pinned author PPO constructor adapter; no source edits and no environments."""
import importlib.util
from pathlib import Path
import sys
import threading
from unittest.mock import patch

import gymnasium as gym
import numpy as np


class NativeFactory:
    def __init__(self,author_root,width=400,critic_width=400,rate=.0003,batch_size=3125,device='cpu',task_names=None):
        root=Path(author_root).resolve();sys.path.insert(0,str(root))
        from mtrl.utils.config import load_config
        spec=importlib.util.spec_from_file_location('incumbent_adaptive_author_train',root/'train.py')
        self.train=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.train)
        self.config=load_config(str(root/'configs/ppo_mt10_vanilla_rfv2.yaml'),
                               cli_args=['--device',device,'--network.ln-critic','--ppo.use-popart'])
        self.config.network.actor=[width];self.config.network.critic=[critic_width]*3
        self.config.training.learning_rate=rate;self.config.training.batch_size=batch_size
        self.config.ppo.popart_method='welford'
        self.device=device;self.task_names=task_names or [f'task{i}' for i in range(10)]
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(49,),np.float32)
        self.action_space=gym.spaces.Box(-1.,1.,(4,),np.float32)
        self._lock=threading.RLock()

    def __call__(self,actor=None,critic=None):
        args=(self.config,self.task_names,49,4,self.action_space,self.observation_space,self.device)
        with self._lock:
            if actor is None:return self.train.create_ppo_algorithm(*args)
            assert critic is not None
            with patch.object(self.train,'MultiTaskActor',lambda **kw:actor),patch.object(self.train,'MultiTaskCritic',lambda **kw:critic):
                return self.train.create_ppo_algorithm(*args)
