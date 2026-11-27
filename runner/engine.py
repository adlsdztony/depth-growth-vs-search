"""On-policy protected grow/keep mechanics; not a validated RL selector.

All rollouts and evaluations are charged. Each path owns its buffer, critic,
normalization history and RNG streams. A rollout/update is an atomic operation:
the incumbent cannot move the prefix underneath a pending candidate rollout.
No simulator state is included in these checkpoints.
One active synchronous session per process is supported; CUDA is unvalidated.
"""
import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
import random
import threading

import numpy as np
import torch

from tianshou.algorithm.algorithm_base import policy_within_training_step


class PathRNG:
    def __init__(self,seed,device='cpu'):
        self.device=torch.device(device)
        ambient=self.capture()
        try:
            torch.manual_seed(seed);np.random.seed(seed % 2**32);random.seed(seed)
            self.state=self.capture()
        finally:self.assign(ambient)

    def capture(self):
        return dict(torch=torch.get_rng_state().clone(),numpy=copy.deepcopy(np.random.get_state()),
                    python=random.getstate(),cuda=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [])

    def assign(self,state):
        torch.set_rng_state(state['torch']);np.random.set_state(state['numpy']);random.setstate(state['python'])
        if self.device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda'])

    @contextmanager
    def activate(self):
        ambient=self.capture();self.assign(self.state)
        try:yield
        finally:
            self.state=self.capture();self.assign(ambient)


class StepTally:
    def __init__(self):self.returned=0;self.unreturned=[]
    def __call__(self,steps):self.returned+=int(steps)
    def unknown(self,upper,reason):self.unreturned.append(dict(upper_steps=int(upper),reason=str(reason)))


@contextmanager
def count_vector_steps(env,observer):
    """Count at raw synchronous vector-return boundary, before Collector hooks.

    A failed env.step may have completed an unknown subset of submitted work.
    Preserve that attempted upper bound rather than claiming exact zero cost.
    """
    assert not getattr(env,'is_async',False),'Asynchronous vector environments are unsupported'
    assert callable(observer) and callable(observer.unknown)
    original=env.step
    def step(action,*args,**kwargs):
        assert action is not None
        attempted=len(action)
        try:result=original(action,*args,**kwargs)
        except BaseException as error:
            observer.unknown(attempted,repr(error));raise
        try:
            reward=np.asarray(result[1]);returned=reward.shape[0] if reward.ndim else 1
        except BaseException as error:
            observer.unknown(attempted,'Unreadable vector return: '+repr(error));raise
        observer(returned)
        if returned!=attempted:
            if attempted>returned:observer.unknown(attempted-returned,'Incomplete synchronous vector return')
            raise RuntimeError('Synchronous vector step returned an unexpected batch size')
        return result
    env.step=step
    try:yield observer
    finally:env.step=original


@dataclass
class CostLedger:
    online_limit:int
    final_limit:int
    charges:dict=field(default_factory=dict)
    events:list=field(default_factory=list)
    unresolved:list=field(default_factory=list)

    @property
    def online(self):return sum(v for k,v in self.charges.items() if k!='final')
    @property
    def final(self):return self.charges.get('final',0)
    @property
    def total(self):return self.online+self.final
    @property
    def online_upper(self):return self.online+sum(r['upper_steps'] for r in self.unresolved if r['category']!='final')
    @property
    def final_upper(self):return self.final+sum(r['upper_steps'] for r in self.unresolved if r['category']=='final')
    @property
    def total_upper(self):return self.online_upper+self.final_upper

    def reserve(self,category,upper):
        assert upper>=0
        used=self.final_upper if category=='final' else self.online_upper
        limit=self.final_limit if category=='final' else self.online_limit
        if used+upper>limit:raise RuntimeError(f'Insufficient {category} budget: {used}+{upper}>{limit}')

    def debit(self,category,steps,**metadata):
        assert steps>=0
        # Reservation blocks planned overspending. Actual completed work must
        # still be recorded if an environment unexpectedly exceeds its bound.
        self.charges[category]=self.charges.get(category,0)+steps
        self.events.append(dict(category=category,steps=steps,total=self.total,**metadata))
        used=self.final if category=='final' else self.online
        limit=self.final_limit if category=='final' else self.online_limit
        if used>limit:
            self.events[-1]['budget_overrun']=True
            raise RuntimeError(f'Actual {category} cost exceeded its bound: {used}>{limit}')

    def note_unreturned(self,category,upper_steps,reason,**metadata):
        assert upper_steps>=0
        record=dict(category=category,upper_steps=upper_steps,lower_steps=0,reason=reason,**metadata)
        self.unresolved.append(record)
        self.events.append(dict(kind='unreturned_step_attempt',**record,total_known=self.total,total_upper=self.total_upper))


class OnPolicyPath:
    def __init__(self,name,algorithm,collector,seed,ledger,batch_size,repeat,hook_builder=None):
        self.name=name;self.algorithm=algorithm;self.collector=collector;self.ledger=ledger
        self.batch_size=batch_size;self.repeat=repeat;self.hook_builder=hook_builder
        device=next(algorithm.parameters()).device
        self.train_rng=PathRNG(seed,device);self.eval_rng=PathRNG(seed+50000001,device)
        self.updates=0;self._pending=False;self.failed=False

    def train(self,n_step,category):
        assert not self.failed and not self._pending and len(self.collector.buffer)==0
        assert n_step % self.collector.env_num==0,'Do not silently overshoot vector collections'
        self.ledger.reserve(category,n_step)
        self._pending=True
        tally=StepTally();charged=False;original_hook=self.collector._on_step_hook
        try:
            with self.train_rng.activate(),policy_within_training_step(self.algorithm.policy),count_vector_steps(self.collector.env,tally):
                collected=self.collector.collect(n_step=n_step)
                actual=int(collected.n_collected_steps);assert actual==tally.returned
                # Charge successful collection before attempting an update.
                charged=True
                self.ledger.debit(category,actual,path=self.name)
                assert actual==n_step
                result=self.algorithm.update(buffer=self.collector.buffer,batch_size=self.batch_size,repeat=self.repeat)
                self.collector.reset_buffer(keep_statistics=True)
            self.updates+=1
            return result
        except BaseException:
            self.failed=True
            raise
        finally:
            try:
                try:
                    if not charged and tally.returned:self.ledger.debit(category,tally.returned,path=self.name,partial_collection=True)
                finally:
                    for unknown in tally.unreturned:self.ledger.note_unreturned(category,path=self.name,**unknown)
            finally:
                self.collector._on_step_hook=original_hook
                self._pending=False

    def assess(self,evaluator,n_episode,max_episode_steps,category):
        assert not self.failed and not self._pending and len(self.collector.buffer)==0
        self.ledger.reserve(category,n_episode*max_episode_steps)
        tally=StepTally()
        try:
            with self.eval_rng.activate():result=evaluator(self.algorithm,n_episode,tally)
            assert result['measurement_basis']=='raw_vector_env_step_return'
            actual=int(result['steps']);assert actual==tally.returned
            assert actual<=n_episode*max_episode_steps
        finally:
            try:
                if tally.returned:self.ledger.debit(category,tally.returned,path=self.name,episodes=n_episode)
            finally:
                for unknown in tally.unreturned:self.ledger.note_unreturned(category,path=self.name,**unknown)
        return result

    def rebind(self,algorithm):
        assert not self.failed and not self._pending and len(self.collector.buffer)==0,'Never reuse a failed or stale rollout'
        self.algorithm=algorithm;self.collector.policy=algorithm.policy
        if self.hook_builder is not None:
            self.collector._on_episode_done_hook=self.hook_builder(algorithm)
        # Keep the candidate's ongoing simulator observations, episode state,
        # and path RNG streams. Only the policy/critic owner changes.


@dataclass
class DecisionRule:
    margin:float=.02
    required_streak:int=2
    wins:int=0
    nominated:bool=False

    def screen(self,keep,grow):
        assert len(keep)==len(grow) and len(keep)>0
        assert all(x in [0,1] for x in list(keep)+list(grow))
        effect=float(np.mean(np.asarray(grow,dtype=np.float64)-np.asarray(keep,dtype=np.float64)))
        self.wins=self.wins+1 if effect>=self.margin else 0
        self.nominated=self.wins>=self.required_streak
        return dict(effect=effect,streak=self.wins,nominated=self.nominated)

    def confirm(self,keep,grow):
        assert self.nominated,'Never pay confirmation before nomination'
        assert len(keep)==len(grow) and len(keep)>0
        assert all(x in [0,1] for x in list(keep)+list(grow))
        effect=float(np.mean(np.asarray(grow,dtype=np.float64)-np.asarray(keep,dtype=np.float64)))
        return dict(effect=effect,accept=effect>=self.margin,
                    scope='Development rule, not a per-decision scientific significance claim')


@dataclass(frozen=True)
class TrialPlan:
    quantum:int=100000
    warmup_rounds:int=2
    screen_rounds:tuple=(4,6,8,10)
    required_streak:int=2

    def __post_init__(self):
        assert self.quantum>0 and self.warmup_rounds>0
        assert list(self.screen_rounds)==sorted(set(self.screen_rounds))
        assert self.screen_rounds[0]>self.warmup_rounds
        assert len(self.screen_rounds)>self.required_streak>=1

    @property
    def learner_upper(self):return 2*self.quantum*self.screen_rounds[-1]

    def screening_upper(self,episodes_per_path,max_episode_steps):
        return len(self.screen_rounds)*2*episodes_per_path*max_episode_steps


def _clone_actor(actor):
    latent=actor._last_shared_latent
    try:
        actor._last_shared_latent=None
        return copy.deepcopy(actor)
    finally:actor._last_shared_latent=latent


def commit_candidate(trial,builder,probe):
    """Materialize a standalone native actor and transfer the correct Adam owners.

    Prefix states come from keep; block/head/std/critic states come from grow.
    No source model/optimizer is mutated. Caller must drain both path buffers
    and rebind the accepted candidate path without restarting its simulator.
    """
    # Imports the frozen component rather than editing it.
    from protected import clone_critic, private_rng
    source=trial.incumbent.policy.actor;candidate=trial.candidate.policy.actor
    old_keep=trial.incumbent._get_raw_optimizer();old_grow=trial.candidate._get_raw_optimizer()
    actor=_clone_actor(source)
    mapping={name:(old_keep,p) for name,p in source.named_parameters()}
    index=sum(isinstance(m,torch.nn.Linear) for m in actor.actor_body.modules())
    for j,block in enumerate(candidate.candidate_body.children()):
        name=f'adaptive_{index+j}'
        actor.actor_body.add_module(name,copy.deepcopy(block))
        for suffix,p in block.named_parameters():mapping[f'actor_body.{name}.{suffix}']=(old_grow,p)
    actor.actor_output=copy.deepcopy(candidate.actor_output)
    actor.sigma_param=torch.nn.Parameter(candidate.sigma_param.detach().clone())
    for name,p in candidate.named_parameters():
        if name=='sigma_param' or name.startswith('actor_output.'):
            mapping[name]=(old_grow,p)
    critic=clone_critic(trial.candidate.critic)
    device=next(actor.parameters()).device
    with private_rng(device,8871):algorithm=builder(actor,critic)
    raw=algorithm._get_raw_optimizer()
    for name,p in actor.named_parameters():
        old,pold=mapping[name]
        if pold in old.state:raw.state[p]=copy.deepcopy(old.state[pold])
    old_critic=dict(trial.candidate.critic.named_parameters())
    for name,p in critic.named_parameters():
        if old_critic[name] in old_grow.state:
            raw.state[p]=copy.deepcopy(old_grow.state[old_critic[name]])
    with torch.no_grad():
        before=candidate(probe)[0];after=actor(probe)[0]
        assert torch.allclose(before[0],after[0],atol=1e-6,rtol=1e-6) and torch.equal(before[1],after[1])
        assert torch.equal(trial.candidate.critic(probe),critic(probe))
    return algorithm


class AdaptiveSession:
    """Synchronous scheduler: prefix stays fixed across each rollout and update."""
    def __init__(self,keep_path,candidate_path,trial,plan=None,rule=None):
        self.keep=keep_path;self.grow=candidate_path;self.trial=trial
        assert keep_path.ledger is candidate_path.ledger
        assert keep_path.name!=candidate_path.name,'Use unique path IDs across repeated decisions'
        self._lock=threading.RLock();self.round=0;self.closed=False
        self.plan=plan or TrialPlan();self.rule=rule or DecisionRule(required_streak=self.plan.required_streak)
        assert self.plan.required_streak==self.rule.required_streak
        self.screened_rounds=[];self.events=[];self.selected=None

    def round_step(self,steps):
        with self._lock:
            assert not self.closed
            self.keep.train(steps,'learner_keep')
            self.grow.train(steps,'learner_candidate')
            self.round+=1

    def assess_pair(self,keep_evaluator,grow_evaluator,n_episode,max_episode_steps,kind='screen'):
        with self._lock:
            assert not self.closed
            # Reserve the whole paired assessment before spending either half.
            self.keep.ledger.reserve(kind+'_keep',2*n_episode*max_episode_steps)
            keep=self.keep.assess(keep_evaluator,n_episode,max_episode_steps,kind+'_keep')
            grow=self.grow.assess(grow_evaluator,n_episode,max_episode_steps,kind+'_candidate')
            return keep,grow

    def resolve(self,accept,builder,probe):
        with self._lock:
            assert not self.closed
            assert all(not p.failed and not p._pending and len(p.collector.buffer)==0 for p in [self.keep,self.grow])
            if accept:
                algorithm=commit_candidate(self.trial,builder,probe)
                self.grow.rebind(algorithm)
                selected=self.grow
            else:selected=self.keep
            self.closed=True;self.selected=selected
            return selected

    def checkpoint_open_trial(self):
        with self._lock:
            assert not self.closed
            assert all(not p.failed and not p._pending and len(p.collector.buffer)==0 for p in [self.keep,self.grow])
            return dict(trial=self.trial.checkpoint(),round=self.round,plan=copy.deepcopy(self.plan.__dict__),
                        rule=copy.deepcopy(self.rule.__dict__),screened_rounds=list(self.screened_rounds),events=copy.deepcopy(self.events),
                        paths={p.name:dict(updates=p.updates,train_rng=copy.deepcopy(p.train_rng.state),eval_rng=copy.deepcopy(p.eval_rng.state)) for p in [self.keep,self.grow]},
                        ledger=copy.deepcopy(self.keep.ledger.__dict__),
                        scope='Models, Adam, controller state and path RNG only. Buffers must be drained. No simulator state: use explicitly matched restart treatment.')

    def restore_open_trial(self,saved):
        with self._lock:
            assert not self.closed and all(not p.failed and not p._pending and len(p.collector.buffer)==0 for p in [self.keep,self.grow])
            assert self.keep.ledger.online_limit==saved['ledger']['online_limit']
            assert self.keep.ledger.final_limit==saved['ledger']['final_limit']
            self.trial.restore(saved['trial'])
            self.round=saved['round'];self.plan=TrialPlan(**saved['plan']);self.rule=DecisionRule(**saved['rule'])
            self.screened_rounds=list(saved['screened_rounds']);self.events=copy.deepcopy(saved['events'])
            self.keep.ledger.charges=copy.deepcopy(saved['ledger']['charges']);self.keep.ledger.events=copy.deepcopy(saved['ledger']['events'])
            self.keep.ledger.unresolved=copy.deepcopy(saved['ledger'].get('unresolved',[]))
            for p in [self.keep,self.grow]:
                data=saved['paths'][p.name];p.updates=data['updates']
                p.train_rng.state=copy.deepcopy(data['train_rng']);p.eval_rng.state=copy.deepcopy(data['eval_rng'])

    def release_unselected(self):
        assert self.closed and self.selected is not None
        rejected=self.keep if self.selected is self.grow else self.grow
        if rejected is not None and rejected is not self.selected:rejected.collector.close()
        self.trial=None;self.keep=self.selected;self.grow=None
        # External diagnostic references can still retain old teachers. A real
        # controller must release those too; peak memory includes commit copies.

    def run_until_decision(self,builder,probe_provider,keep_evaluator,grow_evaluator,
                           screen_episodes,confirmation_episodes,max_episode_steps):
        """Bounded development controller with conditional paid confirmation.

        Evaluators return {ids, success, steps}; ids must identify matching
        task/goal/reset units in each paired call, never pooled native stats.
        This practical rule needs development calibration and fresh-seed
        validation; no per-decision significance interpretation is made.
        """
        remaining_screens=sum(r>self.round for r in self.plan.screen_rounds)
        upper=2*self.plan.quantum*(self.plan.screen_rounds[-1]-self.round)
        upper+=remaining_screens*2*screen_episodes*max_episode_steps
        upper+=2*confirmation_episodes*max_episode_steps
        self.keep.ledger.reserve('trial_preflight',upper)
        decision=False;reason='no_nomination_inconclusive'
        while self.round<self.plan.screen_rounds[-1]:
            self.round_step(self.plan.quantum)
            if self.round not in self.plan.screen_rounds:continue
            keep,grow=self.assess_pair(keep_evaluator,grow_evaluator,screen_episodes,max_episode_steps)
            assert keep['ids']==grow['ids'] and len(keep['ids'])==screen_episodes
            assert len(set(keep['ids']))==screen_episodes
            assert len(keep['success'])==len(grow['success'])==screen_episodes
            result=self.rule.screen(keep['success'],grow['success']);self.screened_rounds.append(self.round)
            self.events.append(dict(kind='screen',round=self.round,**result,online_steps=self.keep.ledger.online))
            if not result['nominated']:continue
            keep,grow=self.assess_pair(keep_evaluator,grow_evaluator,confirmation_episodes,max_episode_steps,kind='confirmation')
            assert keep['ids']==grow['ids'] and len(keep['ids'])==confirmation_episodes
            assert len(set(keep['ids']))==confirmation_episodes
            assert len(keep['success'])==len(grow['success'])==confirmation_episodes
            confirmed=self.rule.confirm(keep['success'],grow['success']);decision=confirmed['accept']
            reason='confirmed_development_margin' if decision else 'confirmation_failed'
            self.events.append(dict(kind='confirmation',round=self.round,**confirmed,online_steps=self.keep.ledger.online))
            break
        selected=self.resolve(decision,builder,probe_provider())
        record=dict(action='grow' if decision else 'keep',reason=reason,round=self.round,
                    online_steps=selected.ledger.online,events=copy.deepcopy(self.events),
                    scope='Development controller decision; no RL efficacy claim from unit data')
        self.release_unselected()
        return selected,record
