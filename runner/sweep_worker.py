"""Separate checkpoint-restart grow/continue development diagnostic.

Every setup/load/insertion/PPO/evaluation/cleanup operation is
inside the parent's absolute allocation. This worker cannot select its parent.
"""
import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--uuid', required=True)
    parser.add_argument('--allocation-start-ns', type=int, required=True)
    parser.add_argument('--deadline-ns', type=int, required=True)
    parser.add_argument('--synthetic-fixture', action='store_true')
    args = parser.parse_args()
    # The clock's origin is supplied by the parent, not established here.
    assert 0 < args.allocation_start_ns < args.deadline_ns
    assert args.allocation_start_ns <= time.monotonic_ns()
    root = Path(args.campaign).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = json.loads(config_path.read_text())
    run = root / cfg.get('output_relative', 'runs/' + cfg['run_id'])
    run.mkdir(parents=True, exist_ok=True)
    assert not (run / 'state.json').exists(), 'Never silently resume an allocation'
    # Dependency copies live beside a frozen worker. Development fallback is
    # explicit and is recorded; the coordinator freezes all these bytes.
    here = Path(__file__).resolve().parent
    if not (here / 'engine.py').exists():
        sys.path[:0] = [str(here.parent / 'incumbent_adaptive'), str(here.parent / 'growth_ppo')]
    from budget import AllocationClock, BudgetExpired
    clock = AllocationClock(args.allocation_start_ns, args.deadline_ns,
                            cfg['final_assessment_reserve_s'], cfg['cleanup_reserve_s'])
    cpu = args.uuid == 'cpu'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ('' if cpu else args.uuid)
    if cpu:
        assert cfg['phase'] == 'validation'
    else:
        status = json.loads((root / 'status.json').read_text())
        assert args.uuid in status['verified_gpu_uuids']
    assert not args.synthetic_fixture or cpu
    # Exploratory depth-sweep extension of the sealed timing runner.
    assert cfg['method'] in ('fixed', 'grow')
    assert cfg['phase'] in ('development', 'validation')
    assert cfg['insertion_seed']==cfg['seed']+500
    grow_at = sorted(int(x) for x in cfg.get('grow_at', []))
    assert cfg['method'] == 'grow' or not grow_at
    affinity = sorted(os.sched_getaffinity(0))
    state = dict(run_id=cfg['run_id'], method=cfg['method'], variant=cfg['variant'],
                 phase=cfg['phase'], seed=cfg['seed'], uuid=args.uuid, status='running',
                 pid=os.getpid(), started_utc=utc(), allocation_start_ns=args.allocation_start_ns,
                 deadline_ns=args.deadline_ns, allocated_s=(args.deadline_ns-args.allocation_start_ns)/1e9,
                 cpu_affinity=affinity, cpu_cores=len(affinity), cpu_threads=2,
                 width=cfg['width'], start_depth=cfg['depth'], grow_at=grow_at,
                 learner_gpu_s=0., update_gpu_s=0., inference_gpu_s=0.,
                 training_steps=0, teacher_data_steps=0, monitor_steps=0, screening_steps=0, confirmation_steps=0,
                 final_assessment_steps=0, total_interactions=0, online_interactions=0,
                 round=0, depth=2, peak_depth=2, trials=0, accepted_grow=0, accepted_prune=0,
                 actor_parameter_transition_exposure=0, actor_plus_critic_parameter_transition_exposure=0,
                 worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                 assessment_protocol='disjoint_MT10_30_train_10_dev_10_test_paired_v1',
                 assessment_goals_per_task=cfg['assessment_goals_per_task'],
                 training_budget_basis='absolute elapsed process allocation including imports/setup/train/evaluation/cleanup',
                 no_learner_step_cap=False, final_endpoint_available=False,
                 reset_protocol='seeded per-environment stream; pre-reset RNG hashes recorded',
                 simulator_resume=False, initial_assessment_performed=False,
                 unreturned_environment_attempts=[], actual_adam_steps=0, PPO_optimizer_steps=0,
                 distillation_optimizer_steps=0, inherited_history_not_new_cost=True,
                 diagnostic_scope='fresh full MT10 whole-quantum growth timing; independent restart of paired branches',
                 teacher_actor_tensor_bytes=0, teacher_observation_bytes=0)
    envs = []
    paths = []
    episodes = []
    algorithm = None
    torch = None
    stop = [False]
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda a, b: stop.__setitem__(0, True))

    def append(filename, value):
        with (run / filename).open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')

    def persist():
        state['elapsed_s'] = clock.elapsed_s
        state['remaining_s'] = clock.remaining_s
        state['allocation_clock'] = clock.report()
        state['updated_utc'] = utc()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        state['cpu_peak_rss_bytes'] = int(usage.ru_maxrss)*1024
        state['cpu_s'] = usage.ru_utime + usage.ru_stime
        state['simulation_cpu_s'] = children.ru_utime + children.ru_stime
        state['total_cpu_s'] = state['cpu_s'] + state['simulation_cpu_s']
        state['cpu_coverage'] = ('self plus reaped simulator children after cleanup' if 'ended_utc' in state
                                else 'self plus reaped simulator children; live-child CPU absent until cleanup')
        state['online_interactions'] = sum(state[k] for k in ('training_steps', 'teacher_data_steps', 'monitor_steps', 'screening_steps', 'confirmation_steps'))
        state['total_interactions'] = state['online_interactions'] + state['final_assessment_steps']
        if torch is not None and not cpu and torch.cuda.is_initialized():
            state['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            state['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        atomic(run / 'state.json', state)

    def check_hard():
        if stop[0]:
            raise BudgetExpired('Parent requested stop')
        # Cooperative checks cannot preempt an unreturned blocking simulator RPC.
        clock.require('assessment')

    persist()
    try:
        clock.require('train')
        import numpy as np
        import torch as torch_module
        import gymnasium as gym
        torch = torch_module
        torch.set_num_threads(2)
        from engine import CostLedger, OnPolicyPath, PathRNG, count_vector_steps, StepTally
        from growth import tensor_digest,append_layer,optimizer_digest
        from prune_ops import named_adam,named_steps,equal
        from audit_optimizer import load_native_checkpoint
        from tianshou.data import Collector, VectorReplayBuffer
        author = root / 'reference/TOPPO_author'
        provenance = json.loads((author / 'source_provenance_complete.json').read_text())
        for record in provenance['files']:
            data = (author / record['path']).read_bytes()
            digest = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            assert digest == record['git_blob_sha']
        state['author_commit'] = provenance['commit']
        author_root = author / 'experiments/metaworld_ppo'
        sys.path.insert(0, str(author_root))
        spec = importlib.util.spec_from_file_location('compute_native_author', author_root / 'train.py')
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        from mtrl.utils.config import load_config
        from mtrl.utils.multitask_collector import MultiTaskCollector
        import mtrl.envs.metaworld_wrapper as wrapper
        native_config = Path(cfg['author_config'])
        if not native_config.is_absolute():
            native_config = root / native_config
        config = load_config(str(native_config), cli_args=[
            '--seed', str(cfg['seed']), '--device', 'cpu' if cpu else 'cuda',
            '--logging.no-use-wandb', '--logging.no-use-tensorboard'])
        config.network.actor = [int(cfg['width'])]*int(cfg['depth'])
        assert config.network.critic == [400]*3
        assert config.network.ln_critic and config.ppo.use_popart
        assert config.training.step_per_collect == 100000 and config.training.repeat_per_collect == 16
        assert config.training.batch_size == 3125 and config.env.num_envs == 10 and config.env.num_tasks == 10
        assert config.training.learning_rate == cfg['learning_rate']
        atomic(run / 'effective_author_config.json', dataclasses.asdict(config))
        native.set_seed(cfg['seed'])
        trainer_pid = os.getpid()
        original_factory = wrapper._make_single_env
        # Assessment bank = goals 20..49: panel index 0-9 train goals (20-29),
        # 10-19 development goals (30-39), 20-29 held-out test goals (40-49).
        goals = int(cfg['assessment_goals_per_task'])
        assert goals in (1, 3, 30)
        assert goals == 30 or cfg['phase'] == 'validation'
        original_goal_init = wrapper.GoalIterationWrapper.__init__
        def goal_init(self, env, tasks):
            return original_goal_init(self, env, tasks[:goals])
        wrapper.GoalIterationWrapper.__init__ = goal_init

        class GoalMetadata(gym.Wrapper):
            def seed(self, seed):
                self.env.unwrapped.seed(seed)
                node = self.env
                while node is not None:
                    if isinstance(node, wrapper.GoalIterationWrapper):
                        node._task_idx = 0
                        break
                    node = getattr(node, 'env', None)
                return [seed]
            def reset(self, **kwargs):
                self.reset_hash = hashlib.sha256(json.dumps(self.env.unwrapped.np_random.bit_generator.state, sort_keys=True).encode()).hexdigest()
                obs, info = self.env.reset(**kwargs)
                node = self.env
                self.goal_index = -1
                while node is not None:
                    if isinstance(node, wrapper.GoalIterationWrapper):
                        self.goal_index = (node._task_idx-1) % len(node._tasks)
                        break
                    node = getattr(node, 'env', None)
                return obs, self.metadata(info)
            def metadata(self, info):
                info['reference_goal_index'] = self.goal_index
                info['reset_rng_hash'] = self.reset_hash
                info['simulator_cuda_hidden'] = os.environ.get('CUDA_VISIBLE_DEVICES') == ''
                info['simulator_pid'] = os.getpid()
                return info
            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                return obs, reward, terminated, truncated, self.metadata(info)

        def factory(*args_, **kwargs):
            if os.getpid() != trainer_pid:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
            training = kwargs.get('randomize_goals', False)
            bank=kwargs['goal_tasks']
            assert len(bank)==50
            kwargs['goal_tasks']=bank[:30] if training else bank[20:50]
            args_=list(args_)
            args_[2]=kwargs['goal_tasks'][0]
            env = original_factory(*args_, **kwargs)
            task_id = args_[3]
            env.unwrapped.seed(int(cfg['seed'])*100003+int(task_id)*1009+(0 if training else 50000001))
            return GoalMetadata(env)
        wrapper._make_single_env = factory
        # Construct all subprocess environments before CUDA allocation.
        clock.require('train')
        assert not args.synthetic_fixture
        import metaworld
        from collections import defaultdict
        from unittest.mock import patch
        from types import SimpleNamespace
        import pickle
        bench = metaworld.MT10(seed=cfg['benchmark_seed'])
        banks=defaultdict(list)
        for goal in bench.train_tasks: banks[goal.env_name].append(goal)
        assert len(bench.train_classes)==10 and len(banks)==10
        splits={}
        for task,entries in sorted(banks.items()):
            vec=[np.asarray(pickle.loads(g.data)['rand_vec'],dtype='<f8').tobytes() for g in entries]
            assert len(vec)==len(set(vec))==50
            splits[task]={name:[hashlib.sha256(v).hexdigest() for v in vec[l:r]] for name,l,r in [('train',0,30),('dev',30,40),('test',40,50)]}
        atomic(run/'reset_split.json',splits)
        with patch.object(metaworld,'MT10',lambda seed=None:bench):
            result = native.make_metaworld_envs(config, seed=cfg['benchmark_seed'])
        state['environment_kind'] = 'MetaWorld_full_MT10'
        state['train_goal_count_per_task']=30
        state['dev_goal_count_per_task']=10
        state['test_goal_count_per_task']=10
        train_env, test_env, _, task_names, _, _ = result
        envs.extend([train_env, test_env])
        assert test_env is not None and len(task_names)==10
        state['task_names'] = task_names
        state['goal_counts'] = [goals]*len(task_names)
        observation_space = train_env.observation_space[0]
        action_space = train_env.action_space[0]
        class NativeBuilder:
            train=native
            def __call__(self):
                return native.create_ppo_algorithm(config,task_names,
                    int(np.prod(observation_space.shape)),int(np.prod(action_space.shape)),
                    action_space,observation_space,'cpu' if cpu else 'cuda')
        algorithm=NativeBuilder()()
        state['fresh_init']=True
        n_linear=sum(isinstance(m, torch.nn.Linear) for m in algorithm.policy.actor.actor_body.modules())
        assert n_linear==int(cfg['depth']), (n_linear, cfg['depth'])
        state['depth']=state['peak_depth']=int(cfg['depth'])
        # Learner (GPU-process) time: synchronized wall time of PPO updates and
        # of rollout/assessment policy forwards. Simulator time is excluded.
        def sync():
            if not cpu: torch.cuda.synchronize()
        original_update=algorithm.update
        in_update=[False]
        def timed_update(*a_, **k_):
            sync(); t0=time.perf_counter(); in_update[0]=True
            try: out=original_update(*a_, **k_)
            finally: in_update[0]=False
            sync(); d=time.perf_counter()-t0
            state['update_gpu_s']+=d; state['learner_gpu_s']+=d
            return out
        algorithm.update=timed_update
        original_forward=algorithm.policy.forward
        def timed_forward(*a_, **k_):
            if in_update[0]: return original_forward(*a_, **k_)
            sync(); t0=time.perf_counter()
            out=original_forward(*a_, **k_)
            sync(); d=time.perf_counter()-t0
            state['inference_gpu_s']+=d; state['learner_gpu_s']+=d
            return out
        algorithm.policy.forward=timed_forward
        state['actor_parameters']=state['peak_actor_parameters']=sum(p.numel() for p in algorithm.policy.actor.parameters())
        state['critic_parameters']=sum(p.numel() for p in algorithm.critic.parameters())
        if not cpu:
            assert torch.cuda.device_count() == 1
            state['cuda_name'] = torch.cuda.get_device_name(0)
            torch.empty(1, device='cuda')
            placement = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'], text=True)
            state['actual_gpu_placement'] = [line for line in placement.splitlines() if line.split(',')[0].strip() == str(os.getpid())]
            # Containers may hide host PIDs from nvidia-smi; record, don't require.
            assert all(args.uuid in line for line in state['actual_gpu_placement'])
        original_adam = torch.optim.Adam.step
        def adam_step(optimizer, *args_, **kwargs):
            value = original_adam(optimizer, *args_, **kwargs)
            state['actual_adam_steps'] += 1
            state['PPO_optimizer_steps']+=1
            return value
        torch.optim.Adam.step = adam_step
        ledger = CostLedger(2**60, 2**60)  # accounting sentinel, never an online stopping rule
        training_episodes=[]
        native_train_ep=native.make_episode_hook(algorithm,mode='train')
        def train_episode(batch):
            win=float(batch._cumulative_win) if hasattr(batch,'_cumulative_win') else float(np.max(batch.info['win']))
            training_episodes.append((str(batch.info[-1]['task_name']),int(win>0)))
            return native_train_ep(batch)
        train_collector = MultiTaskCollector(policy=algorithm, env=train_env,
            buffer=VectorReplayBuffer(total_size=100000, buffer_num=10), exploration_noise=True,
            on_episode_done_hook=train_episode)
        train_collector.reset()
        path = OnPolicyPath('deployment', algorithm, train_collector, cfg['seed'], ledger, (250 if cpu else 3125), 16,
                            hook_builder=lambda alg:native.make_episode_hook(alg, mode='train'))
        paths.append(path)
        episode_hook = native.make_episode_hook(algorithm, mode='test')
        def collect_episode(batch):
            value = episode_hook(batch)
            info = batch.info[-1]
            episodes.append(dict(task_name=str(info['task_name']), goal_index=int(info['reference_goal_index']),
                success=int(np.max(batch.info['win'])), length=len(batch), reset_rng_hash=str(info['reset_rng_hash']),
                simulator_cuda_hidden=bool(info['simulator_cuda_hidden']), simulator_pid=int(info['simulator_pid'])))
            return value
        test_collector = Collector(policy=algorithm, env=test_env, on_episode_done_hook=collect_episode)
        test_collector.reset()
        # Reset/init are paid setup. Fractions refer only to this available
        # training interval, excluding the two prespecified reserves.
        train_start_ns = time.monotonic_ns()
        train_deadline_ns = args.deadline_ns-int((cfg['final_assessment_reserve_s']+cfg['cleanup_reserve_s'])*1e9)
        state['training_interval_start_ns'] = train_start_ns
        state['training_interval_deadline_ns'] = train_deadline_ns
        durations = []
        probe_bank=[]

        class Observer:
            def __init__(self, category): self.category=category
            def __call__(self, count):
                state[self.category] += int(count)
                actor = state['actor_parameters']
                state['actor_parameter_transition_exposure'] += int(count)*actor
                state['actor_plus_critic_parameter_transition_exposure'] += int(count)*(actor+state['critic_parameters'])
                if state[self.category] % 10000 == 0: persist()
                check_hard()
            def unknown(self, upper, reason):
                state['unreturned_environment_attempts'].append(dict(category=self.category, upper_steps=int(upper), reason=reason))

        def measure(stage):
            final_seeds=[int(cfg['seed'])*100003+i*1009+900000001 for i in range(len(task_names))]
            test_env.seed(final_seeds)
            test_collector.reset()
            episodes.clear()
            algorithm.eval()
            with PathRNG(cfg['seed']+50000001, next(algorithm.parameters()).device).activate(), torch.no_grad(), count_vector_steps(test_env, Observer('final_assessment_steps')):
                stats = test_collector.collect(n_episode=goals*len(task_names))
            assert int(stats.n_collected_steps)==goals*len(task_names)*500
            columns=[sorted([row for row in episodes if row['task_name']==name],key=lambda row:row['goal_index']) for name in task_names]
            assert all([row['goal_index'] for row in column]==list(range(goals)) for column in columns)
            matrix=np.asarray([[column[index]['success'] for column in columns] for index in range(goals)])
            k=max(1,goals//3)
            split=dict(train=matrix[:k],dev=matrix[k:2*k],test=matrix[2*k:3*k])
            report=dict(stream=stage,success=float(matrix.mean()),per_task_success=matrix.mean(0).tolist(),
                **{f'success_{n}':float(m.mean()) for n,m in split.items() if m.size},
                **{f'per_task_{n}':m.mean(0).tolist() for n,m in split.items() if m.size},
                episode_success=matrix.tolist(),training_steps=state['training_steps'],depth=state['depth'],
                elapsed_s=clock.elapsed_s,learner_gpu_s=state['learner_gpu_s'],update_gpu_s=state['update_gpu_s'],
                inference_gpu_s=state['inference_gpu_s'],actual_adam_steps=state['actual_adam_steps'])
            append('evaluations.jsonl',report)
            return report

        def grow(stage):
            assert len(train_collector.buffer)==0 and not path._pending
            before_adam=named_adam(algorithm)
            probe=torch.as_tensor(np.asarray(train_collector._pre_collect_obs_RO,dtype=np.float32),device='cpu' if cpu else 'cuda')
            kind=cfg.get('insert','identity')
            event=append_layer(algorithm,kind,int(cfg['insertion_seed'])+state['depth'],probe)
            assert event['hidden_depth']==state['depth']+1 and event['critic_max_error']==0
            if kind=='identity':
                assert event['std_max_error']==0 and event['mean_max_error']<=1e-5
            assert equal(before_adam,named_adam(algorithm))
            state.update(depth=event['hidden_depth'],peak_depth=event['hidden_depth'],actor_parameters=event['actor_parameters'],
                peak_actor_parameters=event['actor_parameters'],accepted_grow=state['accepted_grow']+1)
            append('growth_events.jsonl',dict(utc=utc(),stage=stage,training_steps=state['training_steps'],**event));persist()
        def checkpoint(stage):
            raw=algorithm._get_raw_optimizer()
            dest=run/('state_'+stage+'.pt'); tmp=dest.with_suffix('.tmp')
            with tmp.open('xb') as file:
                torch.save(dict(algorithm=algorithm.state_dict(),optimizer=raw.state_dict(),config=cfg,state=state),file)
                file.flush();os.fsync(file.fileno())
            tmp.replace(dest)
        eval_every=int(cfg['eval_every'])
        measure('init')
        quantum=int(cfg.get('rollout_per_quantum',100000))
        assert quantum==100000 or cfg['phase']=='validation'
        while not stop[0] and state['training_steps']<cfg['training_steps_cap']:
            estimate = max(durations, default=float(cfg.get('initial_quantum_upper_s', 90)))*1.25
            if not clock.can_start('train', estimate):
                state['training_stop_reason'] = 'next_atomic_rollout_update_does_not_fit_reserved_window'
                break
            before = time.monotonic_ns()
            algorithm.train()
            training_episodes.clear()
            with count_vector_steps(train_env, Observer('training_steps')):
                learner_stats=path.train(quantum, 'learner')
            duration = (time.monotonic_ns()-before)/1e9
            durations.append(duration)
            state['round'] += 1
            state['quantum_durations_s'] = durations
            episode_summary={name:{'episodes':sum(t==name for t,_ in training_episodes),'successes':sum(v for t,v in training_episodes if t==name)} for name in task_names}
            def scalars(v):
                if dataclasses.is_dataclass(v):return scalars(dataclasses.asdict(v))
                if isinstance(v,dict):return {str(k):scalars(x) for k,x in v.items() if scalars(x) is not None}
                if isinstance(v,(int,float,np.number)) and math.isfinite(float(v)):return float(v)
                return None
            append('metrics.jsonl', dict(utc=utc(), round=state['round'], training_steps=state['training_steps'],
                elapsed_s=clock.elapsed_s, quantum_s=duration, actual_adam_steps=state['actual_adam_steps'], depth=state['depth'],
                actor_parameters=state['actor_parameters'], learner_gpu_s=state['learner_gpu_s'],
                update_gpu_s=state['update_gpu_s'], inference_gpu_s=state['inference_gpu_s'],
                train_completed_episode_success=episode_summary, learner_stats=scalars(learner_stats)))
            persist()
            if state['training_steps']%eval_every==0 and state['training_steps']<cfg['training_steps_cap']:
                measure(f"step_{state['training_steps']}")
            for g in grow_at:
                if state['training_steps']==g: grow(f'grow_{g}')
        assert state['training_steps']==cfg['training_steps_cap'], 'Prespecified step quota incomplete'
        state['training_finished_ns']=time.monotonic_ns()
        result=measure('final')
        state['provisional_final_success']=result['success_test']
        state['provisional_final_per_task_success']=result['per_task_test']
        state['final_success_train']=result['success_train']
        state['final_success_dev']=result['success_dev']
        checkpoint('endpoint')
        state['status'] = 'assessment_completed_pending_cleanup'
    except BudgetExpired as error:
        state.update(status='budget_interrupted', failure=str(error))
    except BaseException as error:
        state.update(status='failed', failure=repr(error), traceback=traceback.format_exc())
    finally:
        # Endpoint admission includes closing/reaping simulators and checkpoint
        # serialization. A completed assessment alone is not enough.
        for env in envs:
            try:
                if env is not None and not getattr(env, 'is_closed', False): env.close()
            except BaseException as error:
                state.setdefault('cleanup_errors', []).append(repr(error))
        if algorithm is not None and torch is not None:
            try:
                tmp=run/'last_checkpoint.tmp'
                with tmp.open('xb') as f:
                    torch.save(dict(algorithm=algorithm.state_dict(), optimizer=algorithm._get_raw_optimizer().state_dict(),config=cfg,state=state),f);f.flush();os.fsync(f.fileno())
                tmp.replace(run/'last_checkpoint.pt')
            except BaseException as error:
                state.setdefault('cleanup_errors', []).append(repr(error))
        state['cleanup_completed_ns'] = time.monotonic_ns()
        state['ended_utc'] = utc()
        state['deadline_overrun_s'] = max(0., (state['cleanup_completed_ns']-args.deadline_ns)/1e9)
        if state['status']=='assessment_completed_pending_cleanup' and not state.get('cleanup_errors') and state['deadline_overrun_s']==0:
            state['status'] = 'completed'
            state['final_endpoint_available'] = True
            state['final_success'] = state.get('provisional_final_success')
            state['final_per_task_success'] = state.get('provisional_final_per_task_success')
        else:
            state['final_endpoint_available'] = False
            if state['status']=='assessment_completed_pending_cleanup': state['status']='budget_interrupted'
        if args.synthetic_fixture:
            state['synthetic_environment_steps'] = sum(state[key] for key in ('training_steps','teacher_data_steps','monitor_steps','screening_steps','confirmation_steps','final_assessment_steps'))
        else:
            state['metaworld_interactions'] = sum(state[key] for key in ('training_steps','teacher_data_steps','monitor_steps','screening_steps','confirmation_steps','final_assessment_steps'))
        state['cpu_coverage'] = 'self plus reaped simulator children after cleanup'
        persist()
    if state['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
