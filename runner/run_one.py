"""Launch one depth-sweep / scheduled-growth run in its own campaign copy.

Usage (from anywhere):
  python run_one.py --out RUN_ROOT --method fixed --depth 2 --seed 1 --uuid GPU-...
  python run_one.py --out RUN_ROOT --method grow --depth 1 --grow-at 1000000 2000000 --seed 1 --uuid GPU-...
  python run_one.py --out RUN_ROOT --method grow --depth 1 --grow-at 2000 --seed 1 --uuid cpu --smoke
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--method', choices=['fixed', 'grow'], required=True)
    p.add_argument('--depth', type=int, required=True)
    p.add_argument('--width', type=int, default=400)
    p.add_argument('--grow-at', type=int, nargs='*', default=[])
    p.add_argument('--insert', choices=['identity', 'random'], default='identity')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--uuid', required=True)
    p.add_argument('--steps', type=int, default=3_000_000)
    p.add_argument('--eval-every', type=int, default=500_000)
    p.add_argument('--wall-hours', type=float, default=6.0)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    out = Path(a.out).resolve()
    assert not out.exists(), f'{out} exists; never reuse a run directory'
    shutil.copytree(HERE, out, symlinks=True, ignore=shutil.ignore_patterns('runs', '__pycache__', '*.pt'))
    cpu = a.uuid == 'cpu'
    if a.smoke:
        assert cpu
        steps, quantum, goals, every = 3000, 1000, 3, 1000
    else:
        steps, quantum, goals, every = a.steps, 100_000, 30, a.eval_every
    run_id = f"{a.method}_w{a.width}_d{a.depth}" + (('_g' + '-'.join(str(g // 1000) + 'k' for g in a.grow_at)) if a.grow_at else '') + ('_rand' if a.insert == 'random' else '') + f"_s{a.seed}"
    cfg = dict(phase='validation' if cpu else 'development', method=a.method, run_id=run_id,
               variant='depth_sweep_heldout_goals_v1', seed=a.seed, benchmark_seed=17719,
               insertion_seed=a.seed + 500, training_steps_cap=steps, rollout_per_quantum=quantum,
               assessment_goals_per_task=goals, final_assessment_reserve_s=60 if cpu else 600,
               cleanup_reserve_s=30, initial_quantum_upper_s=30 if cpu else 150,
               author_config='configs/learner.yaml', learning_rate=0.0003,
               width=a.width, depth=a.depth, grow_at=a.grow_at, eval_every=every, insert=a.insert)
    (out / 'configs').mkdir(exist_ok=True)
    cfg_path = out / 'configs' / 'effective.json'
    cfg_path.write_text(json.dumps(cfg, indent=2))
    (out / 'status.json').write_text(json.dumps({'verified_gpu_uuids': [] if cpu else [a.uuid]}))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if cpu else a.uuid, WANDB_MODE='disabled',
               PYTHONNOUSERSITE='1', MPLBACKEND='Agg', MUJOCO_GL='egl')
    now = time.monotonic_ns()
    deadline = now + int(a.wall_hours * 3600 * 1e9)
    cmd = [sys.executable, '-B', str(out / 'sweep_worker.py'), '--campaign', str(out), '--config', str(cfg_path),
           '--uuid', a.uuid, '--allocation-start-ns', str(now), '--deadline-ns', str(deadline)]
    with (out / 'worker.log').open('w') as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=out)
    state = json.loads((out / 'runs' / run_id / 'state.json').read_text())
    print(run_id, 'exit', code, 'status', state['status'], 'final_test', state.get('final_success'),
          'train', state.get('final_success_train'), 'elapsed', round(state.get('elapsed_s', 0)))
    sys.exit(code)


if __name__ == '__main__':
    main()
