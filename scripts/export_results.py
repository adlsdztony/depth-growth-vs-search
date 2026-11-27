"""Copy the lightweight logs of completed runs into results/runs/<run_id>/.

usage: python scripts/export_results.py RAW_DIR [RAW_DIR ...]

Each RAW_DIR holds campaign directories created by runner/run_one.py
(<campaign>/configs/effective.json and <campaign>/runs/<run_id>/...).
Machine-specific fields (GPU UUID, PID, CPU affinity) are dropped.
If a configuration was trained twice, the first campaign in sorted order is kept.
"""
import json, shutil, sys
from pathlib import Path

DROP = {'uuid', 'pid', 'cpu_affinity'}
KEEP = ['evaluations.jsonl', 'metrics.jsonl', 'growth_events.jsonl']
OUT = Path(__file__).resolve().parent.parent / 'results' / 'runs'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for raw in sys.argv[1:]:
        for camp in sorted(Path(raw).iterdir()):
            for run in sorted((camp / 'runs').glob('*')) if (camp / 'runs').is_dir() else []:
                sp = run / 'state.json'
                if not sp.exists():
                    continue
                s = json.loads(sp.read_text())
                if s.get('phase') != 'development' or s.get('status') != 'completed':
                    continue
                dest = OUT / s['run_id']
                if dest.exists():
                    print('skip duplicate', camp.name)
                    continue
                dest.mkdir()
                (dest / 'state.json').write_text(json.dumps({k: v for k, v in s.items() if k not in DROP}, indent=1))
                for f in KEEP:
                    if (run / f).exists():
                        shutil.copy(run / f, dest / f)
                cfg = camp / 'configs' / 'effective.json'
                if cfg.exists():
                    shutil.copy(cfg, dest / 'config.json')
                n += 1
    print(n, 'runs exported to', OUT)


if __name__ == '__main__':
    main()
