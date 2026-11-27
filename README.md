# Grow Instead of Search?

Code, logs and figures for the report **"Grow Instead of Search? Function-Preserving Depth
Growth for Multi-Task PPO on MT10"** (`report/main.pdf`).

Zicheng Gong, Zhewei Kang, Zilong Zhou (equal contribution, alphabetical by surname).

**Question.** When the right actor depth is unknown, can one run that starts at depth 1 and
inserts layers during training match a searched fixed depth, once the cost of the search is
counted?

**Method.** A new layer is a bias-free `Linear(W, W)` initialised to the identity, followed by a
ReLU, appended after the last hidden ReLU. Since the hidden activations are non-negative, the
policy is exactly unchanged at insertion; existing weights keep their Adam state.

## Main result (width 400, 5 paired seeds, held-out goals)

| Strategy | Success (%) | Env. steps (M) | Learner PFLOP |
| --- | ---: | ---: | ---: |
| Grid search over depths 1–4 | 89.0 ± 1.3 | 12.0 | 0.71 |
| Successive halving (simulated from logged curves) | 85.2 ± 2.9 | 5.5 | 0.36 |
| Default depth 2 | 80.8 ± 1.5 | 3.0 | 0.15 |
| Random depth | 79.2 ± 1.1 | 3.0 | 0.18 |
| **Grow 1→4 at 0.5M, 1M, 1.5M** | **87.6 ± 1.2** | 3.0 | 0.20 |
| Grow 1→3 at 0.5M, 1M | 84.0 ± 1.7 | 3.0 | 0.18 |
| Grow 1→3 at 0.2M, 0.4M | 80.0 ± 2.7 | 3.0 | 0.19 |
| Grow 1→3 at 1M, 2M | 79.0 ± 1.5 | 3.0 | 0.15 |
| Grow 1→3 at 0.5M, 1M, random-init layers (ablation) | 75.4 ± 2.0 | 3.0 | 0.18 |
| Best fixed depth in hindsight (not achievable) | 89.0 ± 1.3 | 3.0 | 0.23 |

## Width 128 (5 paired seeds, held-out goals)

Here depth 3, not depth 4, is the best fixed depth on average.

| Strategy | Success (%) | Env. steps (M) | Learner PFLOP |
| --- | ---: | ---: | ---: |
| Grid search over depths 1–4 | 81.0 ± 1.1 | 12.0 | 0.44 |
| Successive halving (simulated) | 77.8 ± 2.1 | 5.5 | 0.21 |
| Default depth 2 | 72.2 ± 2.6 | 3.0 | 0.11 |
| Random depth | 71.2 ± 0.8 | 3.0 | 0.11 |
| Grow 1→4 at 0.5M, 1M, 1.5M | 76.2 ± 2.7 | 3.0 | 0.11 |
| Grow 1→3 at 0.5M, 1M | 73.6 ± 1.7 | 3.0 | 0.11 |
| Grow 1→3 at 1M, 2M | 69.6 ± 2.9 | 3.0 | 0.11 |
| Best fixed depth in hindsight (not achievable) | 81.0 ± 1.1 | 3.0 | 0.11 |

At width 128, growing to depth 4 is 4.8 ± 8.0 points below grid search (paired 95% interval), so
the width-400 result does not carry over unchanged.

Mean ± standard error. All numbers are produced by `analysis/make_figures.py` and stored in
`results/summary.json`.

## Repository layout

| Path | Content |
| --- | --- |
| `runner/` | Training code. `run_one.py` launches one run; `sweep_worker.py` is the training loop (multi-task PPO, evaluation on train/validation/test goals every 0.5M steps, scheduled layer insertion); `growth.py` implements identity/random insertion. |
| `scripts/run_study.sh` | One line per run reported in the paper. |
| `scripts/export_results.py` | Copies the lightweight logs of finished runs into `results/runs/`. |
| `results/runs/<run_id>/` | Logs of all 80 runs: `config.json`, `state.json` (final state, final success on train/validation/test goals), `evaluations.jsonl` (every 0.5M steps), `metrics.jsonl` (every 100k-step rollout), `growth_events.jsonl` (insertions). No checkpoints. |
| `results/summary.json` | Every number in the report: strategy results, paired differences with 95% intervals, per-task success, FLOPs. |
| `analysis/make_figures.py` | All data figures and `results/summary.json` from `results/runs/`. |
| `analysis/schematics.py` | Schematic figures and the MT10 task panel. |
| `analysis/render_mt10.py` | Renders one frame per MT10 task (needs Meta-World). |
| `figures/` | Rendered figures used in the report. |
| `report/` | LaTeX source (NeurIPS 2026 style) and `main.pdf`. |

Run ids encode the configuration: `fixed_w400_d3_s2` is a fixed depth-3, width-400 actor with
seed 2; `grow_w400_d1_g500k-1000k_s2` starts at depth 1 and inserts a layer at 0.5M and 1M
steps; the suffix `_rand` marks random initialisation of inserted layers.

## Reproduce the figures and the report

Needs Python with `numpy`, `matplotlib` and `Pillow`:

```bash
python analysis/make_figures.py          # figures/w400, figures/w128, results/summary.json
python analysis/schematics.py            # figures/static
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Reproduce the training runs

Linux x86-64, Python 3.12, an NVIDIA GPU and about 12 CPU cores per run. A 3M-step run takes
35–65 minutes.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r runner/resolved.lock

# quick CPU check (3,000 steps, one insertion)
.venv/bin/python runner/run_one.py --out runs/smoke --method grow --depth 1 --grow-at 2000 \
    --seed 1 --uuid cpu --smoke

# one configuration
GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
.venv/bin/python runner/run_one.py --out runs/grow_early4_s1 --method grow --depth 1 \
    --grow-at 500000 1000000 1500000 --seed 1 --uuid $GPU

# every run in the paper (sequential; use the FILTER argument to split across GPUs)
PYTHON=.venv/bin/python bash scripts/run_study.sh runs $GPU

# collect finished runs and redraw
python scripts/export_results.py runs
python analysis/make_figures.py
```

Each run writes a fresh campaign directory and refuses to reuse an existing one. The benchmark
seed (17719) fixes the train/validation/test goal split; `--seed` sets the training seed.

## Notes

- Successive halving is simulated from the logged validation curves of the fixed-depth runs
  (rungs at 0.5M and 1.5M steps, keep half); grid search selects by final validation success.
- Learner FLOPs per sample are 2P (rollout) + 16 × 6P (PPO epochs), with P the actor plus critic
  parameters at that time.
- One configuration (width 400, depth 4, seed 2) was trained twice; the first run is kept.
