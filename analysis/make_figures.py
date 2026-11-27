"""Make every data figure and number used in the report from results/runs.

usage: python analysis/make_figures.py [RESULTS_DIR] [FIG_DIR]
defaults: results/runs  figures
Writes figures/w400/*.png, figures/w128/*.png and results/summary.json.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results' / 'runs'
FIGS = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'figures'

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'legend.frameon': False,
    'lines.linewidth': 2.2, 'axes.titleweight': 'bold', 'axes.titlesize': 14,
})

# configuration key -> (label, short label, colour)
CONF = {
    'd1': ('Fixed depth 1', 'Depth 1', '#86b6ef'),
    'd2': ('Fixed depth 2', 'Depth 2', '#5598e7'),
    'd3': ('Fixed depth 3', 'Depth 3', '#256abf'),
    'd4': ('Fixed depth 4', 'Depth 4', '#0d366b'),
    'vearly': ('Grow 1→3 at 0.2M, 0.4M', 'Grow\nvery early', '#c2459e'),
    'early': ('Grow 1→3 at 0.5M, 1M', 'Grow\nearly', '#6250d6'),
    'late': ('Grow 1→3 at 1M, 2M', 'Grow\nlate', '#eb6834'),
    'early4': ('Grow 1→4 at 0.5M, 1M, 1.5M', 'Grow early\nto depth 4', '#b07d00'),
    'early_rand': ('Grow 1→3 at 0.5M, 1M, random init', 'Grow early\nrandom init', '#d62f2f'),
}
FIXED = ['d1', 'd2', 'd3', 'd4']
GROW = ['vearly', 'early', 'late', 'early4', 'early_rand']
SCHED = {(200_000, 400_000): 'vearly', (500_000, 1_000_000): 'early', (1_000_000, 2_000_000): 'late',
         (500_000, 1_000_000, 1_500_000): 'early4'}
T = 3_000_000
R1, R2 = T // 6, T // 2          # successive-halving rungs
EVAL_STEPS = [i * 500_000 for i in range(7)]


def jl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def load():
    runs = []
    for d in sorted(RUNS.iterdir()):
        s = json.loads((d / 'state.json').read_text())
        cfg = json.loads((d / 'config.json').read_text()) if (d / 'config.json').exists() else {}
        if s['method'] == 'fixed':
            key = f"d{s['start_depth']}"
        else:
            key = SCHED[tuple(s['grow_at'])]
            if cfg.get('insert', 'identity') == 'random':
                key = 'early_rand'
        runs.append(dict(id=d.name, key=key, width=s['width'], seed=s['seed'], state=s,
                         ev=jl(d / 'evaluations.jsonl'), me=jl(d / 'metrics.jsonl'), gr=jl(d / 'growth_events.jsonl')))
    return runs


def lab(k):
    return CONF[k][0]


def col(k):
    return CONF[k][2]


def at(run, step, metric='success_test'):
    v = [e[metric] for e in run['ev'] if e['training_steps'] == step]
    return v[0] if v else None


def cum_flops(run, step=T):
    """Learner FLOPs up to `step`: per sample 2P (rollout forward) + 16 epochs x 6P (PPO update), P = actor + critic."""
    crit = run['state']['critic_parameters']
    return sum(100_000 * (2 + 16 * 6) * (m['actor_parameters'] + crit) for m in run['me'] if m['training_steps'] <= step)


def final(run, split='test'):
    return run['state']['final_success' if split == 'test' else f'final_success_{split}']


def tci(x):
    """mean and 95% t-interval half-width"""
    t975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float('nan')
    return float(x.mean()), float(t975[len(x) - 1] * x.std(ddof=1) / np.sqrt(len(x)))


def save(fig, out, name):
    fig.savefig(out / name, dpi=200, bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)


# ---------------------------------------------------------------- strategies
def strategies(runs):
    """Per seed: test success, env steps and FLOPs of each way of producing one policy."""
    seeds = sorted({r['seed'] for r in runs})
    res = defaultdict(list)
    for s in seeds:
        by = {r['key']: r for r in runs if r['seed'] == s}
        if not all(k in by for k in FIXED):
            continue
        fl = {k: cum_flops(by[k]) for k in by}
        dev = lambda k, t: at(by[k], t, 'success_dev') or 0
        best = max(FIXED, key=lambda k: (final(by[k], 'dev'), -int(k[1:])))
        res['grid'].append(dict(seed=s, test=final(by[best]), steps=4 * T, flops=sum(fl[k] for k in FIXED), pick=best))
        alive = sorted(FIXED, key=lambda k: -dev(k, R1))[:2]
        win = max(alive, key=lambda k: dev(k, R2))
        f = (sum(cum_flops(by[k], R1) for k in FIXED) + sum(cum_flops(by[k], R2) - cum_flops(by[k], R1) for k in alive)
             + fl[win] - cum_flops(by[win], R2))
        res['sh'].append(dict(seed=s, test=final(by[win]), steps=4 * R1 + 2 * (R2 - R1) + (T - R2), flops=f, pick=win))
        orc = max(FIXED, key=lambda k: final(by[k]))
        res['oracle'].append(dict(seed=s, test=final(by[orc]), steps=T, flops=fl[orc], pick=orc))
        res['random'].append(dict(seed=s, test=np.mean([final(by[k]) for k in FIXED]), steps=T,
                                  flops=np.mean([fl[k] for k in FIXED]), pick='mean'))
        res['default'].append(dict(seed=s, test=final(by['d2']), steps=T, flops=fl['d2'], pick='d2'))
        for k in FIXED + GROW:
            if k in by:
                res[k].append(dict(seed=s, test=final(by[k]), steps=T, flops=fl[k], pick=k))
    return res


SNAME = {'oracle': 'Best fixed depth (hindsight)', 'grid': 'Grid search, 4 depths', 'sh': 'Successive halving',
         'default': 'Default depth 2', 'random': 'Random depth', **{k: CONF[k][0] for k in GROW}}
SCOL = {'oracle': '#0d366b', 'grid': '#52514e', 'sh': '#008300', 'default': '#5598e7', 'random': '#86b6ef',
        **{k: CONF[k][2] for k in GROW}}


def strategy_panel(res, out, order):
    rows = [k for k in order if res.get(k)]
    n = len(res[rows[0]])
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 0.6 * len(rows) + 1.3), sharey=True,
                               gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.08))
    cmax = max(np.mean([r['flops'] for r in res[k]]) / 1e15 for k in rows)
    for i, k in enumerate(rows):
        y = len(rows) - 1 - i
        t = np.array([r['test'] for r in res[k]]) * 100
        se = t.std(ddof=1) / np.sqrt(len(t)) if len(t) > 1 else 0
        a.errorbar(t.mean(), y, xerr=se, fmt='o', color=SCOL[k], ms=10, lw=2, capsize=0, mec=SURF, mew=1.5)
        a.text(t.mean() + se + 0.6, y, f'{t.mean():.0f}', color=INK, fontsize=12, va='center')
        c = np.mean([r['flops'] for r in res[k]]) / 1e15
        b.barh(y, c, 0.5, color=SCOL[k], edgecolor='none')
        b.text(c + 0.012 * cmax, y, f'{c:.2f}', va='center', fontsize=12, color=INK)
    a.set_yticks(range(len(rows)), [SNAME[k] for k in rows][::-1])
    lo = min(np.mean([r['test'] for r in res[k]]) for k in rows) * 100
    a.set_xlim(5 * np.floor((lo - 6) / 5), 96)
    a.set_xlabel(f'Held-out success (%), mean ± s.e. over {n} seeds')
    b.set_xlabel('Total learner compute incl. search (PFLOP)'); b.set_xlim(0, cmax * 1.18)
    for ax in (a, b):
        ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0)
    b.spines['left'].set_visible(False)
    a.axhline(len(rows) - 1.5, color=GRID, lw=1)
    save(fig, out, 'strategy_panel.png')


def anytime(runs, out, grow_keys):
    """Held-out success of the policy each strategy would return vs. learner FLOPs spent so far."""
    seeds = sorted({r['seed'] for r in runs})
    cur = defaultdict(list)
    idx = {t: i for i, t in enumerate(EVAL_STEPS)}
    for s in seeds:
        by = {r['key']: r for r in runs if r['seed'] == s}
        if not all(k in by for k in FIXED):
            continue
        dev = lambda k, t: at(by[k], t, 'success_dev') or 0
        tst = lambda k, t: at(by[k], t) or 0
        F = {k: [cum_flops(by[k], t) for t in EVAL_STEPS] for k in by}
        cur['grid'].append([(sum(F[k][i] for k in FIXED), tst(max(FIXED, key=lambda k: dev(k, t)), t)) for i, t in enumerate(EVAL_STEPS)])
        alive = sorted(FIXED, key=lambda k: -dev(k, R1))[:2]
        win = max(alive, key=lambda k: dev(k, R2))
        sh = []
        for i, t in enumerate(EVAL_STEPS):
            if t <= R1:
                c, pool = sum(F[k][i] for k in FIXED), FIXED
            elif t <= R2:
                c, pool = sum(F[k][idx[R1]] for k in FIXED) + sum(F[k][i] - F[k][idx[R1]] for k in alive), alive
            else:
                c = (sum(F[k][idx[R1]] for k in FIXED) + sum(F[k][idx[R2]] - F[k][idx[R1]] for k in alive)
                     + F[win][i] - F[win][idx[R2]]); pool = [win]
            sh.append((c, tst(max(pool, key=lambda k: dev(k, t)), t)))
        cur['sh'].append(sh)
        for k in grow_keys:
            if k in by:
                cur[k].append([(F[k][i], tst(k, t)) for i, t in enumerate(EVAL_STEPS)])
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for k, cs in cur.items():
        arr = np.array(cs)
        x, y = arr[:, :, 0].mean(0) / 1e15, arr[:, :, 1].mean(0) * 100
        ax.plot(x, y, color=SCOL[k], lw=2.8 if k == 'early' else 2.0, marker='o' if k in GROW else ('s' if k == 'grid' else 'D'),
                ms=6, mec=SURF, mew=1.0, label=SNAME[k], zorder=3 if k in GROW else 2)
    ax.set_xlabel('Total learner compute spent, including search (PFLOP)')
    ax.set_ylabel('Held-out success of returned policy (%)')
    ax.set_ylim(0, 100); ax.set_xlim(0, None)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 0.06), fontsize=11)
    ax.text(0.99, 0.01, f'mean over {len(cur["sh"])} seeds; points every 0.5M steps', transform=ax.transAxes,
            ha='right', fontsize=10, color=INK2)
    save(fig, out, 'anytime_flops.png')


# ---------------------------------------------------------------- per-configuration figures
def curves(runs, out, metric='success_test', keys=FIXED, name=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in keys:
        rs = [r for r in runs if r['key'] == k]
        if not rs:
            continue
        v = np.array([[at(r, t, metric) for t in EVAL_STEPS] for r in rs], float) * 100
        x = np.array(EVAL_STEPS) / 1e6
        ax.plot(x, v.mean(0), color=col(k), label=f'{lab(k)} (n={len(rs)})', lw=2.6 if k in GROW else 2.0)
        ax.fill_between(x, v.min(0), v.max(0), color=col(k), alpha=0.08, lw=0)
    ax.set_xlabel('Environment steps (M)')
    ax.set_ylabel({'success_test': 'Held-out success (%)', 'success_train': 'Training-goal success (%)',
                   'success_dev': 'Validation success (%)'}[metric])
    ax.set_ylim(0, 100); ax.legend(loc='lower right', fontsize=11)
    save(fig, out, name or f'curves_{metric}.png')


def train_vs_test(runs, out, keys):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    rows = []
    for i, k in enumerate(keys):
        rs = [r for r in runs if r['key'] == k]
        tr = [final(r, 'train') * 100 for r in rs]; te = [final(r) * 100 for r in rs]
        rows.append(dict(key=k, train=float(np.mean(tr)), test=float(np.mean(te)), n=len(rs)))
        ax.bar(i - 0.19, np.mean(tr), 0.36, color=col(k), alpha=0.35, edgecolor='none')
        ax.bar(i + 0.19, np.mean(te), 0.36, color=col(k), edgecolor='none')
        ax.scatter([i + 0.19] * len(te), te, s=22, color=INK, zorder=3)
        ax.text(i + 0.19, max(te) + 2.5, f'{np.mean(te):.0f}', ha='center', fontsize=11.5)
    ax.set_xticks(range(len(keys)), [CONF[k][1] for k in keys], fontsize=11.5)
    ax.set_ylabel('Final success (%)'); ax.set_ylim(0, 104); ax.grid(axis='x', visible=False)
    ax.legend(handles=[Patch(color=INK2, alpha=0.35, label='Training goals'), Patch(color=INK2, label='Held-out goals')],
              loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, fontsize=11)
    save(fig, out, 'final_train_vs_test.png')
    return rows


def per_task(runs, out, keys):
    tasks = runs[0]['state']['task_names']
    M = np.array([np.mean([r['state']['final_per_task_success'] for r in runs if r['key'] == k], 0) * 100 for k in keys])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('b', ['#f4f8fd', '#b7d3f6', '#5598e7', '#1c5cab', '#0d366b'])
    fig, ax = plt.subplots(figsize=(12, 0.5 * len(keys) + 1.8))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=11, color='white' if M[i, j] > 60 else INK)
    ax.set_yticks(range(len(keys)), [CONF[k][1].replace('\n', ' ') for k in keys])
    ax.set_xticks(range(len(tasks)), [t.replace('-v3', '') for t in tasks], rotation=35, ha='right')
    ax.grid(False); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, out, 'per_task.png')
    return dict(tasks=tasks, keys=keys, matrix=M.round(1).tolist())


def growth_traj(runs, out, panels):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.0), sharey=True, gridspec_kw=dict(wspace=0.07))
    for ax, g in zip(np.atleast_1d(axes), panels):
        for ref in ('d1', 'd3', 'd4') if g == 'early4' else ('d1', 'd3'):
            for r in [r for r in runs if r['key'] == ref]:
                ax.plot(np.array(EVAL_STEPS) / 1e6, [at(r, t) * 100 for t in EVAL_STEPS], color=col(ref), lw=1.1, alpha=0.6)
        gr = [r for r in runs if r['key'] == g]
        for r in gr:
            ax.plot(np.array(EVAL_STEPS) / 1e6, [at(r, t) * 100 for t in EVAL_STEPS], color=col(g), lw=2.1)
        for x in gr[0]['state']['grow_at']:
            ax.axvline(x / 1e6, color=col(g), lw=1, ls=':')
        ax.set_title(lab(g).replace('Grow ', 'Grow ').replace(' at ', '\nat '), fontsize=12.5, color=INK)
        ax.set_xlabel('Environment steps (M)'); ax.set_ylim(0, 100)
    np.atleast_1d(axes)[0].set_ylabel('Held-out success (%)')
    np.atleast_1d(axes)[-1].legend(handles=[Line2D([], [], color=col('d1'), lw=1.5, label='Fixed depth 1'),
                                            Line2D([], [], color=col('d3'), lw=1.5, label='Fixed depth 3'),
                                            Line2D([], [], color=col('d4'), lw=1.5, label='Fixed depth 4'),
                                            Line2D([], [], color=INK2, lw=2.1, label='Growth run')],
                                   loc='lower right', fontsize=10)
    save(fig, out, 'growth_traj.png')


def insertion(runs, out):
    """Identity vs random initialisation of the inserted layers (same schedule): success on the
    training episodes completed in each 100k-step rollout, and held-out success at evaluations."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.25))
    stats = {}
    for k in ('early', 'early_rand', 'd1'):
        rs = [r for r in runs if r['key'] == k]
        if not rs:
            continue
        xs = [m['training_steps'] for m in rs[0]['me']]
        v = []
        for r in rs:
            ts = r['me']
            v.append([sum(x['successes'] for x in m['train_completed_episode_success'].values()) /
                      max(1, sum(x['episodes'] for x in m['train_completed_episode_success'].values())) * 100 for m in ts])
        v = np.array(v)
        a.plot(np.array(xs) / 1e6, v.mean(0), color=col(k), lw=1.6 if k == 'd1' else 2.3,
               label=f'{lab(k)} (n={len(rs)})' if k != 'd1' else 'Fixed depth 1 (no insertion)', alpha=0.8 if k == 'd1' else 1)
        # change across each insertion: last round before vs first round after
        if k != 'd1':
            d = []
            for g in rs[0]['state']['grow_at']:
                i = xs.index(g)
                d.append((v[:, i + 1] - v[:, i]).tolist())
            stats[k] = d
    for g in (500_000, 1_000_000):
        a.axvline(g / 1e6, color=INK2, lw=1, ls=':')
    a.set_xlabel('Environment steps (M)'); a.set_ylabel('Success of training episodes (%)')
    a.set_ylim(0, 100); a.legend(loc='lower right', fontsize=10.5)
    a.set_title('Per-rollout training success', fontsize=13)
    keys = [k for k in ('d1', 'early_rand', 'early', 'd3') if any(r['key'] == k for r in runs)]
    for i, k in enumerate(keys):
        te = [final(r) * 100 for r in runs if r['key'] == k]
        b.bar(i, np.mean(te), 0.6, color=col(k), edgecolor='none')
        b.scatter([i] * len(te), te, s=20, color=INK, zorder=3)
        b.text(i, max(te) + 2.5, f'{np.mean(te):.0f}', ha='center', fontsize=11.5)
    b.set_xticks(range(len(keys)), [CONF[k][1] for k in keys], fontsize=10.5)
    b.set_ylim(0, 104); b.grid(axis='x', visible=False); b.set_ylabel('Final held-out success (%)')
    b.set_title('After 3M steps', fontsize=13)
    save(fig, out, 'insertion.png')
    return stats


def flops_per_run(runs, out, keys):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    vals = {}
    for i, k in enumerate(keys):
        f = np.mean([cum_flops(r) for r in runs if r['key'] == k]) / 1e15
        vals[k] = float(f)
        ax.bar(i, f, 0.6, color=col(k), edgecolor='none')
        ax.text(i, f + 0.004, f'{f:.2f}', ha='center', va='bottom', fontsize=11.5)
    ax.set_xticks(range(len(keys)), [CONF[k][1] for k in keys], fontsize=11)
    ax.set_ylabel('Learner compute per run (PFLOP)'); ax.grid(axis='x', visible=False)
    save(fig, out, 'flops_per_run.png')
    return vals


# ---------------------------------------------------------------- main
def summarise(res, runs):
    s = {}
    for k, v in res.items():
        t = [x['test'] * 100 for x in v]
        s[k] = dict(n=len(v), mean=float(np.mean(t)), se=float(np.std(t, ddof=1) / np.sqrt(len(t))) if len(t) > 1 else None,
                    steps_M=float(np.mean([x['steps'] for x in v]) / 1e6), pflop=float(np.mean([x['flops'] for x in v]) / 1e15),
                    picks=[x['pick'] for x in v], per_seed=dict(zip([x['seed'] for x in v], [round(x, 1) for x in t])))
    paired = {}
    for a in [k for k in GROW if res.get(k)]:
        for b in ['oracle', 'grid', 'sh', 'default', 'random'] + [k for k in GROW if k != a and res.get(k)]:
            ma = {x['seed']: x['test'] for x in res[a]}; mb = {x['seed']: x['test'] for x in res[b]}
            common = sorted(set(ma) & set(mb))
            if len(common) >= 2:
                m, h = tci([100 * (ma[i] - mb[i]) for i in common])
                paired[f'{a} - {b}'] = dict(n=len(common), mean=round(m, 2), ci95=round(h, 2))
    return s, paired


def main():
    runs = load()
    summary = dict(n_runs=len(runs))
    for w in sorted({r['width'] for r in runs}):
        sub = [r for r in runs if r['width'] == w]
        out = FIGS / f'w{w}'; out.mkdir(parents=True, exist_ok=True)
        present = [k for k in FIXED + GROW if any(r['key'] == k for r in sub)]
        grows = [k for k in GROW if k in present]
        res = strategies(sub)
        strat, paired = summarise(res, sub)
        strategy_panel(res, out, ['oracle', 'grid', 'sh'] + [k for k in ('early4', 'early', 'vearly', 'late') if k in grows] + ['default', 'random'])
        anytime(sub, out, [k for k in ('early4', 'early', 'vearly', 'late') if k in grows])
        curves(sub, out, 'success_test'); curves(sub, out, 'success_train'); curves(sub, out, 'success_dev')
        tv = train_vs_test(sub, out, present)
        pt = per_task(sub, out, present)
        fl = flops_per_run(sub, out, present)
        ins = {}
        if w == 400:
            growth_traj(sub, out, [k for k in ('vearly', 'early', 'late', 'early4') if k in grows])
            if 'early_rand' in grows:
                ins = insertion(sub, out)
        max_err = max((e.get('mean_max_error', 0) for r in sub for e in r['gr'] if e['kind'] == 'identity'), default=None)
        n_ident = sum(1 for r in sub for e in r['gr'] if e['kind'] == 'identity')
        summary[f'w{w}'] = dict(configs={k: sum(r['key'] == k for r in sub) for k in present}, strategies=strat, paired=paired,
                                train_vs_test=tv, per_task=pt, pflop_per_run=fl, insertion_change=ins,
                                identity_insertions=n_ident, identity_max_mean_error=max_err,
                                per_seed={k: {r['seed']: round(final(r) * 100, 1) for r in sub if r['key'] == k} for k in present})
        print(f'width {w}:', {k: (round(v['mean'], 1), round(v['pflop'], 3)) for k, v in strat.items()})
    (ROOT / 'results' / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('wrote', ROOT / 'results' / 'summary.json')


if __name__ == '__main__':
    main()
