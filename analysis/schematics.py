"""Schematic figures for the report: search vs growth, goal split, identity insertion, MT10 frames.

usage: python analysis/schematics.py [OUT_DIR]   (default figures/static; MT10 frames are read from OUT_DIR/mt10_renders,
which analysis/render_mt10.py produces on a machine with metaworld)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / 'figures' / 'static'
OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
DEPTH_COL = {1: '#86b6ef', 2: '#5598e7', 3: '#256abf', 4: '#0d366b'}
GROW_COL = '#eb6834'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF})


def bar(ax, x0, x1, y, col, h=0.5, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h, boxstyle='round,pad=0,rounding_size=0.12',
                                fc=col, ec='none', alpha=alpha))


# 1. Search vs grow schematic
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(wspace=0.12))
for ax in (a, b):
    ax.set_xlim(-0.2, 3.3); ax.set_ylim(-0.6, 4.4); ax.axis('off')
for i, d in enumerate([4, 3, 2, 1]):
    bar(a, 0, 3, i, DEPTH_COL[d])
    a.text(-0.12, i, f'depth {d}', ha='right', va='center', color=INK2, fontsize=13)
a.text(1.5, 4.05, 'Search: K runs, keep the best', ha='center', color=INK, fontsize=16, weight='bold')
a.text(3.15, 1.5, '× K\ncost', ha='left', va='center', color=INK2, fontsize=13)
for (x0, x1, d) in [(0, 1, 1), (1, 2, 2), (2, 3, 3)]:
    bar(b, x0 + (0.02 if x0 else 0), x1, 1.5, DEPTH_COL[d] if False else GROW_COL, alpha=0.45 + 0.25 * (d - 1))
    b.text((x0 + x1) / 2, 1.5, f'depth {d}', ha='center', va='center', color='white', fontsize=13, weight='bold')
for x in (1, 2):
    b.annotate('', xy=(x, 2.05), xytext=(x, 2.75), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.6))
    b.text(x, 2.95, '+ layer', ha='center', color=INK, fontsize=12)
b.text(1.5, 4.05, 'Grow: one run, add depth on the way', ha='center', color=INK, fontsize=16, weight='bold')
b.text(3.15, 1.5, '× 1\ncost', ha='left', va='center', color=INK2, fontsize=13)
fig.savefig(OUT / 'schematic_search_vs_grow.png', dpi=200, bbox_inches='tight'); plt.close(fig)

# 2. Goal split (per task, 50 goal variants)
fig, ax = plt.subplots(figsize=(12, 1.9))
ax.set_xlim(0, 50); ax.set_ylim(0, 1); ax.axis('off')
for i in range(50):
    c = '#b7d3f6' if i < 30 else ('#9085e9' if i < 40 else GROW_COL)
    ax.add_patch(FancyBboxPatch((i + 0.08, 0.35), 0.84, 0.5, boxstyle='round,pad=0,rounding_size=0.12', fc=c, ec='none'))
for x, t in [(15, 'train goals (30)'), (35, 'validation (10)'), (45, 'held-out test (10)')]:
    ax.text(x, 0.08, t, ha='center', color=INK, fontsize=14)
fig.savefig(OUT / 'goal_split.png', dpi=200, bbox_inches='tight'); plt.close(fig)


# 4. Identity insertion schematic
def block(ax, x, y, w, h, text, fc, tc='white', sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.08', fc=fc, ec='none'))
    ax.text(x + w / 2, y + h / 2 + (0.09 if sub else 0), text, ha='center', va='center', color=tc, fontsize=14, weight='bold')
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.17, sub, ha='center', va='center', color=tc, fontsize=11)


def arrow(ax, x0, x1, y):
    ax.annotate('', xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.6))


fig, ax = plt.subplots(figsize=(14, 5.0))
ax.set_xlim(0, 14.2); ax.set_ylim(-0.5, 4.5); ax.axis('off')
KEEP = '#256abf'; HEAD = '#52514e'
for y, new in [(2.9, False), (0.8, True)]:
    ax.text(0.0, y + 1.0, 'After inserting a layer' if new else 'Before', fontsize=15, weight='bold', color=INK)
    ax.text(0.3, y + 0.35, 's', fontsize=18, style='italic', ha='center', va='center', color=INK)
    arrow(ax, 0.6, 1.1, y + 0.35)
    block(ax, 1.1, y, 2.5, 0.7, 'Linear + ReLU', KEEP, sub='trained layers')
    arrow(ax, 3.7, 4.25, y + 0.35)
    ax.text(4.8, y + 0.35, 'h ≥ 0', fontsize=16, ha='center', va='center', color=INK)
    if new:
        arrow(ax, 5.35, 5.9, y + 0.35)
        block(ax, 5.9, y, 3.3, 0.7, 'Identity layer + ReLU', GROW_COL, sub='W = I, no bias')
        arrow(ax, 9.3, 9.8, y + 0.35)
        ax.text(10.05, y + 0.35, 'h', fontsize=16, ha='center', va='center', color=INK)
        arrow(ax, 10.3, 10.75, y + 0.35)
    else:
        arrow(ax, 5.35, 10.75, y + 0.35)
    block(ax, 10.75, y, 2.1, 0.7, 'Policy head', HEAD, sub='mean, std')
    arrow(ax, 12.95, 13.35, y + 0.35)
    ax.text(13.8, y + 0.35, 'π(a|s)', fontsize=16, ha='center', va='center', color=INK)
ax.text(7.1, -0.25, 'h ≥ 0  ⇒  ReLU(I·h) = h  ⇒  the policy is identical at the moment of insertion',
        ha='center', va='center', fontsize=15, color=INK)
fig.savefig(OUT / 'identity_schematic.png', dpi=200, bbox_inches='tight', pad_inches=0.1); plt.close(fig)
print('schematic ok')

# 8. Real MT10 renders grid (frames from render_mt10.py)
from PIL import Image as _Image
rd = OUT / 'mt10_renders'
if rd.exists():
    order = ['reach', 'push', 'pick-place', 'door-open', 'drawer-open',
             'drawer-close', 'button-press-topdown', 'peg-insert-side', 'window-open', 'window-close']
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.1), gridspec_kw=dict(wspace=0.04, hspace=0.12))
    for ax, name in zip(axes.flat, order):
        im = _Image.open(rd / f'{name}-v3.png')
        w, h = im.size
        ax.imshow(im.crop((0, int(h * 0.06), w, int(h * 0.94))))
        ax.set_title(name, fontsize=14, color=INK, pad=6); ax.axis('off')
    fig.savefig(OUT / 'mt10_real.png', dpi=150, bbox_inches='tight', pad_inches=0.05); plt.close(fig)
    print('mt10 real grid ok')
