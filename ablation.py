"""
Ablation Study — MMM02 Feature Contribution
Runs 25 trials for each progressive configuration of MMM02,
isolating the contribution of each component.

Run AFTER implementing the full MMM02 in BSE_Coursework.py.

Configurations (progressive):
    MMM01 Baseline  — benchmark, no features
    +Inventory      — FIFO 3-unit inventory only
    +Regime         — + session regime switching
    +Kelly          — + fractional Kelly position sizing
    +AsymmetricSell — + asymmetric sell / stop-loss
    Full MMM02      — + XGBoost buy signal (complete algorithm)
"""

import sys, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import importlib.util

# ── import BSE ────────────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("BSE_Coursework", "BSE_Coursework.py")
BSE  = importlib.util.module_from_spec(spec)
sys.modules['BSE_Coursework'] = BSE
spec.loader.exec_module(BSE)

# ── monkey-patch to capture balance history ───────────────────────────────────
_balance_history = {}

def _record(trader, time):
    tid = trader.tid
    if tid not in _balance_history:
        _balance_history[tid] = []
    lpp = trader.last_purchase_price if trader.last_purchase_price else 0
    _balance_history[tid].append((time, trader.balance, trader.balance + lpp))

_orig01_init = BSE.TraderMMM01.__init__
_orig02_init = BSE.TraderMMM02.__init__

def _init01(self, ttype, tid, balance, params, time):
    _orig01_init(self, ttype, tid, balance, params, time)
    _record(self, time)

def _init02(self, ttype, tid, balance, params, time):
    _orig02_init(self, ttype, tid, balance, params, time)
    _record(self, time)

BSE.TraderMMM01.__init__ = _init01
BSE.TraderMMM02.__init__ = _init02

_orig01_bk = BSE.TraderMMM01.bookkeep
_orig02_bk = BSE.TraderMMM02.bookkeep

def _bk01(self, time, trade, order, vrbs):
    _orig01_bk(self, time, trade, order, vrbs)
    _record(self, time)

def _bk02(self, time, trade, order, vrbs):
    _orig02_bk(self, time, trade, order, vrbs)
    _record(self, time)

BSE.TraderMMM01.bookkeep = _bk01
BSE.TraderMMM02.bookkeep = _bk02

# ── preprocess SPY CSV ────────────────────────────────────────────────────────
raw_spy   = 'spy_1m_2026-Training.csv'
clean_spy = 'spy_bse_format.csv'
with open(raw_spy, newline='') as fin, open(clean_spy, 'w', newline='') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    next(reader)
    for row in reader:
        writer.writerow([row[7].split(' ')[1][:8], row[3]])
print("SPY data preprocessed.")

# ── simulation config ─────────────────────────────────────────────────────────
N_TRIALS   = 25
INIT_BAL   = 500
start_time = 0
end_time   = int(7.5 * 60 * 60)

sellers_spec = [('ZIP', 3), ('SHVR', 3), ('GVWY', 3), ('SNPR', 3)]
buyers_spec  = [('ZIP', 3), ('ZIC', 3),  ('SHVR', 3), ('GVWY', 3)]

offsetfn, offsetfn_params = BSE.offset_from_file(clean_spy, 0, 1, 75, end_time)
supply_schedule = [{'from': start_time, 'to': end_time,
                    'ranges': [(100, 200)], 'stepmode': 'random',
                    'offsetfn': offsetfn, 'offsetfn_params': offsetfn_params}]
demand_schedule = [{'from': start_time, 'to': end_time,
                    'ranges': [(50, 150)], 'stepmode': 'random',
                    'offsetfn': offsetfn, 'offsetfn_params': offsetfn_params}]
order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
               'interval': 30, 'timemode': 'drip-jitter'}
dump_flags  = {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False,
               'dump_avgbals': False, 'dump_tape': False}

# ── ablation configurations ───────────────────────────────────────────────────
# Each dict is passed as the params argument to MMM02.
# Flags are read inside MMM02's respond() / __init__ to enable/disable features.
# Inventory management is always active (baked into bookkeep rewrite).
configs = [
    {
        'label':               'MMM01\nBaseline',
        'use_regime':          False,
        'use_kelly':           False,
        'use_asymmetric_sell': False,
        'use_xgb':             False,
    },
    {
        'label':               '+Inventory\n(FIFO×3)',
        'use_regime':          False,
        'use_kelly':           False,
        'use_asymmetric_sell': False,
        'use_xgb':             False,
    },
    {
        'label':               '+Regime\nSwitching',
        'use_regime':          True,
        'use_kelly':           False,
        'use_asymmetric_sell': False,
        'use_xgb':             False,
    },
    {
        'label':               '+Fractional\nKelly',
        'use_regime':          True,
        'use_kelly':           True,
        'use_asymmetric_sell': False,
        'use_xgb':             False,
    },
    {
        'label':               '+Asymmetric\nSell',
        'use_regime':          True,
        'use_kelly':           True,
        'use_asymmetric_sell': True,
        'use_xgb':             False,
    },
    {
        'label':               'Full MMM02\n(+XGBoost)',
        'use_regime':          True,
        'use_kelly':           True,
        'use_asymmetric_sell': True,
        'use_xgb':             True,
    },
]

# ── run ablation loop ─────────────────────────────────────────────────────────
results = {}   # label -> {'pnl01': array, 'pnl02': array, 'trades02': array}

for cfg_idx, config in enumerate(configs):
    label = config['label']
    pnl01_list, pnl02_list, trades02_list = [], [], []

    print(f"\n{'='*55}")
    print(f"Config {cfg_idx+1}/{len(configs)}: {label.replace(chr(10), ' ')}")
    print(f"{'='*55}")

    for trial in range(N_TRIALS):
        _balance_history.clear()

        mm01_params = {}                     # MMM01 always unchanged
        mm02_params = {k: v for k, v in config.items() if k != 'label'}

        mm_spec      = [('MMM01', 1, mm01_params), ('MMM02', 1, mm02_params)]
        traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec,
                        'mrktmakers': mm_spec}

        BSE.market_session(
            f'ablation_{cfg_idx:02d}_trial_{trial:02d}',
            start_time, end_time, traders_spec, order_sched, dump_flags, False
        )

        keys = sorted(_balance_history.keys())
        if len(keys) < 2:
            print(f"  WARNING trial {trial}: fewer than 2 MM traders — skipping")
            continue

        p01 = _balance_history[keys[0]][-1][1] - INIT_BAL
        p02 = _balance_history[keys[1]][-1][1] - INIT_BAL
        n02 = len(_balance_history[keys[1]]) - 1   # subtract initial record

        pnl01_list.append(p01)
        pnl02_list.append(p02)
        trades02_list.append(n02)

        print(f"  Trial {trial+1:2d}/{N_TRIALS}  "
              f"MMM01={p01:+.0f}¢  MMM02={p02:+.0f}¢  "
              f"diff={p02-p01:+.0f}¢")

    results[label] = {
        'pnl01':    np.array(pnl01_list,    dtype=float),
        'pnl02':    np.array(pnl02_list,    dtype=float),
        'trades02': np.array(trades02_list, dtype=float),
    }

    arr01 = results[label]['pnl01']
    arr02 = results[label]['pnl02']
    t_stat, p_val   = stats.ttest_rel(arr01, arr02)
    w_stat, p_val_w = stats.wilcoxon(arr01, arr02)
    cohens_d = (arr02.mean() - arr01.mean()) / (np.std(arr02 - arr01) + 1e-8)
    print(f"  Summary: MMM02 mean={arr02.mean():.1f}¢  std={arr02.std():.1f}¢")
    print(f"  vs MMM01 mean={arr01.mean():.1f}¢  diff={arr02.mean()-arr01.mean():+.1f}¢")
    print(f"  Paired t: t={t_stat:.3f} p={p_val:.4f}  "
          f"Wilcoxon: W={w_stat:.1f} p={p_val_w:.4f}  "
          f"Cohen's d={cohens_d:.3f}")

# ── compute stats for all configs ─────────────────────────────────────────────
print("\n\n" + "="*70)
print("ABLATION RESULTS SUMMARY")
print("="*70)
print(f"{'Config':<25} {'MMM02 Mean':>10} {'MMM02 Std':>10} "
      f"{'vs MMM01':>10} {'p (t-test)':>12} {'Cohen d':>9}")
print("-"*70)

mmm01_pnl = results[configs[0]['label']]['pnl01']   # MMM01 baseline P&L

for config in configs:
    label  = config['label'].replace('\n', ' ')
    r      = results[config['label']]
    arr02  = r['pnl02']
    arr01  = r['pnl01']
    diff   = arr02.mean() - arr01.mean()
    _, pv  = stats.ttest_rel(arr01, arr02)
    cd     = (arr02.mean() - arr01.mean()) / (np.std(arr02 - arr01) + 1e-8)
    print(f"  {label:<23} {arr02.mean():>10.1f}¢ {arr02.std():>10.1f}¢ "
          f"{diff:>+10.1f}¢ {pv:>12.4f} {cd:>9.3f}")

# ── plotting ──────────────────────────────────────────────────────────────────
DARK  = '#0d1117'
PANEL = '#161b22'
GRID  = '#21262d'
TEXT  = '#e6edf3'
MUTED = '#8b949e'
GOLD  = '#d29922'
C1    = '#58a6ff'

# gradient colour per config — from muted to bright
config_colours = ['#8b949e', '#3fb950', '#58a6ff', '#d29922', '#f78166', '#bc8cff']
labels_short   = [c['label'] for c in configs]

fig = plt.figure(figsize=(16, 13), facecolor=DARK)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ax_means = fig.add_subplot(gs[0, :])   # mean P&L per config — full width
ax_box   = fig.add_subplot(gs[1, 0])   # box plot
ax_delta = fig.add_subplot(gs[1, 1])   # incremental gain per feature

for ax in [ax_means, ax_box, ax_delta]:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, lw=0.5, ls='--', alpha=0.8)

# ── mean P&L bar chart with error bars ────────────────────────────────────────
means01 = [results[c['label']]['pnl01'].mean() for c in configs]
means02 = [results[c['label']]['pnl02'].mean() for c in configs]
stds02  = [results[c['label']]['pnl02'].std()  for c in configs]
p_vals  = []
for c in configs:
    r = results[c['label']]
    _, pv = stats.ttest_rel(r['pnl01'], r['pnl02'])
    p_vals.append(pv)

x      = np.arange(len(configs))
width  = 0.35

# MMM01 bars (consistent reference)
ax_means.bar(x - width/2, means01, width, color=C1, alpha=0.5,
             edgecolor=DARK, label='MMM01 (benchmark)')
# MMM02 bars
bars = ax_means.bar(x + width/2, means02, width, color=config_colours,
                    alpha=0.85, edgecolor=DARK, label='MMM02 (config)')
ax_means.errorbar(x + width/2, means02, yerr=stds02,
                  fmt='none', color=TEXT, capsize=4, lw=1.2)

# significance stars
for i, (m, p) in enumerate(zip(means02, p_vals)):
    star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    col  = '#3fb950' if p < 0.05 else MUTED
    ax_means.text(x[i] + width/2, m + stds02[i] + 15,
                  star, ha='center', fontsize=9, color=col, fontweight='bold')

ax_means.set_xticks(x)
ax_means.set_xticklabels(labels_short, fontsize=8)
ax_means.set_ylabel('Mean Final P&L (¢)', fontsize=9)
ax_means.set_title(f'Ablation Study — Mean P&L per Configuration (N={N_TRIALS} trials each)\n'
                   '* p<0.05  ** p<0.01  *** p<0.001  ns = not significant',
                   fontsize=10, pad=6)
ax_means.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT,
                framealpha=0.9, edgecolor=GRID)

# ── box plot of MMM02 P&L distributions ───────────────────────────────────────
pnl02_arrays = [results[c['label']]['pnl02'] for c in configs]
bp = ax_box.boxplot(pnl02_arrays, patch_artist=True,
                    medianprops=dict(color='white', lw=2),
                    whiskerprops=dict(color=TEXT, lw=1),
                    capprops=dict(color=TEXT, lw=1),
                    flierprops=dict(marker='.', markerfacecolor=MUTED,
                                   markersize=3, alpha=0.5))
for patch, col in zip(bp['boxes'], config_colours):
    patch.set_facecolor(col + '66')
    patch.set_edgecolor(col)

# MMM01 reference line
ax_box.axhline(np.mean(means01), color=C1, lw=1.5, ls='--', alpha=0.7,
               label=f'MMM01 mean = {np.mean(means01):.0f}¢')
ax_box.set_xticks(range(1, len(configs)+1))
ax_box.set_xticklabels(labels_short, fontsize=7)
ax_box.set_title('MMM02 P&L Distribution per Configuration', fontsize=10, pad=6)
ax_box.set_ylabel('Final P&L (¢)', fontsize=9)
ax_box.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT,
              framealpha=0.9, edgecolor=GRID)

# ── incremental gain per feature ──────────────────────────────────────────────
# difference in mean MMM02 P&L from one config to the next
incremental_labels = [
    'Inventory\nvs Baseline',
    'Regime\nvs Inventory',
    'Kelly\nvs Regime',
    'AsymSell\nvs Kelly',
    'XGBoost\nvs AsymSell',
]
incremental_gains = [means02[i+1] - means02[i] for i in range(len(means02)-1)]
inc_colours = ['#3fb950' if g >= 0 else '#f78166' for g in incremental_gains]

bars2 = ax_delta.bar(range(len(incremental_gains)), incremental_gains,
                     color=inc_colours, edgecolor=DARK, alpha=0.85)
ax_delta.axhline(0, color=TEXT, lw=1)
for bar, val in zip(bars2, incremental_gains):
    ypos = val + 5 if val >= 0 else val - 20
    ax_delta.text(bar.get_x() + bar.get_width()/2, ypos,
                  f'{val:+.0f}¢', ha='center', fontsize=9,
                  color=TEXT, fontweight='bold')
ax_delta.set_xticks(range(len(incremental_labels)))
ax_delta.set_xticklabels(incremental_labels, fontsize=8)
ax_delta.set_title('Incremental P&L Gain per Feature Added', fontsize=10, pad=6)
ax_delta.set_ylabel('Incremental Mean P&L Change (¢)', fontsize=9)

fig.suptitle(f'MMM02 Ablation Study — Feature Contribution Analysis  (N={N_TRIALS} trials per config)',
             fontsize=13, fontweight='bold', color=TEXT, y=0.998)

out = 'mmm_ablation.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK)
print(f"\nSaved → {out}")
plt.close()

# ── save results table as CSV for the paper ───────────────────────────────────
with open('mmm_ablation_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Config', 'MMM01 Mean', 'MMM01 Std',
                     'MMM02 Mean', 'MMM02 Std', 'Diff',
                     't-stat', 'p-value (t)', 'p-value (W)', "Cohen's d"])
    for config in configs:
        label = config['label'].replace('\n', ' ')
        r     = results[config['label']]
        a1, a2 = r['pnl01'], r['pnl02']
        t_s, p_t = stats.ttest_rel(a1, a2)
        w_s, p_w = stats.wilcoxon(a1, a2)
        cd       = (a2.mean() - a1.mean()) / (np.std(a2 - a1) + 1e-8)
        writer.writerow([
            label,
            f'{a1.mean():.1f}', f'{a1.std():.1f}',
            f'{a2.mean():.1f}', f'{a2.std():.1f}',
            f'{a2.mean()-a1.mean():+.1f}',
            f'{t_s:.3f}', f'{p_t:.4f}', f'{p_w:.4f}', f'{cd:.3f}'
        ])

print("Saved → mmm_ablation_results.csv")
print("\nDone. Files produced:")
print("  mmm_ablation.png          — figure for paper")
print("  mmm_ablation_results.csv  — table data for paper") 