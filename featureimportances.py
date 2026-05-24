"""
Feature Importance Extraction — Full MMM02 XGBoost
Runs 25 trials of Full MMM02 and extracts XGBoost feature importances
from each calibration, then plots the averaged importances.

Run AFTER implementing Full MMM02 in BSE_Coursework.py.
Takes ~20-30 minutes.
"""

import sys, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib.util

# ── import BSE ────────────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("BSE_Coursework", "BSE_Coursework.py")
BSE  = importlib.util.module_from_spec(spec)
sys.modules['BSE_Coursework'] = BSE
spec.loader.exec_module(BSE)

# ── monkey-patch to capture feature importances and balance ───────────────────
_balance_history    = {}
_feature_importances = []   # list of importance arrays, one per trial

# feature names matching _build_features() order
FEATURE_NAMES = [
    'Normalised\nRolling Mean',
    'Rolling\nStd Dev',
    'Lag-1\nAutocorr',
    'Last 1-bar\nReturn',
    '3-bar\nReturn',
    'Session\nFraction',
    'Deviation\nfrom Mean',
    'Vol\nRatio',
]

def _record(trader, time):
    tid = trader.tid
    if tid not in _balance_history:
        _balance_history[tid] = []
    lpp = trader.last_purchase_price if trader.last_purchase_price else 0
    _balance_history[tid].append((time, trader.balance, trader.balance + lpp))

# patch MMM01
_orig01_init = BSE.TraderMMM01.__init__
_orig01_bk   = BSE.TraderMMM01.bookkeep

def _init01(self, ttype, tid, balance, params, time):
    _orig01_init(self, ttype, tid, balance, params, time)
    _record(self, time)

def _bk01(self, time, trade, order, vrbs):
    _orig01_bk(self, time, trade, order, vrbs)
    _record(self, time)

BSE.TraderMMM01.__init__ = _init01
BSE.TraderMMM01.bookkeep = _bk01

# patch MMM02 — also intercept _calibrate to grab feature importances
_orig02_init     = BSE.TraderMMM02.__init__
_orig02_bk       = BSE.TraderMMM02.bookkeep
_orig02_calibrate = BSE.TraderMMM02._calibrate

def _init02(self, ttype, tid, balance, params, time):
    _orig02_init(self, ttype, tid, balance, params, time)
    _record(self, time)

def _bk02(self, time, trade, order, vrbs):
    _orig02_bk(self, time, trade, order, vrbs)
    _record(self, time)

def _calibrate_patched(self, lob, time, end_time):
    # call original calibration
    _orig02_calibrate(self, lob, time, end_time)
    # extract feature importances if XGBoost model was trained
    if self.xgb_model is not None:
        try:
            importances = self.xgb_model.feature_importances_
            _feature_importances.append(importances.copy())
            print(f'  Feature importances captured: {importances.round(3)}')
        except Exception as e:
            print(f'  Could not extract feature importances: {e}')

BSE.TraderMMM02.__init__    = _init02
BSE.TraderMMM02.bookkeep    = _bk02
BSE.TraderMMM02._calibrate  = _calibrate_patched

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

# full MMM02 params
full_params = {
    'use_regime':          True,
    'use_kelly':           True,
    'use_asymmetric_sell': True,
    'use_xgb':             True,
}

# ── run trials ────────────────────────────────────────────────────────────────
pnl01_list, pnl02_list = [], []

print(f"\nRunning {N_TRIALS} trials of Full MMM02 vs MMM01...")
print("="*55)

for trial in range(N_TRIALS):
    _balance_history.clear()

    mm_spec      = [('MMM01', 1, {}), ('MMM02', 1, full_params)]
    traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec,
                    'mrktmakers': mm_spec}

    BSE.market_session(
        f'feat_imp_trial_{trial:02d}',
        start_time, end_time, traders_spec, order_sched, dump_flags, False
    )

    keys = sorted(_balance_history.keys())
    if len(keys) < 2:
        print(f"WARNING trial {trial}: fewer than 2 MM traders")
        continue

    p01 = _balance_history[keys[0]][-1][1] - INIT_BAL
    p02 = _balance_history[keys[1]][-1][1] - INIT_BAL
    pnl01_list.append(p01)
    pnl02_list.append(p02)

    print(f"Trial {trial+1:2d}/{N_TRIALS}  "
          f"MMM01={p01:+.0f}¢  MMM02={p02:+.0f}¢  "
          f"diff={p02-p01:+.0f}¢  "
          f"importances captured: {len(_feature_importances)}")

pnl01 = np.array(pnl01_list, dtype=float)
pnl02 = np.array(pnl02_list, dtype=float)

print(f"\nMMM01 mean={pnl01.mean():.1f}¢  std={pnl01.std():.1f}¢")
print(f"MMM02 mean={pnl02.mean():.1f}¢  std={pnl02.std():.1f}¢")
print(f"Feature importance arrays collected: {len(_feature_importances)}")

# ── compute mean feature importances ─────────────────────────────────────────
if len(_feature_importances) == 0:
    print("ERROR: No feature importances captured — XGBoost may not have trained.")
    sys.exit(1)

imp_array = np.array(_feature_importances, dtype=float)   # shape: (n_trials, 8)
mean_imp  = imp_array.mean(axis=0)
std_imp   = imp_array.std(axis=0)

print("\nMean feature importances:")
for name, m, s in zip(FEATURE_NAMES, mean_imp, std_imp):
    print(f"  {name.replace(chr(10), ' '):30s}: {m:.4f} ± {s:.4f}")

# sort by importance
sort_idx   = np.argsort(mean_imp)[::-1]
sorted_names = [FEATURE_NAMES[i] for i in sort_idx]
sorted_mean  = mean_imp[sort_idx]
sorted_std   = std_imp[sort_idx]

# ── plotting ──────────────────────────────────────────────────────────────────
DARK  = '#0d1117'
PANEL = '#161b22'
GRID  = '#21262d'
TEXT  = '#e6edf3'
MUTED = '#8b949e'
C1    = '#58a6ff'
C2    = '#f78166'
GOLD  = '#d29922'

# colour bars by which data analysis finding they relate to
# session fraction → regime finding (orange)
# autocorr, dev from mean → mean reversion finding (blue)
# vol ratio, std → ARCH/volatility finding (purple)
# ret1, ret3 → return signal (green)
# normalised mean → general (grey)
feature_group_colours = {
    'Normalised\nRolling Mean': '#8b949e',
    'Rolling\nStd Dev':         '#bc8cff',
    'Lag-1\nAutocorr':          '#58a6ff',
    'Last 1-bar\nReturn':       '#3fb950',
    '3-bar\nReturn':            '#3fb950',
    'Session\nFraction':        '#d29922',
    'Deviation\nfrom Mean':     '#58a6ff',
    'Vol\nRatio':               '#bc8cff',
}
bar_colours = [feature_group_colours[n] for n in sorted_names]

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=DARK)
fig.subplots_adjust(wspace=0.38)

ax_imp  = axes[0]
ax_dist = axes[1]

for ax in axes:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, lw=0.5, ls='--', alpha=0.8)

# ── feature importance bar chart ──────────────────────────────────────────────
x = np.arange(len(sorted_names))
bars = ax_imp.bar(x, sorted_mean, color=bar_colours, edgecolor=DARK,
                  alpha=0.85, yerr=sorted_std, capsize=4,
                  error_kw=dict(color=TEXT, lw=1.2))

ax_imp.set_xticks(x)
ax_imp.set_xticklabels(sorted_names, fontsize=8)
ax_imp.set_ylabel('Mean Feature Importance (XGBoost gain)', fontsize=9)
ax_imp.set_title(f'XGBoost Feature Importances\nAveraged across {len(_feature_importances)} trials',
                 fontsize=10, pad=6)

# value labels
for bar, val, err in zip(bars, sorted_mean, sorted_std):
    ax_imp.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + err + 0.005,
                f'{val:.3f}', ha='center', fontsize=8,
                color=TEXT, fontweight='bold')

# legend for colour groups
from matplotlib.patches import Patch
legend_elements = [
    Patch(color='#d29922', label='Session/regime signal'),
    Patch(color='#58a6ff', label='Mean reversion signal'),
    Patch(color='#bc8cff', label='Volatility/ARCH signal'),
    Patch(color='#3fb950', label='Return signal'),
    Patch(color='#8b949e', label='Price level signal'),
]
ax_imp.legend(handles=legend_elements, fontsize=7, facecolor=PANEL,
              labelcolor=TEXT, framealpha=0.9, edgecolor=GRID,
              loc='upper right')

# ── P&L distribution for this run ────────────────────────────────────────────
bins = np.linspace(min(pnl01.min(), pnl02.min()) - 50,
                   max(pnl01.max(), pnl02.max()) + 50, 15)
ax_dist.hist(pnl01, bins=bins, alpha=0.55, color=C1,
             label='MMM01', edgecolor=DARK)
ax_dist.hist(pnl02, bins=bins, alpha=0.55, color=C2,
             label='Full MMM02', edgecolor=DARK)
ax_dist.axvline(pnl01.mean(), color=C1, lw=2, ls='--',
                label=f'MMM01 mean = {pnl01.mean():.0f}¢')
ax_dist.axvline(pnl02.mean(), color=C2, lw=2, ls='--',
                label=f'MMM02 mean = {pnl02.mean():.0f}¢')

from scipy import stats
t_stat, p_val   = stats.ttest_rel(pnl01, pnl02)
w_stat, p_val_w = stats.wilcoxon(pnl01, pnl02)
cohens_d = (pnl02.mean() - pnl01.mean()) / (np.std(pnl02 - pnl01) + 1e-8)

ax_dist.set_title(f'Full MMM02 vs MMM01 — P&L Distribution (N={N_TRIALS})\n'
                  f't-test p={p_val:.3f}  Wilcoxon p={p_val_w:.3f}  '
                  f"Cohen's d={cohens_d:.3f}",
                  fontsize=10, pad=6)
ax_dist.set_xlabel('Final P&L (¢)', fontsize=9)
ax_dist.set_ylabel('Frequency', fontsize=9)
ax_dist.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT,
               framealpha=0.9, edgecolor=GRID)

fig.suptitle('XGBoost Feature Importance Analysis — Full MMM02',
             fontsize=13, fontweight='bold', color=TEXT, y=1.01)

out = 'mmm_feature_importance.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK)
print(f"\nSaved → {out}")
plt.close()

# ── save importances to CSV ───────────────────────────────────────────────────
with open('mmm_feature_importances.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Feature', 'Mean Importance', 'Std', 'Rank'])
    for rank, (i, m, s) in enumerate(zip(sort_idx, sorted_mean, sorted_std), 1):
        writer.writerow([FEATURE_NAMES[i].replace('\n', ' '), f'{m:.4f}', f'{s:.4f}', rank])

print("Saved → mmm_feature_importances.csv")
print("\nDone. Files produced:")
print("  mmm_feature_importance.png      — figure for paper")
print("  mmm_feature_importances.csv     — table data for paper")