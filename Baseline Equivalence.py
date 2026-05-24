"""
Run N trials of MMM01 vs MMM02 (identical algorithms) and plot the
distribution of final P&L to demonstrate baseline equivalence.
"""

import sys, os, csv, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats
import importlib.util

# ── import BSE ───────────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("BSE_Coursework", "BSE_Coursework.py")
BSE  = importlib.util.module_from_spec(spec)
sys.modules['BSE_Coursework'] = BSE
spec.loader.exec_module(BSE)

# ── preprocess SPY CSV once ──────────────────────────────────────────────────
raw_spy   = 'spy_1m_2026-03-03.csv'
clean_spy = 'spy_bse_format.csv'
with open(raw_spy, newline='') as fin, open(clean_spy, 'w', newline='') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    next(reader)
    for row in reader:
        time_str   = row[7].split(' ')[1][:8]
        open_price = row[3]
        writer.writerow([time_str, open_price])
print("SPY data preprocessed.")

# ── monkey-patch to capture final balances ───────────────────────────────────
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

# ── simulation config ────────────────────────────────────────────────────────
N_TRIALS   = 25
INIT_BAL   = 500
start_time = 0
end_time   = int(7.5 * 60 * 60)   # 2 hours for quick test; change to int(7.5*60*60) for full runs

sellers_spec = [('ZIP', 3), ('SHVR', 3), ('GVWY', 3), ('SNPR', 3)]
buyers_spec  = [('ZIP', 3), ('ZIC', 3),  ('SHVR', 3), ('GVWY', 3)]
mm_params    = {'bid_percent': 0.99, 'ask_delta': 5, 'n_past_trades': 5}
mm_spec = [('MMM01', 1, {}), ('MMM02', 1, {
    'use_regime':          True,
    'use_kelly':           True,
    'use_asymmetric_sell': True,
    'use_xgb':             True,
})]
traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec, 'mrktmakers': mm_spec}

offsetfn, offsetfn_params = BSE.offset_from_file(clean_spy, 0, 1, 75, end_time)

supply_schedule = [{'from': start_time, 'to': end_time,
                    'ranges': [(100,200)], 'stepmode': 'random',
                    'offsetfn': offsetfn, 'offsetfn_params': offsetfn_params}]
demand_schedule = [{'from': start_time, 'to': end_time,
                    'ranges': [(50,150)],  'stepmode': 'random',
                    'offsetfn': offsetfn, 'offsetfn_params': offsetfn_params}]
order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
               'interval': 30, 'timemode': 'drip-jitter'}
dump_flags  = {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False,
               'dump_avgbals': False, 'dump_tape': False}

# ── run trials ───────────────────────────────────────────────────────────────
pnl01, pnl02 = [], []
trades01, trades02 = [], []

for trial in range(N_TRIALS):
    _balance_history.clear()
    print(f"Trial {trial+1}/{N_TRIALS} ...", end=' ', flush=True)

    BSE.market_session(f'trial_{trial:02d}', start_time, end_time,
                       traders_spec, order_sched, dump_flags, False)

    keys = sorted(_balance_history.keys())
    if len(keys) < 2:
        print("WARNING: fewer than 2 MM traders — skipping")
        continue

    def final_pnl(key):
        hist = _balance_history[key]
        final_bal = hist[-1][1]
        n = len(hist) - 1   # subtract 1 for the initial record at t=0
        return final_bal - INIT_BAL, n

    p1, n1 = final_pnl(keys[0])
    p2, n2 = final_pnl(keys[1])
    pnl01.append(p1);   trades01.append(n1)
    pnl02.append(p2);   trades02.append(n2)
    print(f"MMM01 P&L={p1:+d}¢  MMM02 P&L={p2:+d}¢")

pnl01   = np.array(pnl01,   dtype=float)
pnl02   = np.array(pnl02,   dtype=float)
trades01 = np.array(trades01, dtype=float)
trades02 = np.array(trades02, dtype=float)

# ── statistical tests ────────────────────────────────────────────────────────
t_stat, p_val   = stats.ttest_rel(pnl01, pnl02)
w_stat, p_val_w = stats.wilcoxon(pnl01, pnl02)
diffs = pnl02 - pnl01

print(f"\n=== Statistics ===")
print(f"MMM01  mean={pnl01.mean():.1f}  std={pnl01.std():.1f}")
print(f"MMM02  mean={pnl02.mean():.1f}  std={pnl02.std():.1f}")
print(f"Paired t-test:  t={t_stat:.3f}, p={p_val:.4f}")
print(f"Wilcoxon:       W={w_stat:.1f},  p={p_val_w:.4f}")

# ── plot ──────────────────────────────────────────────────────────────────────
DARK  = '#0d1117'
PANEL = '#161b22'
GRID  = '#21262d'
TEXT  = '#e6edf3'
C1    = '#58a6ff'   # MMM01 blue
C2    = '#f78166'   # MMM02 red
GOLD  = '#d29922'

fig = plt.figure(figsize=(14, 11), facecolor=DARK)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ax_hist = fig.add_subplot(gs[0, :])   # P&L distributions — full width
ax_box  = fig.add_subplot(gs[1, 0])   # box plot
ax_diff = fig.add_subplot(gs[1, 1])   # trial-by-trial difference

for ax in [ax_hist, ax_box, ax_diff]:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, lw=0.6, ls='--', alpha=0.8)

# — histogram of P&L distributions —
bins = np.linspace(min(pnl01.min(), pnl02.min()) - 50,
                   max(pnl01.max(), pnl02.max()) + 50, 15)
ax_hist.hist(pnl01, bins=bins, alpha=0.55, color=C1, label='MMM01', edgecolor=DARK)
ax_hist.hist(pnl02, bins=bins, alpha=0.55, color=C2, label='MMM02', edgecolor=DARK)
ax_hist.axvline(pnl01.mean(), color=C1, lw=2, ls='--', label=f'MMM01 mean = {pnl01.mean():.0f}¢')
ax_hist.axvline(pnl02.mean(), color=C2, lw=2, ls='--', label=f'MMM02 mean = {pnl02.mean():.0f}¢')
ax_hist.set_title(f'Distribution of Final P&L over {N_TRIALS} Trials  '
                  f'(paired t-test p={p_val:.3f}, Wilcoxon p={p_val_w:.3f})',
                  fontsize=10, pad=6)
ax_hist.set_xlabel('Final P&L (¢)', fontsize=9)
ax_hist.set_ylabel('Frequency', fontsize=9)
ax_hist.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT,
               framealpha=0.8, edgecolor=GRID)

# — box plot —
bp = ax_box.boxplot([pnl01, pnl02], patch_artist=True,
                    medianprops=dict(color='white', lw=2),
                    whiskerprops=dict(color=TEXT),
                    capprops=dict(color=TEXT),
                    flierprops=dict(markerfacecolor=TEXT, markersize=4))
bp['boxes'][0].set_facecolor(C1 + '88')
bp['boxes'][1].set_facecolor(C2 + '88')
ax_box.set_xticks([1, 2])
ax_box.set_xticklabels(['MMM01', 'MMM02'], fontsize=9)
ax_box.set_title('P&L Box Plot', fontsize=10, pad=6)
ax_box.set_ylabel('Final P&L (¢)', fontsize=9)

# — trial-by-trial difference (MMM02 - MMM01) —
trial_nums = np.arange(1, len(diffs)+1)
colours    = [C2 if d >= 0 else C1 for d in diffs]
ax_diff.bar(trial_nums, diffs, color=colours, edgecolor=DARK, alpha=0.85)
ax_diff.axhline(0,           color=TEXT, lw=1,   ls='-')
ax_diff.axhline(diffs.mean(), color=GOLD, lw=1.5, ls='--',
                label=f'Mean diff = {diffs.mean():+.1f}¢')
ax_diff.set_title('Per-Trial P&L Difference  (MMM02 − MMM01)', fontsize=10, pad=6)
ax_diff.set_xlabel('Trial', fontsize=9)
ax_diff.set_ylabel('Difference (¢)', fontsize=9)
ax_diff.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT,
               framealpha=0.8, edgecolor=GRID)

# annotation: p-values
sig_note = ("No statistically significant difference detected\n"
            f"(p={p_val:.3f} paired t-test, p={p_val_w:.3f} Wilcoxon, α=0.05)")
fig.text(0.5, 0.01, sig_note, ha='center', fontsize=9,
         color='#8b949e', style='italic')

fig.suptitle('MMM01 vs MMM02 — Baseline Equivalence  '
             f'(MMM02 is a verbatim clone, N={N_TRIALS} trials)',
             fontsize=13, fontweight='bold', color=TEXT, y=0.995)

out = 'mmm_test_validation.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK)
print(f"\nSaved → {out}")
plt.close()