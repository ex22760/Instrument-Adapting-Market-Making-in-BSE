"""
Run a BSE market session and plot account balance history for MMM01 vs MMM02.
"""

import sys, os, math, random, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import importlib.util

# ── import BSE as a module (not __main__) ────────────────────────────────────
spec = importlib.util.spec_from_file_location("BSE_Coursework", "BSE_Coursework.py")
BSE = importlib.util.module_from_spec(spec)
sys.modules['BSE_Coursework'] = BSE
spec.loader.exec_module(BSE)

# ── monkeypatch: record balance after every bookkeep call ────────────────────
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

# ── preprocess SPY CSV into BSE-compatible format ────────────────────────────
# BSE's schedule_offsetfn_read_file expects:
#   - NO header row
#   - time column in plain HH:MM:SS format
#   - a numeric price column
# The raw SPY file has a header and full ISO datetimes like '2026-03-02 04:00:00-05:00'
raw_spy = 'spy_1m_2026-03-03.csv'
clean_spy = 'spy_bse_format.csv'
with open(raw_spy, newline='') as fin, open(clean_spy, 'w', newline='') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    next(reader)                          # skip the header row
    for row in reader:
        time_str   = row[7].split(' ')[1][:8]   # '2026-03-02 04:00:00-05:00' → '04:00:00'
        open_price = row[3]                      # 'open' column
        writer.writerow([time_str, open_price])
print(f"Preprocessed SPY data written to {clean_spy}")

# ── simulation setup ─────────────────────────────────────────────────────────
price_offset_filename = 'spy_bse_format.csv'

sellers_spec = [('ZIP', 3), ('SHVR', 3), ('GVWY', 3), ('SNPR', 3)]
buyers_spec  = [('ZIP', 3), ('ZIC', 3),  ('SHVR', 3), ('GVWY', 3)]
mm_params    = {'bid_percent': 0.99, 'ask_delta': 5, 'n_past_trades': 5}
mm_spec      = [('MMM01', 1, mm_params), ('MMM02', 1, mm_params)]

sup_range = (100, 200)
dem_range = (50, 150)

start_time = 0
end_time   = int(7.5 * 60 * 60)

(offsetfn, offsetfn_params) = BSE.offset_from_file(
    price_offset_filename, 0, 1, 75, end_time
)

supply_schedule = [{'from': start_time, 'to': end_time,
                    'ranges': [sup_range], 'stepmode': 'random',
                    'offsetfn': offsetfn, 'offsetfn_params': offsetfn_params}]
demand_schedule = [{'from': start_time, 'to': end_time,
                    'ranges': [dem_range], 'stepmode': 'random',
                    'offsetfn': offsetfn, 'offsetfn_params': offsetfn_params}]

traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec, 'mrktmakers': mm_spec}
order_sched  = {'sup': supply_schedule, 'dem': demand_schedule,
                'interval': 30, 'timemode': 'drip-jitter'}
dump_flags   = {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False,
                'dump_avgbals': False, 'dump_tape': True}

print("Running BSE market session ...")
BSE.market_session('mmm_comparison', start_time, end_time, traders_spec,
                   order_sched, dump_flags, False)
print("Session complete.")

keys = sorted(_balance_history.keys())
print("Balance history keys:", keys)

if len(keys) < 2:
    print("ERROR: fewer than 2 MM traders found.")
    sys.exit(1)

def extract(key):
    hist = sorted(_balance_history[key], key=lambda x: x[0])
    return ([h[0]/3600 for h in hist],
            [h[1]      for h in hist],
            [h[2]      for h in hist])

t1, b1, nw1 = extract(keys[0])
t2, b2, nw2 = extract(keys[1])
lbl1, lbl2  = 'MMM01', 'MMM02'

def metrics(times, balances, net_worths, label):
    init_bal  = 500
    n_trades  = len(times) - 1
    fin_bal   = balances[-1]
    fin_nw    = net_worths[-1]
    total_pnl = fin_bal - init_bal
    peak, max_dd = net_worths[0], 0
    for nw in net_worths:
        peak   = max(peak, nw)
        max_dd = max(max_dd, peak - nw)
    ppt = total_pnl / max(n_trades, 1)
    return dict(label=label, n_trades=n_trades, fin_bal=fin_bal,
                fin_nw=fin_nw, total_pnl=total_pnl, ppt=ppt, max_dd=max_dd)

m1 = metrics(t1, b1, nw1, lbl1)
m2 = metrics(t2, b2, nw2, lbl2)
for m in [m1, m2]:
    print(m)

# read tape
tx_times, tx_prices = [], []
if os.path.exists('mmm_comparison_tape.csv'):
    with open('mmm_comparison_tape.csv') as f:
        for line in f:
            parts = line.strip().split(',')
            try:
                tx_times.append(float(parts[1]) / 3600)
                tx_prices.append(float(parts[2]))
            except:
                pass

# ── plotting ─────────────────────────────────────────────────────────────────
DARK  = '#0d1117'
PANEL = '#161b22'
GRID  = '#21262d'
TEXT  = '#e6edf3'
C1    = '#58a6ff'
C2    = '#f78166'
GOLD  = '#d29922'
TEAL  = '#3fb950'

fig = plt.figure(figsize=(14, 11), facecolor=DARK)
gs  = gridspec.GridSpec(3, 1, figure=fig,
                        height_ratios=[1.6, 2.2, 1.2], hspace=0.45)

ax_tx  = fig.add_subplot(gs[0])
ax_bal = fig.add_subplot(gs[1])
ax_tbl = fig.add_subplot(gs[2])

for ax in [ax_tx, ax_bal]:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, lw=0.6, ls='--', alpha=0.8)

# transaction prices
if tx_times:
    ax_tx.scatter(tx_times, tx_prices, s=1.2, alpha=0.3, color=TEAL,
                  label='Transaction price', rasterized=True, zorder=2)
    if len(tx_prices) >= 100:
        sma = np.convolve(tx_prices, np.ones(100)/100, mode='valid')
        ax_tx.plot(tx_times[99:], sma, color=GOLD, lw=1.4, label='SMA(100)', zorder=3)
ax_tx.set_title('Transaction Prices  (SPY 1-min 2026 data as price offset)', fontsize=10, pad=6)
ax_tx.set_xlabel('Time (hours)', fontsize=9)
ax_tx.set_ylabel('Price (cents)', fontsize=9)
ax_tx.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.8, edgecolor=GRID)

# balance
ax_bal.plot(t1, b1,  color=C1, lw=2.0, label=f'{lbl1} cash balance', zorder=4)
ax_bal.plot(t2, b2,  color=C2, lw=2.0, label=f'{lbl2} cash balance', zorder=4)
ax_bal.plot(t1, nw1, color=C1, lw=0.9, ls='--', alpha=0.5, label=f'{lbl1} net worth', zorder=3)
ax_bal.plot(t2, nw2, color=C2, lw=0.9, ls='--', alpha=0.5, label=f'{lbl2} net worth', zorder=3)
ax_bal.axhline(500, color='#8b949e', lw=1.2, ls=':', zorder=2, label='Starting balance (500¢)')
ax_bal.set_title('Accumulated Balance & Net Worth — MMM01 vs MMM02', fontsize=10, pad=6)
ax_bal.set_xlabel('Time (hours)', fontsize=9)
ax_bal.set_ylabel('Value (cents)', fontsize=9)
ax_bal.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.8,
              edgecolor=GRID, ncol=2)

# table
ax_tbl.set_facecolor(PANEL)
ax_tbl.axis('off')

col_labels = ['Metric', lbl1, lbl2]
rows = [
    ['Trades executed',      f"{m1['n_trades']}",          f"{m2['n_trades']}"],
    ['Final cash balance (¢)', f"{m1['fin_bal']:.0f}",    f"{m2['fin_bal']:.0f}"],
    ['Final net worth (¢)',  f"{m1['fin_nw']:.0f}",        f"{m2['fin_nw']:.0f}"],
    ['Total P&L (¢)',        f"{m1['total_pnl']:+.0f}",    f"{m2['total_pnl']:+.0f}"],
    ['Avg profit / trade (¢)', f"{m1['ppt']:+.2f}",        f"{m2['ppt']:+.2f}"],
    ['Max drawdown (¢)',     f"{m1['max_dd']:.2f}",        f"{m2['max_dd']:.2f}"],
]

tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                   cellLoc='center', loc='center', bbox=[0.0, 0.0, 1.0, 1.0])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(GRID)
    if r == 0:
        cell.set_facecolor('#1f2937')
        cell.set_text_props(color='#ffffff', fontweight='bold')
    else:
        cell.set_facecolor(PANEL)
        if c == 0:
            cell.set_text_props(color=TEXT)
        elif c == 1:
            cell.set_text_props(color=C1, fontweight='bold')
        else:
            cell.set_text_props(color=C2, fontweight='bold')

fig.suptitle('MMM01 vs MMM02 — Single Session Performance Comparison',
             fontsize=13, fontweight='bold', color=TEXT, y=0.995)

out = 'mmm_comparison_test.png'
os.makedirs(os.path.dirname(out), exist_ok=True) if os.path.dirname(out) else None
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK)
print(f"\nSaved → {out}")
plt.close()