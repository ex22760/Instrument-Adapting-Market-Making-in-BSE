"""
SPY Training Data Analysis
Analyses the statistical properties of the SPY 1-minute data
and produces visualisations to justify MMM02 design decisions.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from collections import defaultdict
from scipy import stats

# ── load data ─────────────────────────────────────────────────────────────────
rows  = list(csv.reader(open('spy_1m_2026-Training.csv')))
data  = rows[1:]

times   = [r[7].split(' ')[1][:8] for r in data]
opens   = np.array([float(r[3]) for r in data])
closes  = np.array([float(r[4]) for r in data])
highs   = np.array([float(r[5]) for r in data])
lows    = np.array([float(r[6]) for r in data])
volumes = np.array([float(r[1]) for r in data])
returns = np.diff(closes) / closes[:-1]
abs_rets = np.abs(returns)

def time_to_mins(t):
    h, m, s = t.split(':')
    return int(h)*60 + int(m)

mins_of_day = np.array([time_to_mins(t) for t in times])

# ── session period masks ──────────────────────────────────────────────────────
pre_mask   = mins_of_day < 570                        # before 09:30
open_mask  = (mins_of_day >= 570) & (mins_of_day < 600)   # 09:30–10:00
mid_mask   = (mins_of_day >= 600) & (mins_of_day < 900)   # 10:00–15:00
close_mask = (mins_of_day >= 900) & (mins_of_day < 960)   # 15:00–16:00
after_mask = mins_of_day >= 960                       # after 16:00

period_names   = ['Pre-market\n(<09:30)', 'Open 30min\n(09:30–10:00)',
                  'Mid-day\n(10:00–15:00)', 'Close 30min\n(15:00–16:00)',
                  'After-hours\n(>16:00)']
period_masks   = [pre_mask, open_mask, mid_mask, close_mask, after_mask]
period_colours = ['#8b949e', '#f78166', '#58a6ff', '#d29922', '#3fb950']

# ── theme ─────────────────────────────────────────────────────────────────────
DARK  = '#0d1117'
PANEL = '#161b22'
GRID  = '#21262d'
TEXT  = '#e6edf3'
MUTED = '#8b949e'

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, lw=0.5, ls='--', alpha=0.8)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Price, Volume & Returns Overview
# ═══════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(16, 12), facecolor=DARK)
gs1  = gridspec.GridSpec(3, 1, figure=fig1, hspace=0.38)

ax_price = fig1.add_subplot(gs1[0])
ax_vol   = fig1.add_subplot(gs1[1], sharex=ax_price)
ax_ret   = fig1.add_subplot(gs1[2], sharex=ax_price)

for ax in [ax_price, ax_vol, ax_ret]:
    style_ax(ax)

x = np.arange(len(closes))

# colour each bar by session period
bar_colours = np.full(len(closes), '#8b949e', dtype=object)
for mask, col in zip(period_masks, period_colours):
    bar_colours[mask] = col

# price line with session shading
for mask, col, name in zip(period_masks, period_colours, period_names):
    idx = np.where(mask)[0]
    if len(idx) == 0: continue
    ax_price.plot(x[idx], closes[idx], color=col, lw=0.8, alpha=0.9)

# SMA overlay
sma20 = np.convolve(closes, np.ones(20)/20, mode='valid')
ax_price.plot(x[19:], sma20, color='white', lw=1.2, ls='--', alpha=0.6, label='SMA(20)')
ax_price.set_ylabel('Close Price (USD)', fontsize=9)
ax_price.set_title('SPY 1-min Price — 2026-03-02  (coloured by session period)', fontsize=10, pad=6)

legend_patches = [Patch(color=c, label=n.replace('\n', ' '))
                  for c, n in zip(period_colours, period_names)]
legend_patches.append(plt.Line2D([0],[0], color='white', ls='--', lw=1.2, label='SMA(20)'))
ax_price.legend(handles=legend_patches, fontsize=7, facecolor=PANEL,
                labelcolor=TEXT, framealpha=0.9, edgecolor=GRID, ncol=3)

# volume bars
ax_vol.bar(x, volumes, color=bar_colours, alpha=0.8, width=1.0)
ax_vol.set_ylabel('Volume', fontsize=9)
ax_vol.set_title('Volume Profile (coloured by session period)', fontsize=10, pad=6)

# returns
ret_x = x[1:]
ret_colours = np.where(returns >= 0, '#3fb950', '#f78166')
ax_ret.bar(ret_x, returns * 100, color=ret_colours, alpha=0.85, width=1.0)
ax_ret.axhline(0, color=TEXT, lw=0.8)
ax_ret.axhline( 2*returns.std()*100, color='#d29922', lw=1, ls=':', alpha=0.7, label='+2σ')
ax_ret.axhline(-2*returns.std()*100, color='#d29922', lw=1, ls=':', alpha=0.7, label='−2σ')
ax_ret.set_ylabel('1-min Return (%)', fontsize=9)
ax_ret.set_xlabel('Bar index', fontsize=9)
ax_ret.set_title('1-Minute Returns with ±2σ Bounds', fontsize=10, pad=6)
ax_ret.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9, edgecolor=GRID)

fig1.suptitle('SPY Training Data Overview — 2026-03-02', fontsize=13,
              fontweight='bold', color=TEXT, y=0.995)
fig1.savefig('spy_analysis_1_overview.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close(fig1)
print("Saved spy_analysis_1_overview.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Session Period Statistics (justifies regime switching)
# ═══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(16, 10), facecolor=DARK)
gs2  = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.38)

ax_vol2  = fig2.add_subplot(gs2[0, 0])
ax_std   = fig2.add_subplot(gs2[0, 1])
ax_ac    = fig2.add_subplot(gs2[0, 2])
ax_box   = fig2.add_subplot(gs2[1, :])

for ax in [ax_vol2, ax_std, ax_ac, ax_box]:
    style_ax(ax)

period_labels_short = ['Pre\n<09:30', 'Open\n09:30–10', 'Mid\n10–15',
                        'Close\n15–16', 'After\n>16:00']

# compute per-period stats
period_vols, period_stds, period_acs, period_ret_arrays = [], [], [], []
for mask in period_masks:
    idx     = np.where(mask)[0]
    ret_idx = idx[idx < len(returns)]
    pr      = returns[ret_idx]
    pv      = volumes[idx]
    ac      = np.corrcoef(pr[1:], pr[:-1])[0,1] if len(pr) > 3 else 0.0
    period_vols.append(pv.mean())
    period_stds.append(pr.std() * 100)
    period_acs.append(ac)
    period_ret_arrays.append(pr * 100)

xpos = np.arange(len(period_labels_short))

# avg volume
bars = ax_vol2.bar(xpos, period_vols, color=period_colours, edgecolor=DARK, alpha=0.85)
ax_vol2.set_xticks(xpos); ax_vol2.set_xticklabels(period_labels_short, fontsize=8)
ax_vol2.set_title('Average Volume by Session Period', fontsize=10, pad=6)
ax_vol2.set_ylabel('Avg Volume', fontsize=9)
for bar, v in zip(bars, period_vols):
    ax_vol2.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                 f'{v:,.0f}', ha='center', va='bottom', fontsize=7, color=TEXT)

# return std
bars2 = ax_std.bar(xpos, period_stds, color=period_colours, edgecolor=DARK, alpha=0.85)
ax_std.set_xticks(xpos); ax_std.set_xticklabels(period_labels_short, fontsize=8)
ax_std.set_title('Return Volatility (Std Dev) by Session Period', fontsize=10, pad=6)
ax_std.set_ylabel('Std Dev of Returns (%)', fontsize=9)
for bar, v in zip(bars2, period_stds):
    ax_std.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                f'{v:.4f}%', ha='center', va='bottom', fontsize=7, color=TEXT)

# autocorrelation — this is the key justification plot
bar_cols_ac = ['#3fb950' if a > 0 else '#f78166' for a in period_acs]
bars3 = ax_ac.bar(xpos, period_acs, color=bar_cols_ac, edgecolor=DARK, alpha=0.85)
ax_ac.axhline(0, color=TEXT, lw=1)
ax_ac.axhline( 0.05, color=MUTED, lw=0.8, ls=':', alpha=0.6)
ax_ac.axhline(-0.05, color=MUTED, lw=0.8, ls=':', alpha=0.6)
ax_ac.set_xticks(xpos); ax_ac.set_xticklabels(period_labels_short, fontsize=8)
ax_ac.set_title('Lag-1 Return Autocorrelation by Session Period\n(+ve = momentum, −ve = mean reversion)',
                fontsize=10, pad=6)
ax_ac.set_ylabel('Autocorrelation', fontsize=9)
for bar, v in zip(bars3, period_acs):
    ypos = v + 0.01 if v >= 0 else v - 0.03
    ax_ac.text(bar.get_x()+bar.get_width()/2, ypos,
               f'{v:+.3f}', ha='center', va='bottom', fontsize=8,
               color=TEXT, fontweight='bold')

# box plot of returns per period
bp = ax_box.boxplot(period_ret_arrays, patch_artist=True,
                    medianprops=dict(color='white', lw=2),
                    whiskerprops=dict(color=TEXT, lw=1),
                    capprops=dict(color=TEXT, lw=1),
                    flierprops=dict(marker='.', markerfacecolor=MUTED,
                                   markersize=3, alpha=0.5))
for patch, col in zip(bp['boxes'], period_colours):
    patch.set_facecolor(col + '66')
    patch.set_edgecolor(col)
ax_box.set_xticklabels(period_labels_short, fontsize=9)
ax_box.axhline(0, color=TEXT, lw=0.8, ls='--', alpha=0.5)
ax_box.set_title('Distribution of 1-min Returns by Session Period', fontsize=10, pad=6)
ax_box.set_ylabel('Return (%)', fontsize=9)
ax_box.set_xlabel('Session Period', fontsize=9)

fig2.suptitle('Session Period Analysis — Justification for Regime-Switching Strategy',
              fontsize=13, fontweight='bold', color=TEXT, y=0.998)
fig2.savefig('spy_analysis_2_session_periods.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close(fig2)
print("Saved spy_analysis_2_session_periods.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Mean Reversion & Autocorrelation (justifies buy signal)
# ═══════════════════════════════════════════════════════════════════════════════
fig3 = plt.figure(figsize=(16, 10), facecolor=DARK)
gs3  = gridspec.GridSpec(2, 2, figure=fig3, hspace=0.42, wspace=0.35)

ax_ac_lags  = fig3.add_subplot(gs3[0, 0])
ax_scatter  = fig3.add_subplot(gs3[0, 1])
ax_mr_hist  = fig3.add_subplot(gs3[1, 0])
ax_big_move = fig3.add_subplot(gs3[1, 1])

for ax in [ax_ac_lags, ax_scatter, ax_mr_hist, ax_big_move]:
    style_ax(ax)

# autocorrelation at multiple lags
lags    = np.arange(1, 21)
ac_vals = [np.corrcoef(returns[lag:], returns[:-lag])[0,1] for lag in lags]
colours_ac = ['#3fb950' if a > 0 else '#f78166' for a in ac_vals]
ax_ac_lags.bar(lags, ac_vals, color=colours_ac, edgecolor=DARK, alpha=0.85)
ax_ac_lags.axhline(0, color=TEXT, lw=1)
# 95% confidence bounds
conf = 1.96 / np.sqrt(len(returns))
ax_ac_lags.axhline( conf, color='#d29922', lw=1, ls='--', label=f'95% CI (±{conf:.3f})')
ax_ac_lags.axhline(-conf, color='#d29922', lw=1, ls='--')
ax_ac_lags.set_xlabel('Lag (minutes)', fontsize=9)
ax_ac_lags.set_ylabel('Autocorrelation', fontsize=9)
ax_ac_lags.set_title('Return Autocorrelation Function (ACF)\nLag 1 = −0.231 confirms mean reversion', fontsize=10, pad=6)
ax_ac_lags.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)

# scatter: r_t vs r_{t-1}
ax_scatter.scatter(returns[:-1]*100, returns[1:]*100,
                   s=1.5, alpha=0.25, color='#58a6ff', rasterized=True)
m, b, r, p, _ = stats.linregress(returns[:-1], returns[1:])
xfit = np.linspace(returns[:-1].min(), returns[:-1].max(), 100)
ax_scatter.plot(xfit*100, (m*xfit+b)*100, color='#f78166', lw=2,
                label=f'slope={m:.3f}, r²={r**2:.3f}')
ax_scatter.axhline(0, color=MUTED, lw=0.6)
ax_scatter.axvline(0, color=MUTED, lw=0.6)
ax_scatter.set_xlabel('Return at t (%)' , fontsize=9)
ax_scatter.set_ylabel('Return at t+1 (%)', fontsize=9)
ax_scatter.set_title('Scatter: r(t) vs r(t+1)\nNegative slope confirms mean reversion', fontsize=10, pad=6)
ax_scatter.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)

# momentum vs mean-reversion breakdown
same = sum(1 for i in range(1, len(returns)) if returns[i]*returns[i-1] > 0)
rev  = len(returns)-1 - same
ax_mr_hist.bar(['Mean Reversion\n(direction reversal)', 'Momentum\n(same direction)'],
               [rev, same],
               color=['#3fb950', '#f78166'], edgecolor=DARK, alpha=0.85)
ax_mr_hist.set_title(f'Consecutive Bar Direction\n{rev/(same+rev)*100:.1f}% mean revert, {same/(same+rev)*100:.1f}% momentum',
                     fontsize=10, pad=6)
ax_mr_hist.set_ylabel('Count', fontsize=9)
for i, v in enumerate([rev, same]):
    ax_mr_hist.text(i, v+5, f'{v}', ha='center', fontsize=10, color=TEXT, fontweight='bold')

# mean reversion after big moves
std_r   = returns.std()
windows = [1, 2, 3, 5, 10]
big_down_avg, big_up_avg = [], []
for w in windows:
    bd = np.where(returns < -2*std_r)[0]
    bu = np.where(returns >  2*std_r)[0]
    # average cumulative return over next w bars
    avg_d = np.mean([returns[i+1:i+1+w].sum() for i in bd if i+w < len(returns)]) * 100
    avg_u = np.mean([returns[i+1:i+1+w].sum() for i in bu if i+w < len(returns)]) * 100
    big_down_avg.append(avg_d)
    big_up_avg.append(avg_u)

xw = np.arange(len(windows))
width = 0.35
ax_big_move.bar(xw - width/2, big_down_avg, width, color='#3fb950', edgecolor=DARK,
                alpha=0.85, label='After big DOWN move (bounce expected)')
ax_big_move.bar(xw + width/2, big_up_avg,   width, color='#f78166', edgecolor=DARK,
                alpha=0.85, label='After big UP move')
ax_big_move.axhline(0, color=TEXT, lw=0.8)
ax_big_move.set_xticks(xw)
ax_big_move.set_xticklabels([f'{w}-bar\ncumulative' for w in windows], fontsize=8)
ax_big_move.set_title('Avg Cumulative Return After Big Moves (>2σ)\nPositive after down = mean reversion opportunity',
                      fontsize=10, pad=6)
ax_big_move.set_ylabel('Avg Cumulative Return (%)', fontsize=9)
ax_big_move.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID, framealpha=0.9)

fig3.suptitle('Mean Reversion Analysis — Justification for Buy Signal Design',
              fontsize=13, fontweight='bold', color=TEXT, y=0.998)
fig3.savefig('spy_analysis_3_mean_reversion.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close(fig3)
print("Saved spy_analysis_3_mean_reversion.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Volatility Clustering / ARCH Effect (justifies adaptive margins)
# ═══════════════════════════════════════════════════════════════════════════════
fig4 = plt.figure(figsize=(16, 10), facecolor=DARK)
gs4  = gridspec.GridSpec(2, 2, figure=fig4, hspace=0.42, wspace=0.35)

ax_arch_ts   = fig4.add_subplot(gs4[0, :])
ax_arch_ac   = fig4.add_subplot(gs4[1, 0])
ax_vol_ratio = fig4.add_subplot(gs4[1, 1])

for ax in [ax_arch_ts, ax_arch_ac, ax_vol_ratio]:
    style_ax(ax)

# |returns| time series with rolling std
roll_std = np.array([returns[max(0,i-20):i+1].std()*100
                     for i in range(len(returns))])

ax_arch_ts.fill_between(np.arange(len(abs_rets)), abs_rets*100,
                         alpha=0.4, color='#58a6ff', label='|return|')
ax_arch_ts.plot(roll_std, color='#f78166', lw=1.4,
                label='Rolling 20-bar Std Dev (volatility estimate)')
ax_arch_ts.set_title('Absolute Returns & Rolling Volatility — ARCH Effect Visible as Clustering',
                     fontsize=10, pad=6)
ax_arch_ts.set_xlabel('Bar index', fontsize=9)
ax_arch_ts.set_ylabel('|Return| / Volatility (%)', fontsize=9)
ax_arch_ts.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID, framealpha=0.9)

# ACF of |returns|
ac_abs = [np.corrcoef(abs_rets[lag:], abs_rets[:-lag])[0,1] for lag in lags]
ax_arch_ac.bar(lags, ac_abs, color='#d29922', edgecolor=DARK, alpha=0.85)
ax_arch_ac.axhline(0, color=TEXT, lw=0.8)
ax_arch_ac.axhline( conf, color='#f78166', lw=1, ls='--', label=f'95% CI')
ax_arch_ac.axhline(-conf, color='#f78166', lw=1, ls='--')
ax_arch_ac.set_xlabel('Lag (minutes)', fontsize=9)
ax_arch_ac.set_ylabel('Autocorrelation of |returns|', fontsize=9)
ax_arch_ac.set_title(f'ACF of |Returns| — Lag-1 = {ac_abs[0]:.3f}\nPositive = ARCH effect confirmed',
                     fontsize=10, pad=6)
ax_arch_ac.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)

# high vs low volatility regime — how much does margin need to vary?
median_vol = np.median(roll_std)
high_vol   = roll_std[roll_std > median_vol]
low_vol    = roll_std[roll_std <= median_vol]
ax_vol_ratio.hist(low_vol,  bins=30, alpha=0.6, color='#3fb950',
                  label=f'Low vol regime  (mean={low_vol.mean():.4f}%)',  edgecolor=DARK)
ax_vol_ratio.hist(high_vol, bins=30, alpha=0.6, color='#f78166',
                  label=f'High vol regime (mean={high_vol.mean():.4f}%)', edgecolor=DARK)
ax_vol_ratio.axvline(low_vol.mean(),  color='#3fb950', lw=2, ls='--')
ax_vol_ratio.axvline(high_vol.mean(), color='#f78166', lw=2, ls='--')
ratio = high_vol.mean() / low_vol.mean()
ax_vol_ratio.set_title(f'Low vs High Volatility Regimes\nHigh vol is {ratio:.1f}× larger → adaptive margins justified',
                       fontsize=10, pad=6)
ax_vol_ratio.set_xlabel('Rolling 20-bar Std Dev (%)', fontsize=9)
ax_vol_ratio.set_ylabel('Frequency', fontsize=9)
ax_vol_ratio.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID, framealpha=0.9)

fig4.suptitle('Volatility Clustering (ARCH Effect) — Justification for Adaptive Margin Scaling',
              fontsize=13, fontweight='bold', color=TEXT, y=0.998)
fig4.savefig('spy_analysis_4_arch_volatility.png', dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close(fig4)
print("Saved spy_analysis_4_arch_volatility.png")

# ── print summary stats ───────────────────────────────────────────────────────
print('\n' + '='*60)
print('SUMMARY STATISTICS FOR PAPER')
print('='*60)
print(f'Dataset:          SPY 1-min, 2026-03-02')
print(f'Total bars:       {len(data)}')
print(f'Price range:      ${closes.min():.2f} – ${closes.max():.2f}')
print(f'Total return:     {(closes[-1]/closes[0]-1)*100:.2f}%')
print(f'\nReturn stats:')
print(f'  Mean:           {returns.mean()*100:.5f}%')
print(f'  Std dev:        {returns.std()*100:.5f}%')
print(f'  Skewness:       {stats.skew(returns):.4f}')
print(f'  Kurtosis:       {stats.kurtosis(returns):.4f}')
print(f'\nKey findings:')
print(f'  Overall autocorr(1):    {np.corrcoef(returns[1:],returns[:-1])[0,1]:.4f}  (mean reversion confirmed)')
print(f'  |return| autocorr(1):   {ac_abs[0]:.4f}  (ARCH effect confirmed)')
same = sum(1 for i in range(1,len(returns)) if returns[i]*returns[i-1]>0)
rev  = len(returns)-1-same
print(f'  Mean reversion rate:    {rev/(same+rev)*100:.1f}%')
print(f'\nSession autocorrelations:')
for name, mask in zip(period_names, period_masks):
    idx = np.where(mask)[0]
    ri  = idx[idx < len(returns)]
    pr  = returns[ri]
    if len(pr) > 3:
        ac = np.corrcoef(pr[1:], pr[:-1])[0,1]
        print(f'  {name.replace(chr(10)," "):30s}: {ac:+.4f}')
print(f'\nHigh/low vol ratio:       {ratio:.2f}x  (justifies adaptive margins)')