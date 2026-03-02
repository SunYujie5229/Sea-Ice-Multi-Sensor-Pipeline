import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# 1. 加载并转换为百分比
df = pd.read_csv(r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\20160114\statistic\monthly_stats_ALL_YEARS.csv")


# 转换百分比（如果原始数据不是0-100则执行）
for col in ['CS2_Floe_Density', 'S1_Floe_Density', 'S1_Floeref_desity', 
            'CS2_Lead_Density', 'S1_Lead_Density', 'CS2_Ambi_Density']:
    if df[col].max() <= 1.0:
        df[col] *= 100

month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
               7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
df['month_name'] = df['month'].map(month_names)

# ==========================================
# 2. 差异计算 (引入 Ambiguity 逻辑)
# ==========================================
df['Ice_Diff_Total_Upper'] = df['S1_Floeref_desity'] - df['CS2_Floe_Density']
df['Ice_Diff_Total_Lower'] = df['S1_Floeref_desity'] - (df['CS2_Floe_Density'] + df['CS2_Ambi_Density'])
df['Lead_Diff_Upper'] = df['S1_Lead_Density'] - df['CS2_Lead_Density']
df['Lead_Diff_Lower'] = df['S1_Lead_Density'] - (df['CS2_Lead_Density'] + df['CS2_Ambi_Density'])
df['Ice_Diff_Single'] = df['S1_Floe_Density'] - df['CS2_Floe_Density']

# ==========================================
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False
})

C = {
    'S1_Floe': '#08519C', 'S1_Refrozen': '#6BAED6', 'S1_Lead': '#A50F15',
    'CS2_Floe': '#9ECAE1', 'CS2_Ambi': '#D9D9D9', 'CS2_Lead': '#FB6A4A',
    'Melt_Season': '#F4F4F4', # 统一浅灰色背景
    'Line_Total': '#084594', 'Line_Lead': '#99000D', 'Line_Single': '#238B45'
}

# --- 修改 1: 使用 sharex=True 和极小的 hspace ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 11), sharex=True, 
                               gridspec_kw={'height_ratios': [1.6, 1.4], 'hspace': 0})
x = np.arange(len(df))
width = 0.38 

# --- 修改 2: 融化季背景垂直对齐并连续 ---
for ax in [ax1, ax2]:
    ax.axvspan(4.5, 7.5, color=C['Melt_Season'], zorder=0)

# ==========================================
# 3. 上图：成分占比图 (共用X轴)
# ==========================================
ax1.bar(x - width/2 - 0.01, df['S1_Floe_Density'], width, label='S1 Floe', color=C['S1_Floe'], edgecolor='none')
ax1.bar(x - width/2 - 0.01, df['S1_Floeref_desity'] - df['S1_Floe_Density'], width, bottom=df['S1_Floe_Density'], label='S1 Refrozen', color=C['S1_Refrozen'], edgecolor='none')
ax1.bar(x - width/2 - 0.01, df['S1_Lead_Density'], width, bottom=df['S1_Floeref_desity'], label='S1 Lead', color=C['S1_Lead'], edgecolor='none')

ax1.bar(x + width/2 + 0.01, df['CS2_Floe_Density'], width, label='CS2 Floe', color=C['CS2_Floe'], edgecolor='none')
ax1.bar(x + width/2 + 0.01, df['CS2_Ambi_Density'], width, bottom=df['CS2_Floe_Density'], label='CS2 Ambiguous', color=C['CS2_Ambi'], edgecolor='none')
ax1.bar(x + width/2 + 0.01, df['CS2_Lead_Density'], width, bottom=df['CS2_Floe_Density'] + df['CS2_Ambi_Density'], label='CS2 Lead', color=C['CS2_Lead'], edgecolor='none')

ax1.set_ylabel('Composition (%)', fontweight='bold')
ax1.set_ylim(0, 105)
ax1.grid(axis='y', linestyle='-', alpha=0.1)
for spine in ['top', 'right']: ax1.spines[spine].set_visible(False)

# 上图只显示 Melt Season 文字，不重复显示月份
ax1.text(6, 102, 'Melt Season', ha='center', va='bottom', fontsize=11, color='dimgray', fontweight='bold')
ax1.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.12), frameon=False, fontsize=10)

# ==========================================
# 4. 下图：差异对比图 (承接上图刻度)
# ==========================================
ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5, linestyle='--')

p1, = ax2.plot(x, df['Ice_Diff_Total_Upper'], marker='o', markersize=6, color=C['Line_Total'], linewidth=2, label=r'Total Ice Diff ($S1_{\mathrm{Total}} - CS2_{\mathrm{Floe}}$)')
ax2.fill_between(x, df['Ice_Diff_Total_Lower'], df['Ice_Diff_Total_Upper'], color=C['Line_Total'], alpha=0.08, label='CS2 Ambiguity Range (Ice)')

p2, = ax2.plot(x, df['Lead_Diff_Upper'], marker='s', markersize=6, color=C['Line_Lead'], linewidth=2, label=r'Lead Diff ($S1_{\mathrm{Lead}} - CS2_{\mathrm{Lead}}$)')
ax2.fill_between(x, df['Lead_Diff_Lower'], df['Lead_Diff_Upper'], color=C['Line_Lead'], alpha=0.08, label='CS2 Ambiguity Range (Lead)')

p3, = ax2.plot(x, df['Ice_Diff_Single'], marker='D', markersize=6, color=C['Line_Single'], linestyle='-', linewidth=1.5, label=r'Single Ice Diff ($S1_{\mathrm{Floe}} - CS2_{\mathrm{Floe}}$)')

ax2.set_ylabel('Difference (%)', fontweight='bold')
ax2.set_xlabel('Month', fontweight='bold', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(df['month_name'], fontweight='bold', fontsize=12)
ax2.set_ylim(-65, 65)

# --- 组织下图图例 ---
melt_patch = Patch(color=C['Melt_Season'], label='Melt Season')
h2, l2 = ax2.get_legend_handles_labels()
h2.append(melt_patch)
l2.append('Melt Season')
order = [0, 2, 4, 1, 3, 5]
ax2.legend([h2[i] for i in order], [l2[i] for i in order], ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=10)

ax2.grid(axis='both', linestyle='-', alpha=0.1)
for spine in ['top', 'right']: ax2.spines[spine].set_visible(False)

# ==========================================

plt.savefig(r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\20160114\statistic\refined_comparison_diffs.png", dpi=600, bbox_inches='tight')
plt.show()
