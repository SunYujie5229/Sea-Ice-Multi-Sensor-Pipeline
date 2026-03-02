import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 1. 加载数据
# 注意：请确保路径正确，或将文件放在脚本同级目录下
# df = pd.read_csv(r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\20160114\statistic\monthly_stats_ALL_YEARS_20250114.xlsx")
file_path = r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\20160114\statistic\monthly_stats_ALL_YEARS_20250114.xlsx"    
df = pd.read_excel(file_path, engine='openpyxl')
# 转换百分比
for col in ['CS2_Floe_Density', 'S1_Floe_Density', 'S1_Floeref_desity', 
            'CS2_Lead_Density', 'S1_Lead_Density', 'CS2_Ambi_Density']:
    if df[col].max() <= 1.0:
        df[col] *= 100

month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
               7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
df['month_name'] = df['month'].map(month_names)

# ==========================================
# 2. 差异与新指标计算 (包含 Ambiguity 逻辑)
# ==========================================
# Line 1: Lead Diff
df['Lead_Diff_Upper'] = df['S1_Lead_Density'] - df['CS2_Lead_Density']
df['Lead_Diff_Lower'] = df['Lead_Diff_Upper'] - df['CS2_Ambi_Density']

# Line 2: Ice Diff (Floe)
df['Ice_Diff_Upper'] = df['S1_Floe_Density'] - df['CS2_Floe_Density']
df['Ice_Diff_Lower'] = df['Ice_Diff_Upper'] - df['CS2_Ambi_Density']

# Line 3: S1 Refrozen (Total Ice - Floe Ice)
# 如果 CSV 中已有 'S1_refrozen_density' 列可直接使用，否则按以下逻辑计算
df['S1_Refrozen_Upper'] = df['S1_Floeref_desity'] - df['S1_Floe_Density']
df['S1_Refrozen_Lower'] = df['S1_Refrozen_Upper'] - df['CS2_Ambi_Density']


plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False
})

C = {
    'S1_Floe': '#08519C',      # 深蓝
    'S1_Refrozen': '#6BAED6',  # 中蓝
    'S1_Lead': '#A50F15',      # 深红
    'CS2_Floe': '#9ECAE1',     # 浅蓝
    'CS2_Ambi': '#D9D9D9',     # 浅灰
    'CS2_Lead': '#FB6A4A',     # 浅红
    'Melt_Season': '#F2F2F2',  # 背景阴影
    'Line_Lead': '#A50F15',    # 红线 (Lead)
    'Line_Ice': '#08519C',     # 绿线 (Ice)
    'Line_Refrozen': '#6BAED6' # 蓝线 (Refrozen)
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 12.5), gridspec_kw={'height_ratios': [1.5, 1.5]})
x = np.arange(len(df))
width = 0.38 

# 背景 Melt Season
for ax in [ax1, ax2]:
    ax.axvspan(4.5, 7.5, color=C['Melt_Season'], zorder=0)

# ==========================================
# 3. 上图：堆叠柱状图 (保持原样)
# ==========================================
ax1.bar(x - width/2 - 0.01, df['S1_Floe_Density'], width, label='S1 Floe', color=C['S1_Floe'], edgecolor='none')
ax1.bar(x - width/2 - 0.01, df['S1_Floeref_desity'] - df['S1_Floe_Density'], width, bottom=df['S1_Floe_Density'], label='S1 Refrozen', color=C['S1_Refrozen'], edgecolor='none')
ax1.bar(x - width/2 - 0.01, df['S1_Lead_Density'], width, bottom=df['S1_Floeref_desity'], label='S1 Lead', color=C['S1_Lead'], edgecolor='none')

ax1.bar(x + width/2 + 0.01, df['CS2_Floe_Density'], width, label='CS2 Floe', color=C['CS2_Floe'], edgecolor='none')
ax1.bar(x + width/2 + 0.01, df['CS2_Ambi_Density'], width, bottom=df['CS2_Floe_Density'], label='CS2 Ambiguous', color=C['CS2_Ambi'], edgecolor='none')
ax1.bar(x + width/2 + 0.01, df['CS2_Lead_Density'], width, bottom=df['CS2_Floe_Density'] + df['CS2_Ambi_Density'], label='CS2 Lead', color=C['CS2_Lead'], edgecolor='none')

ax1.text(6, 102, 'Melt Season', ha='center', va='bottom', fontsize=11, color='dimgray', fontweight='bold')
ax1.set_ylabel('Composition (%)', fontweight='bold', labelpad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(df['month_name'], fontweight='bold', fontsize=12)
ax1.set_ylim(0, 105)
ax1.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=10)
ax1.grid(axis='y', linestyle='-', alpha=0.15)
for spine in ['top', 'right']: ax1.spines[spine].set_visible(False)

# ==========================================
# 4. 下图：差异图 (更新 3 条线及阴影)
# ==========================================
ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5, linestyle='--')

# Line 1: Lead Diff + Ambi
p1, = ax2.plot(x, df['Lead_Diff_Upper'], marker='s', markersize=6, color=C['Line_Lead'], linewidth=2, label='Lead Diff (S1-CS2)')
f1 = ax2.fill_between(x, df['Lead_Diff_Lower'], df['Lead_Diff_Upper'], color=C['Line_Lead'], alpha=0.08, label='Lead Ambiguity Range')

# Line 2: Ice Diff + Ambi
p2, = ax2.plot(x, df['Ice_Diff_Upper'], marker='o', markersize=6, color=C['Line_Ice'], linewidth=2, label='Ice Diff (S1-CS2)')
f2 = ax2.fill_between(x, df['Ice_Diff_Lower'], df['Ice_Diff_Upper'], color=C['Line_Ice'], alpha=0.08, label='Ice Ambiguity Range')

# Line 3: S1 Refrozen + Ambi
p3, = ax2.plot(x, df['S1_Refrozen_Upper'], marker='D', markersize=6, color=C['Line_Refrozen'], linewidth=2, label='S1 Refrozen Density')
f3 = ax2.fill_between(x, df['S1_Refrozen_Lower'], df['S1_Refrozen_Upper'], color=C['Line_Refrozen'], alpha=0.08, label='Refrozen Ambiguity Range')

ax2.set_ylabel('Percentage / Difference (%)', fontweight='bold', labelpad=10)
ax2.set_xlabel('Month', fontweight='bold', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(df['month_name'], fontweight='bold', fontsize=12)
ax2.set_ylim(-60, 60)

# 手动组装图例
melt_patch = Patch(color=C['Melt_Season'], label='Melt Season')
h_auto, l_auto = ax2.get_legend_handles_labels()
# 此时 h_auto 包含 6 个元素 (3条线, 3个阴影)，最后加入背景
h_auto.append(melt_patch)
l_auto.append('Melt Season')

# 排序逻辑：[Lead线, Ice线, Refrozen线, Lead阴影, Ice阴影, Refrozen阴影, Melt背景]
order = [0, 2, 4, 1, 3, 5, 6] 
ax2.legend([h_auto[i] for i in order], [l_auto[i] for i in order], 
           ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
           frameon=False, fontsize=10, columnspacing=1.2)

ax2.grid(axis='both', linestyle='-', alpha=0.1)
for spine in ['top', 'right']: ax2.spines[spine].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(hspace=0.35)

plt.savefig(r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\20160114\statistic\refined_comparison_with_refrozen.png", dpi=600, bbox_inches='tight')
plt.show()