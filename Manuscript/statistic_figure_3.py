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
# 3. 全局样式与配色定义
# ==========================================
plt.rcParams.update({'font.size': 10.5, 'font.family': 'sans-serif'})

COLORS = {
    'S1_Floe': '#4A90E2',      # 冷色调：S1
    'S1_Refrozen': '#95C1F1',
    'S1_Lead': '#2C3E50',
    'CS2_Floe': '#E67E22',     # 暖色调：CS2
    'CS2_Ambi': '#FAD7A0',     # 浅沙色：Ambiguous
    'CS2_Lead': '#C0392B',
    'Diff_Total': '#2980B9',   # 差异图线
    'Diff_Lead': '#C0392B',
    'Diff_Single': '#27AE60',
    'Melt_Season': '#F4F6F7'   # 融化季背景色
}

# 创建画布
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 11), gridspec_kw={'height_ratios': [1.8, 1.5]})
x = np.arange(len(df))
width = 0.38 

# ==========================================
# 4. 上图：成分占比堆叠图
# ==========================================
# Sentinel-1 堆叠
ax1.bar(x - width/2 - 0.01, df['S1_Floe_Density'], width, label='S1 Floe', 
        color=COLORS['S1_Floe'], edgecolor='white', linewidth=0.5)
ax1.bar(x - width/2 - 0.01, df['S1_Floeref_desity'] - df['S1_Floe_Density'], width, 
        bottom=df['S1_Floe_Density'], label='S1 Refrozen', color=COLORS['S1_Refrozen'], edgecolor='white', linewidth=0.5)
ax1.bar(x - width/2 - 0.01, df['S1_Lead_Density'], width, 
        bottom=df['S1_Floeref_desity'], label='S1 Lead', color=COLORS['S1_Lead'], edgecolor='white', linewidth=0.5)

# CryoSat-2 堆叠
ax1.bar(x + width/2 + 0.01, df['CS2_Floe_Density'], width, label='CS2 Floe', 
        color=COLORS['CS2_Floe'], edgecolor='white', linewidth=0.5)
ax1.bar(x + width/2 + 0.01, df['CS2_Ambi_Density'], width, 
        bottom=df['CS2_Floe_Density'], label='CS2 Ambiguous', color=COLORS['CS2_Ambi'], edgecolor='white', linewidth=0.5)
ax1.bar(x + width/2 + 0.01, df['CS2_Lead_Density'], width, 
        bottom=df['CS2_Floe_Density'] + df['CS2_Ambi_Density'], label='CS2 Lead', color=COLORS['CS2_Lead'], edgecolor='white', linewidth=0.5)

# 标注融化季 (Melt Season)
ax1.axvspan(4.5, 7.5, color=COLORS['Melt_Season'], alpha=0.9, zorder=0)
ax1.text(6, 106, 'Melt Season', ha='center', va='bottom', fontsize=10, fontweight='bold', color='gray')

# 修饰 ax1
ax1.set_ylabel('Composition (%)', fontweight='bold', fontsize=12)
ax1.set_title('Seasonal Composition of Sea Ice Classification (S1 vs CS2)', fontsize=14, fontweight='bold', pad=25)
ax1.set_xticks(x)
ax1.set_xticklabels(df['month_name'])
ax1.set_ylim(0, 110)
ax1.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.08), frameon=False)
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ==========================================
# 5. 下图：差异与不确定性分析
# ==========================================
# 融化季背景
ax2.axvspan(4.5, 7.5, color=COLORS['Melt_Season'], alpha=0.9, zorder=0)

# 差异曲线与 Ambiguity 阴影
p1, = ax2.plot(x, df['Ice_Diff_Total_Upper'], marker='o', markersize=5, color=COLORS['Diff_Total'], 
               linewidth=2, label='Total Ice Diff ($S1_{Total} - CS2_{Floe}$)')
f1 = ax2.fill_between(x, df['Ice_Diff_Total_Lower'], df['Ice_Diff_Total_Upper'], 
                      color=COLORS['Diff_Total'], alpha=0.1, label='CS2 Ambiguity Range (Ice)')

p2, = ax2.plot(x, df['Lead_Diff_Upper'], marker='s', markersize=5, color=COLORS['Diff_Lead'], 
               linewidth=2, label='Lead Diff ($S1 - CS2_{Lead}$)')
f2 = ax2.fill_between(x, df['Lead_Diff_Lower'], df['Lead_Diff_Upper'], 
                      color=COLORS['Diff_Lead'], alpha=0.1, label='CS2 Ambiguity Range (Lead)')

p3, = ax2.plot(x, df['Ice_Diff_Single'], marker='D', markersize=5, color=COLORS['Diff_Single'], 
               linestyle='--', linewidth=1.8, label='Single Ice Diff ($S1_{Floe} - CS2_{Floe}$)')

# 零位线
ax2.axhline(0, color='black', linewidth=1, alpha=0.8)

# 修饰 ax2
ax2.set_ylabel('Difference (%)', fontweight='bold', fontsize=12)
ax2.set_xlabel('Month', fontweight='bold', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(df['month_name'])
ax2.set_ylim(-65, 65) # 对称 Y 轴

# --- 精确控制下图图例 ---
melt_patch = Patch(color=COLORS['Melt_Season'], label='Melt Season')
handles2, labels2 = ax2.get_legend_handles_labels()
# 排序：线 -> 阴影 -> 背景
new_handles = [p1, p2, p3, f1, f2, melt_patch]
new_labels = [labels2[0], labels2[2], labels2[4], labels2[1], labels2[3], 'Melt Season']

ax2.legend(handles=new_handles, labels=new_labels, ncol=2, loc='upper center', 
           bbox_to_anchor=(0.5, -0.28), frameon=False, fontsize=9.5)

ax2.grid(axis='both', linestyle='--', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ==========================================
# 6. 保存与输出
# ==========================================
plt.tight_layout()
plt.subplots_adjust(hspace=0.3) # 留出足够的空间给中间的图例

plt.savefig(r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\20160114\statistic\refined_comparison_diffs.png", dpi=600, bbox_inches='tight')
plt.show()
