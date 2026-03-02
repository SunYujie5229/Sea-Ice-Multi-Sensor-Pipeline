import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载并转换为百分比
df = pd.read_csv(r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\20160114\statistic\monthly_stats_ALL_YEARS.csv")

for col in ['CS2_Floe_Density', 'S1_Floe_Density', 'S1_Floeref_desity', 
            'CS2_Lead_Density', 'S1_Lead_Density', 'CS2_Ambi_Density']:
    df[col] *= 100

month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
               7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
df['month_name'] = df['month'].map(month_names)

# 2. 差异计算
df['Ice_Diff_Total_Upper'] = df['S1_Floeref_desity'] - df['CS2_Floe_Density']
df['Ice_Diff_Total_Lower'] = df['S1_Floeref_desity'] - (df['CS2_Floe_Density'] + df['CS2_Ambi_Density'])
df['Lead_Diff_Upper'] = df['S1_Lead_Density'] - df['CS2_Lead_Density']
df['Lead_Diff_Lower'] = df['S1_Lead_Density'] - (df['CS2_Lead_Density'] + df['CS2_Ambi_Density'])
df['Ice_Diff_Single'] = df['S1_Floe_Density'] - df['CS2_Floe_Density']

# 3. 设置绘图风格
plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), gridspec_kw={'height_ratios': [2, 1.3]}, dpi=100)
x = np.arange(len(df))
width = 0.38  # 略微增加宽度使图表更充实

# --- 上图：分组堆叠柱状图 (去除 Edge) ---
# Sentinel-1 配色方案 (由浅入深的蓝)
s1_colors = ['#BDD7EE', '#5B9BD5', '#2E75B6'] 
ax1.bar(x - width/2 - 0.02, df['S1_Floe_Density'], width, label='S1 Floe', color=s1_colors[0], edgecolor='none')
ax1.bar(x - width/2 - 0.02, df['S1_Floeref_desity'] - df['S1_Floe_Density'], width, 
        bottom=df['S1_Floe_Density'], label='S1 Refrozen', color=s1_colors[1], edgecolor='none')
ax1.bar(x - width/2 - 0.02, df['S1_Lead_Density'], width, 
        bottom=df['S1_Floeref_desity'], label='S1 Lead', color=s1_colors[2], edgecolor='none')

# CryoSat-2 配色方案 (红灰调，Ambiguous中间)
cs2_colors = ['#5296A1', '#E9947F', '#F4C4B9'] # 灰色的 Ambiguous 强调其中性
ax1.bar(x + width/2 + 0.02, df['CS2_Floe_Density'], width, label='CS2 Floe', color=cs2_colors[0], edgecolor='none')
ax1.bar(x + width/2 + 0.02, df['CS2_Ambi_Density'], width, 
        bottom=df['CS2_Floe_Density'], label='CS2 Ambiguous', color=cs2_colors[1], edgecolor='none')
ax1.bar(x + width/2 + 0.02, df['CS2_Lead_Density'], width, 
        bottom=df['CS2_Floe_Density'] + df['CS2_Ambi_Density'], label='CS2 Lead', color=cs2_colors[2], edgecolor='none')

# 修饰上图
ax1.set_ylabel('Density (%)', fontweight='bold')
ax1.set_title('Sentinel-1 & CryoSat-2 Classification Consistency Analysis', fontsize=12, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(df['month_name'])
ax1.set_ylim(0, 110)
ax1.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False)
ax1.grid(axis='y', linestyle=':', alpha=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ---------------------------------------------------------
# 下图：差异/残差分析图 (包含 Range 图例)
# ---------------------------------------------------------
# 1. 总海冰差异 (Total Ice Diff) 及阴影区间
p1, = ax2.plot(x, df['Ice_Diff_Total_Upper'], marker='o', markersize=4, color='#2E75B6', linewidth=1.5, label='Total Ice Diff (S1_Total - CS2_Floe)')
f1 = ax2.fill_between(x, df['Ice_Diff_Total_Lower'], df['Ice_Diff_Total_Upper'], color='#2E75B6', alpha=0.1, label='Ice Ambiguity Range')

# 2. 水道差异 (Lead Diff) 及阴影区间
p2, = ax2.plot(x, df['Lead_Diff_Upper'], marker='s', markersize=4, color='#C00000', linewidth=1.5, label='Lead Diff (S1 - CS2_Lead)')
f2 = ax2.fill_between(x, df['Lead_Diff_Lower'], df['Lead_Diff_Upper'], color='#C00000', alpha=0.1, label='Lead Ambiguity Range')

# 3. 单独海冰差异 (Single Ice Diff)
p3, = ax2.plot(x, df['Ice_Diff_Single'], marker='D', markersize=4, color='#70AD47', linestyle='--', linewidth=1.2, label='Single Ice Diff (S1_Floe - CS2_Floe)')

# 修饰下图
ax2.axhline(0, color='black', linewidth=0.8, alpha=0.7)
ax2.set_ylabel('Difference (%)', fontweight='bold')
ax2.set_xlabel('Month')
ax2.set_xticks(x)
ax2.set_xticklabels(df['month_name'])

# 获取所有句柄和标签以创建复合图例
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles, labels=labels, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=9)

ax2.grid(axis='both', linestyle=':', alpha=0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\20160114\statistic\refined_comparison_diffs.png", dpi=600)
plt.show()
