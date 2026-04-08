import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 设置输入输出路径
# 使用你指定的保存地址
OUTPUT_DIR = r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\201604021\statistic"
INPUT_FILE = os.path.join(OUTPUT_DIR, "monthly_SIC_stats_ALL_YEARS.csv")

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 2. 读取数据
df = pd.read_csv(INPUT_FILE)

# 3. 数据筛选
target_bins = ["70-80", "80-90", "90-100"]
sub_df = df[df['sic_bin'].isin(target_bins)].copy()
sub_df['sic_bin'] = pd.Categorical(sub_df['sic_bin'], categories=target_bins, ordered=True)

# 设置风格
sns.set_theme(style="whitegrid", font_scale=1.2)

# ==========================================================
# 绘图并保存至指定地址
# ==========================================================

# 图 1: 热力图
plt.figure(figsize=(12, 5))
pivot_df = sub_df.pivot(index="sic_bin", columns="month", values="diff_lead")
sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="RdBu_r", center=0, cbar_kws={'label': 'Δ Density (Lead)'})
plt.title("Monthly Lead Density Difference (CS2 - S1)", fontsize=15, fontweight='bold', pad=20)
plt.savefig(os.path.join(OUTPUT_DIR, "bias_heatmap_high_sic.png"), dpi=300, bbox_inches='tight')
plt.close()

# 图 2: 分组柱状图
plt.figure(figsize=(14, 6))
sns.barplot(data=sub_df, x="month", y="diff_lead", hue="sic_bin", palette="muted")
plt.axhline(0, color='black', linewidth=1.2)
plt.title("Bias Comparison Across Months and High SIC Intervals", fontsize=15, fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, "bias_barplot_high_sic.png"), dpi=300, bbox_inches='tight')
plt.close()

# 图 3: 多面板趋势图
g = sns.FacetGrid(sub_df, col="sic_bin", height=5, aspect=1.1, sharey=True)
g.map(sns.lineplot, "month", "diff_lead", marker="o", color="#d62728", linewidth=2.5)
g.map(plt.axhline, y=0, color="gray", ls="--", alpha=0.7)
plt.subplots_adjust(top=0.8)
g.fig.suptitle("Seasonal Evolution of Bias in High SIC Regions", fontsize=16, fontweight='bold')
g.savefig(os.path.join(OUTPUT_DIR, "bias_facet_timeseries.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"所有图表已成功保存至目录: {OUTPUT_DIR}")