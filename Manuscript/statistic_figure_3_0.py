import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==========================================
# 1. 数据读取与预处理函数
# ==========================================
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    
    # 清洗列名：去除可能的空格
    df.columns = df.columns.str.strip()
    
    # 检查必要列是否存在，并处理列名差异
    # 如果你的 CSV 里叫 'CS2_Amb_Density'，请在此处重命名
    rename_dict = {
        'CS2_Ambi_Density': 'CS2_Ambi_Density', # 保持一致
        'S1_Floeref_desity': 'S1_Floeref_Density' # 顺便修正拼写错误
    }
    # 检查逻辑：如果不存在 'CS2_Ambi_Density'，尝试寻找类似的列名
    if 'CS2_Ambi_Density' not in df.columns:
        print(f"Warning: 'CS2_Ambi_Density' not found. Available columns: {list(df.columns)}")
        # 这里可以添加一个回退逻辑，比如 df['CS2_Ambi_Density'] = 0 或抛出更友好的错误
    
    # 转换百分比 (0-1 -> 0-100)
    target_cols = ['CS2_Floe_Density', 'S1_Floe_Density', 'S1_Floeref_desity', 
                   'CS2_Lead_Density', 'S1_Lead_Density', 'CS2_Ambi_Density']
    for col in [c for c in target_cols if c in df.columns]:
        if df[col].max() <= 1.0:
            df[col] *= 100

    # 月份映射
    month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                   7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    df['month_name'] = df['month'].map(month_names)
    
    # 计算差异逻辑
    df['Ice_Diff_Total_Upper'] = df['S1_Floeref_desity'] - df['CS2_Floe_Density']
    df['Ice_Diff_Total_Lower'] = df['S1_Floeref_desity'] - (df['CS2_Floe_Density'] + df['CS2_Ambi_Density'])
    df['Lead_Diff_Upper'] = df['S1_Lead_Density'] - df['CS2_Lead_Density']
    df['Lead_Diff_Lower'] = df['S1_Lead_Density'] - (df['CS2_Lead_Density'] + df['CS2_Ambi_Density'])
    df['Ice_Diff_Single'] = df['S1_Floe_Density'] - df['CS2_Floe_Density']
    
    return df

# ==========================================
# 2. 绘图配置 (IEEE 风格)
# ==========================================
def plot_sea_ice_comparison(df, save_path):
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.0
    })

    COLORS = {
        'S1_Floe': '#08519C', 'S1_Refrozen': '#6BAED6', 'S1_Lead': '#A50F15',
        'CS2_Floe': '#4292C6', 'CS2_Ambi': '#969696', 'CS2_Lead': '#EF3B2C',
        'Diff_Total': '#084594', 'Diff_Lead': '#CB181D', 'Diff_Single': '#238B45',
        'Melt_Season': '#F0F0F0'
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 9), gridspec_kw={'height_ratios': [1, 0.8]})
    x = np.arange(len(df))
    width = 0.35

    # --- 上图：堆叠柱状图 ---
    ax1.bar(x - width/2 - 0.02, df['S1_Floe_Density'], width, label='$S1_{Floe}$', color=COLORS['S1_Floe'])
    ax1.bar(x - width/2 - 0.02, df['S1_Floeref_desity'] - df['S1_Floe_Density'], width, 
            bottom=df['S1_Floe_Density'], label='$S1_{Refrozen}$', color=COLORS['S1_Refrozen'])
    ax1.bar(x - width/2 - 0.02, df['S1_Lead_Density'], width, 
            bottom=df['S1_Floeref_desity'], label='$S1_{Lead}$', color=COLORS['S1_Lead'])

    ax1.bar(x + width/2 + 0.02, df['CS2_Floe_Density'], width, label='$CS2_{Floe}$', color=COLORS['CS2_Floe'])
    ax1.bar(x + width/2 + 0.02, df['CS2_Ambi_Density'], width, 
            bottom=df['CS2_Floe_Density'], label='$CS2_{Ambi}$', color=COLORS['CS2_Ambi'])
    ax1.bar(x + width/2 + 0.02, df['CS2_Lead_Density'], width, 
            bottom=df['CS2_Floe_Density'] + df['CS2_Ambi_Density'], label='$CS2_{Lead}$', color=COLORS['CS2_Lead'])

    # --- 下图：差异分析 ---
    ax2.plot(x, df['Ice_Diff_Total_Upper'], marker='o', markersize=4, color=COLORS['Diff_Total'], label='Total Ice Diff')
    ax2.fill_between(x, df['Ice_Diff_Total_Lower'], df['Ice_Diff_Total_Upper'], color=COLORS['Diff_Total'], alpha=0.15)
    ax2.plot(x, df['Lead_Diff_Upper'], marker='s', markersize=4, color=COLORS['Diff_Lead'], label='Lead Diff')
    ax2.fill_between(x, df['Lead_Diff_Lower'], df['Lead_Diff_Upper'], color=COLORS['Diff_Lead'], alpha=0.15)
    ax2.plot(x, df['Ice_Diff_Single'], marker='^', linestyle='--', color=COLORS['Diff_Single'], label='Single Ice Diff')
    
    # 细节修饰 (Melt Season, Grid, Spines 等)
    for ax in [ax1, ax2]:
        ax.axvspan(4.5, 7.5, color=COLORS['Melt_Season'], zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(df['month_name'])
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.spines[['top', 'right']].set_visible(False)

    ax1.set_ylabel('Composition (%)', fontweight='bold')
    ax2.set_ylabel('Difference (%)', fontweight='bold')
    ax1.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax2.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.25), frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == "__main__":
    csv_path = r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\20160114\statistic\monthly_stats_ALL_YEARS.csv"
    output_fig = r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\20160114\statistic\refined_comparison_IEEE.png"
    
    try:
        data = load_and_clean_data(csv_path)
        plot_sea_ice_comparison(data, output_fig)
    except Exception as e:
        print(f"Error occurred: {e}")