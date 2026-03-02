import pandas as pd
import glob
import os
import re


# -----------------------------------------
# 函数 1：读取并预处理
# -----------------------------------------
def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    df['date'] = df['scene_name'].str.extract(r'_(\d{8})T', expand=False)
    df['date'] = pd.to_datetime(df['date'], format="%Y%m%d", errors="coerce")

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    return df


# -----------------------------------------
# 函数 2：生成与你论文格式一致的统计表格
# -----------------------------------------
def compute_monthly_stats(df):
    summary = df.groupby("month").agg(

        # Floe density
        CS2_Floe_Density=('density_CS2_floe', 'mean'),
        CS2_Floe_Std    =('density_CS2_floe', 'std'),
        S1_Floe_Density =('density_S1_floe', 'mean'),
        S1_Floe_Std     =('density_S1_floe', 'std'),

        # Lead density
        CS2_Lead_Density=('density_CS2_lead_only', 'mean'),
        CS2_Lead_Std    =('density_CS2_lead_only', 'std'),
        S1_Lead_Density =('density_S1_lead_only', 'mean'),
        S1_Lead_Std     =('density_S1_lead_only', 'std'),
    ).reset_index()

    return summary


# -----------------------------------------
# 模式 1：处理单一年份 CSV
# 输出位置 = 输入 CSV 所在路径
# -----------------------------------------
def process_single_year(csv_path, output_excel=True):
    df = load_and_preprocess(csv_path)
    year = df['year'].iloc[0]

    summary = compute_monthly_stats(df)

    # 输出路径 = 输入文件目录
    root_dir = os.path.dirname(csv_path)

    # 文件名
    csv_out = os.path.join(root_dir, f"monthly_stats_{year}.csv")
    summary.to_csv(csv_out, index=False)
    print(f"[OK] Single year CSV saved: {csv_out}")

    if output_excel:
        excel_out = os.path.join(root_dir, f"monthly_stats_{year}.xlsx")
        summary.to_excel(excel_out, index=False)
        print(f"[OK] Single year Excel saved: {excel_out}")

    return summary



# -----------------------------------------
# 模式 2：批量处理所有年份（输入 folder/*.csv）
# 输出路径 = 每个 CSV 所在的目录 + 合并 ALL_YEARS 输出到根目录
# -----------------------------------------
def process_all_years(folder_pattern, output_excel=True):

    files = glob.glob(folder_pattern)
    if len(files) == 0:
        print("No CSV found.")
        return
    
    yearly_results = {}
    all_df = []

    # 找到根目录（读取 pattern 的上一级目录）
    root_dir = os.path.dirname(folder_pattern.replace("*",""))
    if root_dir.endswith("\\") or root_dir.endswith("/"):
        root_dir = root_dir[:-1]

    for f in files:
        df = load_and_preprocess(f)
        year = df["year"].iloc[0]

        print(f"Processing year: {year}")

        summary = compute_monthly_stats(df)
        yearly_results[year] = summary
        all_df.append(df)

        # 每个 CSV 输出在自己的目录下
        file_dir = os.path.dirname(f)

        # CSV
        summary.to_csv(os.path.join(file_dir, f"monthly_stats_{year}.csv"), index=False)

        # Excel
        if output_excel:
            summary.to_excel(os.path.join(file_dir, f"monthly_stats_{year}.xlsx"), index=False)


    # ====== 合并所有年份 ======
    merged_df = pd.concat(all_df, ignore_index=True)
    merged_summary = compute_monthly_stats(merged_df)

    # 合并表输出到根目录
    if output_excel:
        merged_excel = os.path.join(root_dir, "monthly_stats_ALL_YEARS.xlsx")
        merged_summary.to_excel(merged_excel, index=False)
        print(f"[OK] ALL YEARS Excel saved: {merged_excel}")

    merged_csv = os.path.join(root_dir, "monthly_stats_ALL_YEARS.csv")
    merged_summary.to_csv(merged_csv, index=False)
    print(f"[OK] ALL YEARS CSV saved: {merged_csv}")

    return yearly_results, merged_summary




# -----------------------------------------
# 使用示例（按需启用）
# -----------------------------------------

# 1）处理单一年份
process_single_year(r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\2015_density_simplest_filter\overall_density_summary_2015_raster_filtered_detailed.csv")

# 2）处理全部年份（文件夹下 *.csv）
# process_all_years(r"D:\SeaIce\overlap\*.csv")
