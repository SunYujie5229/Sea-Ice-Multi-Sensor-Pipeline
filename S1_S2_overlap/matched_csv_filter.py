import pandas as pd

# 读取原始CSV
csv_path = r"F:\NWP\S1_S2_matched\matched_satellite_images.csv"
df = pd.read_csv(csv_path)

# 只保留 status=success 的记录
df = df[df['status'] == 'success']

# 定义函数：在一个 group 内找最优的影像
def select_best(group):
    return group.sort_values(
        ['time_difference_hours', 'overlap_ratio'], 
        ascending=[True, False]
    ).head(1)

# 分别筛选 Sentinel-2 和 Landsat
s2_matches = (
    df[df['matched_sensor'].str.contains("Sentinel-2", case=False, na=False)]
    .groupby('s1_image_id', group_keys=False)
    .apply(select_best)
)

landsat_matches = (
    df[df['matched_sensor'].str.contains("Landsat", case=False, na=False)]
    .groupby('s1_image_id', group_keys=False)
    .apply(select_best)
)

# 合并结果
best_matches = pd.concat([s2_matches, landsat_matches], ignore_index=True)

# 保存结果
out_path = r"F:\NWP\S1_S2_matched\filtered_best_matches.csv"
best_matches.to_csv(out_path, index=False)

print(f"筛选完成，每景 S1 保留一个 S2 和一个 L8 匹配，已保存到 {out_path}")
