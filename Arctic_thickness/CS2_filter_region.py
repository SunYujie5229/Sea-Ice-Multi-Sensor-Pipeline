import os
import csv
import shutil
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from collections import defaultdict

# ================= 配置区域 =================
# 定义北极范围
MIN_LAT = 60.0  
# 路径配置
source_root = r'Z:\Cryosat\Cryosat-2 SAR L2'
target_root = r'E:\Arctic\SAR L2'

# CSV 和 图片保存路径
csv_log_path = os.path.join(target_root, "Arctic_Files_List.csv")
plot_save_path = os.path.join(target_root, "Coverage_Statistics.png")
# ===========================================

def plot_coverage(stats):
    """
    绘制覆盖情况图表
    stats: dict, 格式为 { '2024-01': count, '2024-02': count, ... }
    """
    if not stats:
        print("警告：没有统计到任何匹配数据，跳过绘图。")
        return

    # 排序标签
    sorted_keys = sorted(stats.keys())
    counts = [stats[k] for k in sorted_keys]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_keys, counts, color='skyblue', edgecolor='navy')
    
    plt.title('CryoSat-2 Arctic Data Coverage by Month', fontsize=14)
    plt.xlabel('Year-Month', fontsize=12)
    plt.ylabel('File Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # 在柱状图上方标注具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(plot_save_path)
    print(f"统计图表已保存至: {plot_save_path}")

# 确保目标根目录存在
if not os.path.exists(target_root):
    os.makedirs(target_root)

# 初始化记录
csv_data = [["FileName", "YearMonth", "SourcePath"]]
coverage_stats = defaultdict(int)  # 用于绘图统计: {"2024-09": 10}

print(f"开始扫描目录: {source_root} ...")

found_count = 0
total_scanned = 0

for root, dirs, files in os.walk(source_root):
    nc_files = [f for f in files if f.lower().endswith(".nc")]
    if not nc_files:
        continue

    for file in nc_files:
        total_scanned += 1
        file_path = os.path.join(root, file)
        
        try:
            nc_data = Dataset(file_path, "r")
            
            # 兼容性检查变量名
            lat_var_name = None
            for v in ["lat_20_ku", "lat_01", "lat"]:
                if v in nc_data.variables:
                    lat_var_name = v
                    break
            
            if lat_var_name is None:
                # print(f"[跳过] {file}: 找不到纬度变量") # 如果文件太多可以注释掉
                nc_data.close()
                continue

            # 读取纬度数据
            lats = nc_data.variables[lat_var_name][:]
            
            # 核心筛选逻辑：只要有一个点在北极范围内即选中
            if (lats >= MIN_LAT).any():
                # 从路径提取年月 (假设结构为 ...\YYYY\MM\...)
                # 兼容处理：如果路径深度不足，尝试从文件名解析或标记为Unknown
                relative_path = os.path.relpath(root, source_root)
                path_parts = relative_path.split(os.sep)
                
                year = path_parts[0] if len(path_parts) > 0 else "0000"
                month = path_parts[1] if len(path_parts) > 1 else "00"
                ym_label = f"{year}-{month}"

                # 统计
                coverage_stats[ym_label] += 1
                found_count += 1

                # 复制文件
                dest_dir = os.path.join(target_root, year, month)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                
                dest_path = os.path.join(dest_dir, file)
                if not os.path.exists(dest_path):
                    shutil.copy2(file_path, dest_path)

                csv_data.append([file, ym_label, file_path])
            
            nc_data.close()

        except Exception as e:
            print(f"处理文件失败 {file}: {e}")

# 保存结果
with open(csv_log_path, 'w', newline='', encoding='utf-8-sig') as f:
    csv.writer(f).writerows(csv_data)

# 绘图
plot_coverage(coverage_stats)

print("-" * 30)
print(f"任务完成！")
print(f"总扫描文件: {total_scanned}")
print(f"匹配北极数据: {found_count}")
print(f"清单保存至: {csv_log_path}")