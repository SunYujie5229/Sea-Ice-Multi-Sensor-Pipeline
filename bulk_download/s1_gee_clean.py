import os
import re

# === 设置文件夹路径 ===
folder = r"H:\我的云端硬盘\Project_Yujie\2022_1"

# === 正则表达式：匹配时间范围段，如 20200211T133916_20200211T134016 ===
pattern = re.compile(r"_(\d{8}T\d{6}_\d{8}T\d{6})_")

# === 已出现的时间段集合 ===
seen_times = set()

for filename in sorted(os.listdir(folder)):
    if not filename.lower().endswith(('.tif', '.tiff')):
        continue

    match = pattern.search(filename)
    if not match:
        print(f"⚠️ 无法提取时间范围: {filename}")
        continue

    time_span = match.group(1)
    file_path = os.path.join(folder, filename)

    if time_span not in seen_times:
        # 第一次出现该时间段 → 保留
        seen_times.add(time_span)
        print(f"✅ 保留: {filename}")
    else:
        # 重复时间段 → 删除
        os.remove(file_path)
        print(f"🗑️ 删除: {filename}")

print("\n✅ 完成：已删除时间段相同的重复文件。")
