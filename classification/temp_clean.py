import pandas as pd
import os

# --- 1. 设置文件和文件夹路径 ---
# 使用原始字符串(r"...")可以避免反斜杠问题
csv_file_path = r"E:\NWP\CS2_S1_matched\time_match_2023_filter.csv"
target_directory = r"F:\NWP\sentinel1 gee\2023"


# --- 2. 从CSV文件中读取需要保留的场景名称 ---
try:
    print(f"正在从 {csv_file_path} 读取场景名称...")
    df = pd.read_csv(csv_file_path)
    # 确保'sceneName'列存在
    if 'sceneName' not in df.columns:
        print(f"错误：CSV文件中找不到名为 'sceneName' 的列。")
        exit()
    # 将需要保留的名称存储在一个集合中
    names_to_keep = set(df['sceneName'])
    print(f"成功读取 {len(names_to_keep)} 个场景名称用于匹配。")
except FileNotFoundError:
    print(f"错误：找不到CSV文件 '{csv_file_path}'。请检查路径是否正确。")
    exit()
except Exception as e:
    print(f"读取CSV文件时发生错误: {e}")
    exit()

# --- 3. 检查目标文件夹是否存在 ---
if not os.path.isdir(target_directory):
    print(f"错误：找不到目标文件夹 '{target_directory}'。请检查路径是否正确。")
    exit()

print(f"\n开始处理目标文件夹: {target_directory}")

# --- 4. 遍历目标文件夹中的所有文件并执行清理操作 ---
# os.listdir() 会列出文件夹下的所有项目（包括文件和子文件夹）
for filename in os.listdir(target_directory):
    file_path = os.path.join(target_directory, filename)

    # 我们只处理文件，忽略任何子文件夹
    if os.path.isfile(file_path):
        
        # 默认此文件需要被删除，除非满足保留条件
        should_be_kept = False
        
        # 检查文件名是否包含任何一个需要保留的sceneName
        for scene_name in names_to_keep:
            if scene_name in filename:
                # 如果找到了匹配的sceneName，再检查文件后缀是否正确
                if filename.lower().endswith(('.tiff', '.tif', '.png')):
                    should_be_kept = True
                # 只要找到一个匹配的scene_name，就可以停止对这个文件的检查了
                break
        
        # 根据检查结果决定是保留还是删除
        if should_be_kept:
            print(f"  [保留] {filename}")
        else:
            print(f"  [删除] {filename}")
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"    - 删除文件时出错: {e}")

print("\n所有操作已完成。")