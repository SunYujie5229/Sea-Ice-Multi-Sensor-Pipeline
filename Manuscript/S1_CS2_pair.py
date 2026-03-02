import pandas as pd
import os

# 1. 定义文件路径
# 请将此路径修改为你实际存放 time_match 文件的文件夹路径
source_folder = r'F:\NWP\CS2_S1_matched' 
# S1文件名索引路径
overlap_file = r'C:\Users\TJ002\Desktop\CS2_S1_result\overlap\overlap_S1name.csv'
# 输出结果路径
output_file = r'C:\Users\TJ002\Desktop\CS2_S1_result\overlap\matched_pairs_final.csv'

def merge_s1_cs2_data():
    # 读取 S1 文件名列表
    # 假设列名是 'scene_name'，根据你的截图调整
    s1_df = pd.read_csv(overlap_file)
    s1_names = s1_df['scene_name'].unique()
    
    all_matched_data = []

    # 2. 遍历 2015 到 2023 年的所有 filter 文件
    for year in range(2015, 2024):
        file_name = f'time_match_{year}_filter.csv'
        file_path = os.path.join(source_folder, file_name)
        
        if os.path.exists(file_path):
            print(f"正在处理: {file_name}")
            # 读取当年的匹配表
            year_df = pd.read_csv(file_path)
            
            # 3. 检索匹配：筛选出 sceneNam 在 s1_names 中的行
            # 注意：根据你的截图，匹配表里的列名似乎是 'sceneNam'（少个 e）
            filtered_df = year_df[year_df['sceneName'].isin(s1_names)]
            
            if not filtered_df.empty:
                all_matched_data.append(filtered_df)
        else:
            print(f"跳过：未找到文件 {file_name}")

    # 4. 合并所有年份的数据并保存
    if all_matched_data:
        final_df = pd.concat(all_matched_data, ignore_index=True)
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"--- 任务完成 ---")
        print(f"共提取匹配对: {len(final_df)} 条")
        print(f"保存路径: {output_file}")
    else:
        print("未发现匹配项，请检查列名是否对应。")

if __name__ == "__main__":
    merge_s1_cs2_data()