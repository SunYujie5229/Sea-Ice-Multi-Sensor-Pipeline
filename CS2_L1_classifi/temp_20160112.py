import os

# -------------------- 配置部分 --------------------
# 请将此路径修改为你存放年份文件夹的根目录 (例如：C:\Users\...\filter)
TARGET_FOLDER = r"F:\NWP\CS2_S1_matched\CS2 File\CS2 class point"

# -------------------- 执行逻辑 --------------------
def rename_gpkg_files(root_dir):
    print(f"正在扫描目录: {root_dir}")
    rename_count = 0

    # os.walk 会自动遍历所有子文件夹
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            # 检查文件是否以 _class.gpkg 结尾
            if filename.endswith("_class.gpkg"):
                # 构建旧文件完整路径
                old_path = os.path.join(root, filename)
                
                # 生成新文件名：将 _class 替换为 _classified
                new_filename = filename.replace("_class.gpkg", "_classified.gpkg")
                new_path = os.path.join(root, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"成功: {filename} -> {new_filename}")
                    rename_count += 1
                except Exception as e:
                    print(f"失败: 无法重命名 {filename}，错误: {e}")

    print("-" * 50)
    print(f"重命名完成！共处理文件: {rename_count} 个")

if __name__ == "__main__":
    if os.path.exists(TARGET_FOLDER):
        rename_gpkg_files(TARGET_FOLDER)
    else:
        print("错误：指定的目录不存在，请检查 TARGET_FOLDER 路径。")