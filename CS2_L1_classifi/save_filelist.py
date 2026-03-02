import os
import pandas as pd

def save_tif_filelists_for_subfolders(main_folder):
    # 遍历主文件夹下的所有子文件夹
    for sub in os.listdir(main_folder):
        sub_path = os.path.join(main_folder, sub)

        # 只处理文件夹
        if os.path.isdir(sub_path):
            file_list = []

            # 遍历子文件夹内的文件，只保留 .tif / .tiff
            for f in os.listdir(sub_path):
                if f.lower().endswith((".tif", ".tiff")):
                    file_path = os.path.join(sub_path, f)
                    file_list.append({
                        "filename": f,
                        "full_path": file_path,
                        "Available": ""   # 等待你人工标注
                    })

            # 如果子文件夹内没有 TIFF 文件则跳过
            if len(file_list) == 0:
                print(f"⚠ 子文件夹中无 TIFF 文件：{sub}")
                continue

            # 保存 excel
            df = pd.DataFrame(file_list)
            save_path = os.path.join(main_folder, f"{sub}_tif_filelist.xlsx")
            df.to_excel(save_path, index=False)

            print(f"✅ 已保存：{save_path}")


# ==============================
# 使用示例
# ==============================

main_folder = r"C:\Users\TJ002\Desktop\classification result"  # 你的目录路径
save_tif_filelists_for_subfolders(main_folder)
