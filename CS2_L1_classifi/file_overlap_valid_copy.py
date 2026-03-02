import os
import shutil
import pandas as pd
import re


def extract_year_from_filename(filename):
    """从文件名提取年份 20xx"""
    m = re.search(r"(20\d{2})", filename)
    return m.group(1) if m else None


def copy_files_by_xlsx(xlsx_folder, output_root):
    print("📂 开始遍历文件夹：", xlsx_folder)

    for file in os.listdir(xlsx_folder):
        if not file.lower().endswith(".xlsx"):
            continue

        print(f"\n🔍 发现 Excel 文件: {file}")

        year = extract_year_from_filename(file)
        if year is None:
            print(f"⚠ 无法从文件名提取年份，跳过 {file}")
            continue

        excel_path = os.path.join(xlsx_folder, file)

        # ---- 读取 Excel ----
        try:
            df = pd.read_excel(excel_path)
        except Exception as e:
            print(f"❌ 无法读取 Excel：{excel_path} | 错误: {e}")
            continue

        # ---- 检查列是否存在 ----
        if "tif_path" not in df.columns or "gpkg_path" not in df.columns:
            print(f"⚠ Excel 中缺少 tif_path 或 gpkg_path 列，跳过 {file}")
            continue

        # ---- 创建输出目录结构 ----
        year_folder = os.path.join(output_root, year)
        tif_folder = os.path.join(year_folder, "tif")
        gpkg_folder = os.path.join(year_folder, "gpkg")

        os.makedirs(tif_folder, exist_ok=True)
        os.makedirs(gpkg_folder, exist_ok=True)

        # ---- 按行复制文件 ----
        copied_tif = 0
        copied_gpkg = 0

        for idx, row in df.iterrows():
            tif = row["tif_path"]
            gpkg = row["gpkg_path"]

            if isinstance(tif, str) and os.path.isfile(tif):
                shutil.copy(tif, tif_folder)
                copied_tif += 1

            if isinstance(gpkg, str) and os.path.isfile(gpkg):
                shutil.copy(gpkg, gpkg_folder)
                copied_gpkg += 1

        print(f"✔ 年份 {year} 完成复制：TIF={copied_tif}，GPKG={copied_gpkg}")


# ============== 主程序入口 ==============
if __name__ == "__main__":
    print("脚本开始运行…")

    xlsx_folder = r"C:\Users\TJ002\Desktop\CS2_S1_result"   # ← 你的年份xlsx所在文件夹
    output_root = r"C:\Users\TJ002\Desktop\CS2_S1_result\filter1" # ← 输出文件夹根目录

    copy_files_by_xlsx(xlsx_folder, output_root)

    print("所有年份处理完成！")




