import os
import re
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box


# -----------------------------
# 1. 从文件名提取日期（YYYYMMDD）
# -----------------------------
def extract_date_from_name(filename: str):
    """
    适配如下格式：
    - CS_OFFL_SIR_SIN_1B_20230105T180803_...
    - S1A_EW_GRDM_1SDH_20230601T120119_...
    """
    m = re.search(r"_(\d{8})T\d{6}", filename)
    if m:
        return m.group(1)
    return None


# -----------------------------
# 2. 扫描目录下指定后缀文件，并解析日期
# -----------------------------
def scan_files_with_date(root_folder: str, exts):
    records = []
    exts = tuple(e.lower() for e in exts)

    for current_root, _, files in os.walk(root_folder):
        for f in files:
            if f.lower().endswith(exts):
                date = extract_date_from_name(f)
                if date is None:
                    continue
                full_path = os.path.join(current_root, f)
                records.append({
                    "filename": f,
                    "full_path": full_path,
                    "date": date
                })
    return records


# -----------------------------
# 3. 检查一个 GPKG 与一个 TIFF 是否空间 overlap
# -----------------------------
def has_spatial_overlap(gpkg_path: str, tif_path: str) -> bool:
    try:
        # 读 GPKG
        gdf = gpd.read_file(gpkg_path)
        if gdf.empty:
            return False
    except Exception as e:
        print(f"⚠ 读取 GPKG 失败：{gpkg_path} | {e}")
        return False

    try:
        # 读 TIFF
        with rasterio.open(tif_path) as src:
            raster_crs = src.crs
            bounds = src.bounds
    except Exception as e:
        print(f"⚠ 读取 TIFF 失败：{tif_path} | {e}")
        return False

    if bounds is None:
        return False

    raster_poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    # CRS 处理：如果有 CRS，则投影到 tiff 的 CRS
    try:
        if gdf.crs is not None and raster_crs is not None and gdf.crs != raster_crs:
            gdf_proj = gdf.to_crs(raster_crs)
        else:
            gdf_proj = gdf
    except Exception as e:
        print(f"⚠ GPKG 投影失败，假定同一坐标系：{gpkg_path} | {e}")
        gdf_proj = gdf

    # 先用整体 bounding box 做一个快速筛选
    minx, miny, maxx, maxy = gdf_proj.total_bounds
    gpkg_bbox = box(minx, miny, maxx, maxy)

    if not raster_poly.intersects(gpkg_bbox):
        # 连 bbox 都不交，肯定不重叠
        return False

    # 再检查是否至少有一个点/几何在 raster 范围内
    try:
        mask = gdf_proj.geometry.within(raster_poly)
        if mask.any():
            return True
        # 有些几何可能是线/面，用 intersects 更稳妥
        mask2 = gdf_proj.geometry.intersects(raster_poly)
        return bool(mask2.any())
    except Exception as e:
        print(f"⚠ 几何判断失败：{gpkg_path} vs {tif_path} | {e}")
        return False


# -----------------------------
# 4. 主函数：找时间+空间 overlap，并写入 Excel
# -----------------------------
def find_spatiotemporal_overlaps(
    gpkg_folder: str,
    tif_folder: str,
    excel_out: str
):
    # 扫描文件
    print("🔍 正在扫描 GPKG 文件...")
    gpkg_files = scan_files_with_date(gpkg_folder, [".gpkg"])

    print("🔍 正在扫描 TIFF 文件...")
    tif_files = scan_files_with_date(tif_folder, [".tif", ".tiff"])

    if not gpkg_files:
        print("❌ 未找到任何带日期的 GPKG 文件")
        return
    if not tif_files:
        print("❌ 未找到任何带日期的 TIFF 文件")
        return

    # 按日期组织 TIFF，提高匹配效率
    tif_by_date = {}
    for t in tif_files:
        tif_by_date.setdefault(t["date"], []).append(t)

    results = []

    # 遍历每一个 GPKG，寻找同日期的 TIFF，并检查空间 overlap
    for g in gpkg_files:
        date = g["date"]
        if date not in tif_by_date:
            continue

        print(f"\n📅 日期 {date}：GPKG = {os.path.basename(g['full_path'])}")
        candidate_tifs = tif_by_date[date]

        for t in candidate_tifs:
            print(f"  → 检查 TIFF: {os.path.basename(t['full_path'])}")
            if has_spatial_overlap(g["full_path"], t["full_path"]):
                print("    ✅ 空间 + 时间 overlap，有效，记录中...")
                results.append({
                    "date": date,
                    "gpkg_filename": g["filename"],
                    "gpkg_path": g["full_path"],
                    "tif_filename": t["filename"],
                    "tif_path": t["full_path"]
                })
            else:
                print("    ⛔ 无空间 overlap，跳过")

    # 写 Excel
    if results:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(excel_out), exist_ok=True)
        df.to_excel(excel_out, index=False)
        print(f"\n🎉 完成！共找到 {len(results)} 条空间+时间 overlap 记录")
        print(f"📁 结果已保存到：{excel_out}")
    else:
        print("\n⚠ 未找到任何同时满足时间+空间 overlap 的文件组合")


# -----------------------------
# 5. 实际调用
# -----------------------------
if __name__ == "__main__":
    gpkg_folder = r"F:\NWP\CS2_S1_matched\CS2 File\CS2 class point\2018"
    tif_folder  = r"C:\Users\TJ002\Desktop\classification result\2018"
    excel_out   = r"C:\Users\TJ002\Desktop\CS2_S1_spatiotemporal_overlap_2018.xlsx"

    find_spatiotemporal_overlaps(gpkg_folder, tif_folder, excel_out)
