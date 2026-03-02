import os
import re
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
import numpy as np
import warnings
from shapely.geometry import Point
# 注意：移除matplotlib, Polygon, box等不必要的几何库，只保留核心
from rasterio.transform import rowcol 

# 忽略 geopandas/shapely 相关的警告
warnings.filterwarnings("ignore")

# -------------------- USER CONFIG --------------------
# 请根据您的实际路径修改以下配置
CS2_DIR = r"C:\Users\TJ002\Desktop\CS2_S1_result\filter\2023\gpkg"
S1_DIR  = r"C:\Users\TJ002\Desktop\CS2_S1_result\filter\2023\tif"
MATCH_CSV = r"F:\NWP\CS2_S1_matched\time_match_2023_filter.csv"

# 输出目录
OUTPUT_DIR = r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\2023_density_simplest_filter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 采样窗口半宽：12 像素 * 40m/像素 = 480m 半宽 (总宽 ≈ 960m)
WINDOW_RADIUS = 12 

# S1 掩膜编码 (Fixed)
S1_BACKGROUND = 0
S1_ICE = 1
S1_LEAD = 2
S1_REFR = 3


# -------------------- 辅助函数 (HELPERS) --------------------

def find_file_by_pattern(folder, pattern, extension=None):
    """查找文件名包含特定模式的文件路径。"""
    if not os.path.exists(folder): return None
    for file in os.listdir(folder):
        if pattern in file and (extension is None or file.lower().endswith(extension.lower())):
            return os.path.join(folder, file)
    return None

def detect_class_column(gdf):
    """检测 CS2 分类列名。"""
    for c in ["class", "Class", "CLASS", "classification"]:
        if c in gdf.columns: return c
    return None


# -------------------- 密度计算函数 --------------------

def compute_cs2_density(cs2_gdf, class_col):
    """
    计算 CS2 密度 (针对重叠区域)。
    """
    if cs2_gdf.empty:
         return {
            "count_CS2_lead": 0, "count_CS2_ice": 0, "count_CS2_refrozen": 0, "total_CS2": 0,
            "density_CS2_lead_only": 0.0, "density_CS2_leadref": 0.0, "density_CS2_floe": 0.0
        }

    cs2_gdf[class_col] = cs2_gdf[class_col].str.lower().str.strip()

    count_lead     = (cs2_gdf[class_col] == "lead").sum()
    count_ice      = (cs2_gdf[class_col] == "ice").sum()
    count_refrozen = (cs2_gdf[class_col] == "refrozen").sum()

    total = count_lead + count_ice + count_refrozen
    if total == 0: 
        return None

    return {
        "count_CS2_lead": int(count_lead), "count_CS2_ice": int(count_ice), "count_CS2_refrozen": int(count_refrozen), "total_CS2": int(total),
        "density_CS2_lead_only": count_lead / total, "density_CS2_leadref": (count_lead + count_refrozen) / total, "density_CS2_floe": count_ice / total
    }


def compute_s1_density_point_sampling(
        s1_array, transform, cs2_points_gdf, win_radius=WINDOW_RADIUS):
    """
    点对点窗口采样 (方案 1)：
    直接将 CS2 GDF 中的每一个点作为采样中心，进行窗口采样和去重统计。
    """
    rows, cols = s1_array.shape
    inv_transform = ~transform # 预计算逆变换
    unique_pixels = set()
    
    # 确保输入是 GeoDataFrame/GeoSeries
    if isinstance(cs2_points_gdf, gpd.GeoSeries):
        points_to_process = cs2_points_gdf
    else:
        # 如果是 GeoDataFrame，则使用 geometry 列
        points_to_process = cs2_points_gdf.geometry

    if points_to_process.empty:
        # 移除原代码中的 print 语句
        return None

    # ---------------------------------------
    # 1. 遍历所有 CS2 点，进行窗口采样
    # ---------------------------------------
    for pt in points_to_process:
        # 跳过非 Point 几何体，增加健壮性
        if not isinstance(pt, Point):
            continue 
            
        # 获取地理坐标
        x, y = pt.x, pt.y

        # 坐标转换 (沿用原始方式)
        col, row = inv_transform * (x, y)
        row, col = int(row), int(col)

        # 确保采样中心点在 S1 图像边界内 (虽然主函数已过滤，这里是窗口计算的必要条件)
        if 0 <= row < rows and 0 <= col < cols:
            # window bounds (r1, r2, c1, c2)
            r1, r2 = max(row - win_radius, 0), min(row + win_radius, rows)
            c1, c2 = max(col - win_radius, 0), min(col + win_radius, cols)

            # 遍历窗口内的所有像素，并添加到 set 中去重
            for rr in range(r1, r2):
                for cc in range(c1, c2):
                    unique_pixels.add((rr, cc))
        
    if len(unique_pixels) == 0:
        # 移除原代码中的 print 语句
        return None

    # ---------------------------------------
    # 2. 统计唯一像元
    # ---------------------------------------
    lead_total = 0
    ice_total  = 0
    refr_total = 0
    total_pixels = 0 # 统计非背景像素

    for rr, cc in unique_pixels:
        val = s1_array[rr, cc]
        if val == S1_LEAD:
            lead_total += 1
        elif val == S1_ICE:
            ice_total += 1
        elif val == S1_REFR:
            refr_total += 1
            
        # 只统计被分类的像素 (非背景)
        if val != S1_BACKGROUND:
            total_pixels += 1

    if total_pixels == 0:
        return None

    return {
        "S1_lead_pixels": int(lead_total),
        "S1_ice_pixels": int(ice_total),
        "S1_refrozen_pixels": int(refr_total),
        "S1_total_pixels": int(total_pixels),
        "density_S1_lead_only": lead_total / total_pixels,
        "density_S1_leadref": (lead_total + refr_total) / total_pixels,
        "density_S1_floe": ice_total / total_pixels
    }


# -------------------- MAIN 执行函数 (基于栅格坐标的快速过滤) --------------------
def main():
    print("=" * 80)
    print("CS2–S1 DENSITY CALCULATION SCRIPT (FAST RASTER FILTER & Point-by-Point S1)")
    print("=" * 80)
    print(f"CS2 density is calculated using fast RASTER ROW/COL extent filtering.")
    print(f"WINDOW_RADIUS = {WINDOW_RADIUS} pixels (480m half-width)\n")

    match_df = pd.read_csv(MATCH_CSV)
    results = []

    for idx, row in match_df.iterrows():
        print(f"\n[{idx+1}/{len(match_df)}] Scene: {row['sceneName']}")

        # --- load CS2 ---
        cs2_key = re.search(r"(\d{8}T\d{6}_\d{8}T\d{6})", Path(row["cs2_path"]).name)
        if not cs2_key: continue
        cs2_file = find_file_by_pattern(CS2_DIR, cs2_key.group(1), extension=".gpkg")
        s1_file  = find_file_by_pattern(S1_DIR, row["sceneName"], extension=".tif")
        if not cs2_file or not s1_file: 
            print("  ✗ Required files not found.")
            continue

        print(f"  ✓ CS2: {Path(cs2_file).name}")
        print(f"  ✓ S1 : {Path(s1_file).name}")

        try:
            cs2_gdf = gpd.read_file(cs2_file)
        except Exception as e:
            print(f"  ✗ Failed to read CS2 file: {e}")
            continue

        class_col = detect_class_column(cs2_gdf)
        if not class_col: 
            print("  ✗ No classification column found in CS2.")
            continue
        
        total_cs2_pre_filter = len(cs2_gdf)

        # --- load S1 and setup filter bounds ---
        try:
            with rasterio.open(s1_file) as src:
                s1_data = src.read(1)
                s1_crs = src.crs
                s1_transform = src.transform
                s1_rows, s1_cols = src.shape # 获取栅格尺寸 (rows, cols)
        except Exception as e:
            print(f"  ✗ Failed to open S1 file: {e}")
            continue
            
        # 1. 将 CS2 投影到 S1 CRS
        cs2_proj = cs2_gdf.to_crs(s1_crs)
        
        # 2. 批量将投影后的 CS2 地理坐标转换为 S1 栅格的行/列坐标
        x_coords = cs2_proj.geometry.x.tolist()
        y_coords = cs2_proj.geometry.y.tolist()
        
        # 转换并返回行/列数组 (Python lists)
        rows_list, cols_list = rowcol(s1_transform, x_coords, y_coords)
        
        # 3. 转换为 NumPy array 以支持布尔过滤
        rows_arr = np.array(rows_list)
        cols_arr = np.array(cols_list)
        
        # 4. 构造布尔过滤器：点必须在 [0, rows) 和 [0, cols) 范围内
        row_filter = (rows_arr >= 0) & (rows_arr < s1_rows)
        col_filter = (cols_arr >= 0) & (cols_arr < s1_cols)
        
        # 5. 应用过滤器获取重叠点 (CS2 Overlap Points)
        filter_mask = row_filter & col_filter
        cs2_gdf_overlapped = cs2_gdf[filter_mask].copy() # 使用原始CRS的gdf进行过滤
        
        total_cs2_post_filter = len(cs2_gdf_overlapped)
        print(f"  ℹ️ CS2 Points (Pre-Filter): {total_cs2_pre_filter} | (Post-Filter/Overlap): {total_cs2_post_filter}")
        
        # --- compute CS2 density (使用过滤后的数据) ---
        cs2_density = compute_cs2_density(cs2_gdf_overlapped, class_col)
        if cs2_density is None or cs2_density["total_CS2"] == 0:
            print("  ✗ No valid CS2 classified points within the S1 overlap.")
            continue

        # --- reproject CS2 (overlapped points only) to S1 CRS ---
        # 重新投影过滤后的点，用于 S1 采样
        cs2_proj_overlapped = cs2_gdf_overlapped.to_crs(s1_crs)

        # --- compute S1 density ---
        s1_density = compute_s1_density_point_sampling(
            s1_array=s1_data,
            transform=s1_transform,
            cs2_points_gdf=cs2_proj_overlapped.geometry,
            win_radius=WINDOW_RADIUS
        )

        if s1_density is None:
            print("  ✗ No valid S1 pixels were found by the clipped CS2 points.")
            continue

        # --- combine & compute differences (详细输出) ---
        combined = {
            "scene_name": row["sceneName"],
            "window_radius_px": WINDOW_RADIUS,
            
            # CS2 详细分类数据 (重叠区域)
            "count_CS2_lead": cs2_density["count_CS2_lead"],
            "count_CS2_ice": cs2_density["count_CS2_ice"],
            "count_CS2_refrozen": cs2_density["count_CS2_refrozen"],
            "total_CS2_overlap": cs2_density["total_CS2"],
            
            # 密度结果 (CS2)
            "density_CS2_lead_only": cs2_density["density_CS2_lead_only"],
            "density_CS2_leadref": cs2_density["density_CS2_leadref"],
            "density_CS2_floe": cs2_density["density_CS2_floe"],
            
            # S1 详细分类数据 (采样区域)
            "S1_lead_pixels": s1_density["S1_lead_pixels"],
            "S1_ice_pixels": s1_density["S1_ice_pixels"],
            "S1_refrozen_pixels": s1_density["S1_refrozen_pixels"],
            "S1_total_pixels": s1_density["S1_total_pixels"],

            # 密度结果 (S1)
            "density_S1_lead_only": s1_density["density_S1_lead_only"],
            "density_S1_leadref": s1_density["density_S1_leadref"],
            "density_S1_floe": s1_density["density_S1_floe"],

            # 差异
            "diff_lead_density": cs2_density["density_CS2_lead_only"] - s1_density["density_S1_lead_only"],
            "diff_leadref_density": cs2_density["density_CS2_leadref"] - s1_density["density_S1_leadref"],
            "diff_floe_density": cs2_density["density_CS2_floe"] - s1_density["density_S1_floe"]
        }

        results.append(combined)
        print(f"  ✓ Density computed. CS2 Overlap Points: {total_cs2_post_filter}")

    # --- Save summary ---
    if results:
        outcsv = os.path.join(OUTPUT_DIR, "overall_density_summary_2023_raster_filtered_detailed.csv")
        pd.DataFrame(results).to_csv(outcsv, index=False)
        print("\nSaved:", outcsv)
    else:
        print("\n✗ No valid results to save.")

    print("\nDone.")


if __name__ == "__main__":
    main()