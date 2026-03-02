import os
import re
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
import numpy as np
import warnings
from shapely.ops import unary_union
from shapely.geometry import LineString, MultiLineString, Point

warnings.filterwarnings("ignore")

# -------------------- USER CONFIG --------------------
CS2_DIR = r"C:\Users\TJ002\Desktop\CS2_S1_result\filter\2023\gpkg"
S1_DIR  = r"C:\Users\TJ002\Desktop\CS2_S1_result\filter\2023\tif"
MATCH_CSV = r"F:\NWP\CS2_S1_matched\time_match_2023_filter.csv"

OUTPUT_DIR = r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\2023_lead_refrozen_density_point_sampling"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# S1 Resolution: 40m/pixel
# CS2 Point Spacing: 300m
# CS2 Footprint: 300m - 1500m (Avg ≈ 960m)

# --- OPTIMIZED PARAMETERS FOR ROBUSTNESS & COVERAGE ---
# Neighborhood half-window size (pixels): 12 * 40m = 480m half-width (Total width ≈ 960m)
# 采样窗口宽度大于点间距，确保连续覆盖。
WINDOW_RADIUS = 12

# S1 encoding (fixed)
S1_BACKGROUND = 0
S1_ICE = 1
S1_LEAD = 2
S1_REFR = 3


# -------------------- HELPERS (UNCHANGED) --------------------
def find_file_by_pattern(folder, pattern, extension=None):
    """Find file whose name contains pattern (and optional extension)."""
    if not os.path.exists(folder):
        return None
    for file in os.listdir(folder):
        if pattern in file:
            if extension is None or file.lower().endswith(extension.lower()):
                return os.path.join(folder, file)
    return None


def detect_class_column(gdf):
    """Detect CS2 classification column name."""
    for c in ["class", "Class", "CLASS", "classification"]:
        if c in gdf.columns:
            return c
    return None


# -------------------- CS2 DENSITY (UNCHANGED) --------------------
def compute_cs2_density(cs2_gdf, class_col):
    """Compute CS2 lead-only / lead+refrozen / floe densities."""
    cs2_gdf[class_col] = cs2_gdf[class_col].str.lower().str.strip()

    count_lead     = (cs2_gdf[class_col] == "lead").sum()
    count_ice      = (cs2_gdf[class_col] == "ice").sum()
    count_refrozen = (cs2_gdf[class_col] == "refrozen").sum()

    total = count_lead + count_ice + count_refrozen
    if total == 0:
        return None

    return {
        "count_CS2_lead": int(count_lead),
        "count_CS2_ice": int(count_ice),
        "count_CS2_refrozen": int(count_refrozen),
        "total_CS2": int(total),
        "density_CS2_lead_only": count_lead / total,
        "density_CS2_leadref": (count_lead + count_refrozen) / total,
        "density_CS2_floe": count_ice / total
    }


# -------------------- S1 DENSITY (Point-by-Point Sampling - Robust Solution) --------------------
def compute_s1_density_point_sampling(
        s1_array, transform, cs2_points_gdf, win_radius=WINDOW_RADIUS):
    """
    点对点窗口采样 (方案 1)：
    直接将 CS2 GDF 中的每一个点作为采样中心，进行窗口采样和去重统计。
    适用于稀疏或破碎的 CS2 数据，保证有数据即有结果。
    """
    rows, cols = s1_array.shape
    inv_transform = ~transform # 预计算逆变换
    unique_pixels = set()
    
    # 确保输入是 GeoDataFrame/GeoSeries
    if isinstance(cs2_points_gdf, gpd.GeoSeries):
        points_to_process = cs2_points_gdf
    else:
        # 如果不是 GeoSeries，则可能是 GeoDataFrame
        points_to_process = cs2_points_gdf.geometry

    if points_to_process.empty:
        print("    ⚠️ CS2 points are empty after reprojection.")
        return None

    print(f"    ℹ️ Using {len(points_to_process)} CS2 points for direct window sampling (radius={win_radius}px).")

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

        # 确保采样中心点在 S1 图像边界内
        if 0 <= row < rows and 0 <= col < cols:
            # window bounds (r1, r2, c1, c2)
            r1, r2 = max(row - win_radius, 0), min(row + win_radius, rows)
            c1, c2 = max(col - win_radius, 0), min(col + win_radius, cols)

            # 遍历窗口内的所有像素，并添加到 set 中去重
            for rr in range(r1, r2):
                for cc in range(c1, c2):
                    unique_pixels.add((rr, cc))
        
    if len(unique_pixels) == 0:
        print("    ✗ No valid S1 pixels were found in any CS2 point neighborhood.")
        return None

    # ---------------------------------------
    # 2. 统计唯一像元
    # ---------------------------------------
    lead_total = 0
    ice_total  = 0
    refr_total = 0

    for rr, cc in unique_pixels:
        val = s1_array[rr, cc]
        if val == S1_LEAD:
            lead_total += 1
        elif val == S1_ICE:
            ice_total += 1
        elif val == S1_REFR:
            refr_total += 1

    total_pixels = lead_total + ice_total + refr_total
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


# -------------------- MAIN (使用新的采样函数) --------------------
def main():
    print("=" * 80)
    print("CS2–S1 DENSITY (ROBUST POINT-BY-POINT SAMPLING)")
    print("=" * 80)
    print(f"WINDOW_RADIUS = {WINDOW_RADIUS} pixels (480m half-width, Total width ≈ 960m)")
    print(f"CS2 Point Spacing = 300m\n")

    match_df = pd.read_csv(MATCH_CSV)
    results = []

    for idx, row in match_df.iterrows():
        print(f"\n[{idx+1}/{len(match_df)}] Scene: {row['sceneName']}")

        # --- find CS2 / S1 files (UNCHANGED) ---
        cs2_key = re.search(r"(\d{8}T\d{6}_\d{8}T\d{6})", Path(row["cs2_path"]).name)
        if not cs2_key:
            print("  ✗ Cannot extract CS2 timestamp key.")
            continue

        cs2_file = find_file_by_pattern(CS2_DIR, cs2_key.group(1), extension=".gpkg")
        s1_file  = find_file_by_pattern(S1_DIR, row["sceneName"], extension=".tif")

        if not cs2_file or not s1_file:
            print("  ✗ Required files not found.")
            continue

        print(f"  ✓ CS2: {Path(cs2_file).name}")
        print(f"  ✓ S1 : {Path(s1_file).name}")

        # --- load CS2 (UNCHANGED) ---
        cs2_gdf = gpd.read_file(cs2_file)
        class_col = detect_class_column(cs2_gdf)
        if not class_col:
            print("  ✗ No classification column found in CS2.")
            continue

        # --- compute CS2 density ---
        cs2_density = compute_cs2_density(cs2_gdf, class_col)
        if cs2_density is None:
            print("  ✗ No valid CS2 classified points.")
            continue

        # --- load S1 (UNCHANGED) ---
        try:
            with rasterio.open(s1_file) as src:
                s1_data = src.read(1)
                s1_crs = src.crs
                s1_transform = src.transform
        except Exception as e:
            print(f"  ✗ Failed to open S1 file: {e}")
            continue

        # --- reproject CS2 to S1 CRS ---
        cs2_proj = cs2_gdf.to_crs(s1_crs)

        # --- compute S1 density (使用 Point-by-Point 健壮采样) ---
        s1_density = compute_s1_density_point_sampling(
            s1_array=s1_data,
            transform=s1_transform,
            cs2_points_gdf=cs2_proj.geometry, # 传入 GeoSeries
            win_radius=WINDOW_RADIUS
        )

        if s1_density is None:
            print("  ✗ No valid S1 pixels in all neighborhoods.")
            continue

        # --- combine & compute differences ---
        combined = {
            "scene_name": row["sceneName"],
            "window_radius_px": WINDOW_RADIUS,
            "sampling_method": "Point-by-Point",
            **cs2_density,
            **s1_density,
            "diff_lead_density": cs2_density["density_CS2_lead_only"] - s1_density["density_S1_lead_only"],
            "diff_leadref_density": cs2_density["density_CS2_leadref"] - s1_density["density_S1_leadref"],
            "diff_floe_density": cs2_density["density_CS2_floe"] - s1_density["density_S1_floe"]
        }

        results.append(combined)
        print("  ✓ Density computed.")

    # --- Save summary ---
    if results:
        outcsv = os.path.join(OUTPUT_DIR, "overall_density_summary_2023_point_sampling_robust.csv")
        pd.DataFrame(results).to_csv(outcsv, index=False)
        print("\nSaved:", outcsv)
    else:
        print("\n✗ No valid results to save.")

    print("\nDone.")


if __name__ == "__main__":
    main()