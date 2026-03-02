import os
import re
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
import numpy as np
import warnings
from shapely.geometry import Point
from rasterio.transform import rowcol 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# 忽略 geopandas/shapely 相关的警告
warnings.filterwarnings("ignore")

# -------------------- USER CONFIG --------------------
# 1. 定义年份列表 - 批量处理
YEARS = [2023]

WINDOW_RADIUS = 12 

S1_BACKGROUND = 0 # 假设背景值/NoData为0
S1_ICE = 1
S1_LEAD = 2
S1_REFR = 3

VIS_COLORS = {
    'background': 'lightgray', 
    'ice': '#1f77b4',      
    'lead': '#d62728',      
    'refrozen': '#ff7f0e',  
    'hexbin_map': 'viridis' 
}


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


def plot_overlap_density_validation(s1_data, s1_transform, cs2_gdf_proj, scene_name, output_dir):
    """
    绘制 S1 图像作为背景,并使用 Hexbin 图层叠加 CS2 点的密度。
    """
    
    # 获取 S1 图像的地理范围 (Extent)
    rows, cols = s1_data.shape
    
    # S1 的地理范围 (xmin, xmax, ymin, ymax)
    xmin, ymax = s1_transform * (0, 0)
    xmax, ymin = s1_transform * (cols, rows)
    s1_extent = [xmin, xmax, ymin, ymax]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # 1. S1 图像背景 (分类)
    cmap_s1_list = [VIS_COLORS['background'], VIS_COLORS['ice'], VIS_COLORS['lead'], VIS_COLORS['refrozen']]
    cmap_s1 = mcolors.ListedColormap(cmap_s1_list)
    # 确保边界正确,0.5以下为背景
    norm_s1 = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_s1.N) 
    
    ax.imshow(s1_data, extent=s1_extent, cmap=cmap_s1, norm=norm_s1, interpolation='nearest', alpha=0.5)
    
    # 2. CS2 Hexbin 密度图
    if not cs2_gdf_proj.empty:
        hb = ax.hexbin(
            cs2_gdf_proj.geometry.x, 
            cs2_gdf_proj.geometry.y, 
            gridsize=50,                  
            cmap=VIS_COLORS['hexbin_map'],
            alpha=0.7,
            mincnt=1,                     
            extent=s1_extent              
        )
        
        # 添加颜色条
        cb = fig.colorbar(hb, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label('CS2 Overlap Point Density (Count)', fontsize=9)
        
        # 3. 绘制原始 CS2 点 (用于精确位置检查)
        ax.scatter(cs2_gdf_proj.geometry.x, cs2_gdf_proj.geometry.y, 
                   marker='x', color='yellow', s=5, label='CS2 Overlap Points (Location)', zorder=10)

    
    ax.set_title(f"S1 Classification and CS2 Overlap Density\nScene: {scene_name}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Easting / Longitude (meters or degrees)")
    ax.set_ylabel("Northing / Latitude (meters or degrees)")

    # 创建 S1 分类图例
    legend_s1 = [mpatches.Patch(color=VIS_COLORS[c], label=f'{l} ({i})') for i, c, l in 
                 [(S1_ICE, 'ice', 'Ice'), (S1_LEAD, 'lead', 'Lead'), (S1_REFR, 'refrozen', 'Refrozen')]]
    ax.legend(handles=legend_s1, loc='upper left', fontsize=8, title="S1 Classification", framealpha=0.8)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"overlap_density_{scene_name}.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"  ✓ Overall density plot saved: {plot_path}")


# -------------------- 密度计算函数 --------------------

def compute_cs2_density(cs2_gdf, class_col):
    """计算 CS2 密度 (针对重叠区域)。"""
    if cs2_gdf.empty:
         return {
            "count_CS2_lead": 0, "count_CS2_ice": 0, "count_CS2_refrozen": 0, "total_CS2": 0,
            "density_CS2_lead_only": 0.0, "density_CS2_leadref": 0.0, 
            "density_CS2_floe": 0.0, "density_CS2_floeref": 0.0
        }
    cs2_gdf[class_col] = cs2_gdf[class_col].str.lower().str.strip()
    count_lead     = (cs2_gdf[class_col] == "lead").sum()
    count_ice      = (cs2_gdf[class_col] == "ice").sum()
    count_refrozen = (cs2_gdf[class_col] == "refrozen").sum()
    total = count_lead + count_ice + count_refrozen
    if total == 0: return None
    return {
        "count_CS2_lead": int(count_lead), 
        "count_CS2_ice": int(count_ice), 
        "count_CS2_refrozen": int(count_refrozen), 
        "total_CS2": int(total),
        "density_CS2_lead_only": count_lead / total, 
        "density_CS2_leadref": (count_lead + count_refrozen) / total, 
        "density_CS2_floe": count_ice / total,
        "density_CS2_floeref": (count_ice + count_refrozen) / total
    }


def compute_s1_density_point_sampling(s1_array, transform, cs2_points_gdf, win_radius=WINDOW_RADIUS):
    """
    点对点窗口采样:返回密度统计和唯一的采样像素集合。
    """
    rows, cols = s1_array.shape
    inv_transform = ~transform 
    unique_pixels = set()
    
    points_to_process = cs2_points_gdf.geometry if isinstance(cs2_points_gdf, gpd.GeoDataFrame) else cs2_points_gdf

    if points_to_process.empty: return None

    # ---------------------------------------
    # 1. 遍历所有 CS2 点,进行窗口采样
    # ---------------------------------------
    for pt in points_to_process:
        if not isinstance(pt, Point): continue 
            
        x, y = pt.x, pt.y
        col, row = inv_transform * (x, y)
        row, col = int(row), int(col)

        # 核心逻辑:定义窗口并收集像素
        # 注意:此处不再需要边界检查,因为主函数已经过滤掉了位于 S1 矩形之外和 S1 NoData 区的 CS2 点
        
        r1, r2 = max(row - win_radius, 0), min(row + win_radius, rows)
        c1, c2 = max(col - win_radius, 0), min(col + win_radius, cols)

        for rr in range(r1, r2):
            for cc in range(c1, c2):
                unique_pixels.add((rr, cc))
        
    if len(unique_pixels) == 0: return None

    # ---------------------------------------
    # 2. 统计唯一像元
    # ---------------------------------------
    lead_total = 0; ice_total  = 0; refr_total = 0
    total_pixels = 0

    for rr, cc in unique_pixels:
        val = s1_array[rr, cc]
        if val == S1_LEAD: lead_total += 1
        elif val == S1_ICE: ice_total += 1
        elif val == S1_REFR: refr_total += 1
        
        if val != S1_BACKGROUND:
            total_pixels += 1

    if total_pixels == 0: return None 

    density_results = {
        "S1_lead_pixels": int(lead_total), 
        "S1_ice_pixels": int(ice_total), 
        "S1_refrozen_pixels": int(refr_total), 
        "S1_total_pixels": int(total_pixels),
        "density_S1_lead_only": lead_total / total_pixels, 
        "density_S1_leadref": (lead_total + refr_total) / total_pixels, 
        "density_S1_floe": ice_total / total_pixels,
        "density_S1_floeref": (ice_total + refr_total) / total_pixels
    }
    
    # 仅返回密度统计和 unique_pixels (虽然 unique_pixels 未被可视化函数使用,但保留以备将来使用)
    return density_results, unique_pixels


# -------------------- 单年份处理函数 --------------------
def process_single_year(year):
    """处理单个年份的数据"""
    print("\n" + "=" * 80)
    print(f"PROCESSING YEAR: {year}")
    print("=" * 80)
    
    # 定义该年份的路径
    CS2_DIR = rf"C:\Users\TJ002\Desktop\CS2_S1_result\filter1\{year}\gpkg"
    S1_DIR  = rf"C:\Users\TJ002\Desktop\CS2_S1_result\filter1\{year}\tif"
    MATCH_CSV = rf"F:\NWP\CS2_S1_matched\time_match_{year}_filter.csv"
    OUTPUT_DIR = rf"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\20160112\{year}_density_simplest_filter"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 检查CSV文件是否存在
    if not os.path.exists(MATCH_CSV):
        print(f"  ✗ Match CSV not found: {MATCH_CSV}")
        return []
    
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
        
        rows_list, cols_list = rowcol(s1_transform, x_coords, y_coords)
        
        # 3. 转换为 NumPy array
        rows_arr = np.array(rows_list)
        cols_arr = np.array(cols_list)
        
        # 4. 构造矩形边界过滤器 (确保在画布内)
        row_mask = (rows_arr >= 0) & (rows_arr < s1_rows)
        col_mask = (cols_arr >= 0) & (cols_arr < s1_cols)
        rect_mask = row_mask & col_mask
        
        # 5. 【关键】提取 S1 像素值,并加入有效值过滤器
        # 提取在矩形画布内的点的 S1 像素值
        valid_indices = np.where(rect_mask)[0]
        s1_values = np.full(total_cs2_pre_filter, S1_BACKGROUND, dtype=s1_data.dtype) # 默认背景值
        
        if valid_indices.size > 0:
            s1_values[valid_indices] = s1_data[rows_arr[valid_indices], cols_arr[valid_indices]]
            
        # 6. 最终过滤器:必须在矩形画布内 AND S1 值必须是非背景值 (即有效分类)
        value_mask = (s1_values != S1_BACKGROUND)
        final_filter_mask = rect_mask & value_mask

        # 7. 应用过滤器获取重叠点 (CS2 Overlap Points)
        cs2_gdf_overlapped = cs2_gdf[final_filter_mask].copy() 
        
        total_cs2_overlap = len(cs2_gdf_overlapped)
        print(f"  ℹ️ CS2 Points (Overlap Only): {total_cs2_overlap}")
        
        # --- compute CS2 density (使用过滤后的数据) ---
        cs2_density = compute_cs2_density(cs2_gdf_overlapped, class_col)
        if cs2_density is None or cs2_density["total_CS2"] == 0:
            print("  ✗ No valid CS2 classified points within the S1 overlap.")
            continue

        # --- reproject CS2 (overlapped points only) to S1 CRS ---
        cs2_proj_overlapped = cs2_gdf_overlapped.to_crs(s1_crs)

        # --- compute S1 density ---
        s1_results = compute_s1_density_point_sampling(
            s1_array=s1_data,
            transform=s1_transform,
            cs2_points_gdf=cs2_proj_overlapped.geometry,
            win_radius=WINDOW_RADIUS
        )
        
        if s1_results is None:
            print("  ✗ No valid S1 pixels were found by the clipped CS2 points.")
            continue
            
        # 解包 S1 结果
        s1_density, unique_pixels = s1_results
        
        # --- 【新增可视化】Buffer 整体验证 ---
        plot_overlap_density_validation(
            s1_data, 
            s1_transform,
            cs2_proj_overlapped, 
            row["sceneName"],
            OUTPUT_DIR
        )

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
            "density_CS2_floeref": cs2_density["density_CS2_floeref"],
            
            # S1 详细分类数据 (采样区域)
            "S1_lead_pixels": s1_density["S1_lead_pixels"],
            "S1_ice_pixels": s1_density["S1_ice_pixels"],
            "S1_refrozen_pixels": s1_density["S1_refrozen_pixels"],
            "S1_total_pixels": s1_density["S1_total_pixels"],

            # 密度结果 (S1)
            "density_S1_lead_only": s1_density["density_S1_lead_only"],
            "density_S1_leadref": s1_density["density_S1_leadref"],
            "density_S1_floe": s1_density["density_S1_floe"],
            "density_S1_floeref": s1_density["density_S1_floeref"],

            # 差异
            "diff_lead_density": cs2_density["density_CS2_lead_only"] - s1_density["density_S1_lead_only"],
            "diff_leadref_density": cs2_density["density_CS2_leadref"] - s1_density["density_S1_leadref"],
            "diff_floe_density": cs2_density["density_CS2_floe"] - s1_density["density_S1_floe"],
            "diff_floeref_density": cs2_density["density_CS2_floeref"] - s1_density["density_S1_floeref"]
        }

        results.append(combined)
        print(f"  ✓ Density computed. CS2 Overlap Points: {total_cs2_overlap}")

    # --- Save summary for this year ---
    if results:
        outcsv = os.path.join(OUTPUT_DIR, f"overall_density_summary_{year}_raster_filtered_detailed.csv")
        pd.DataFrame(results).to_csv(outcsv, index=False)
        print(f"\n✓ Saved: {outcsv}")
    else:
        print(f"\n✗ No valid results to save for year {year}.")
    
    return results


# -------------------- MAIN 执行函数 --------------------
def main():
    print("=" * 80)
    print("CS2–S1 DENSITY CALCULATION SCRIPT (MULTI-YEAR BATCH PROCESSING)")
    print("=" * 80)
    print(f"Processing years: {YEARS}")
    print(f"WINDOW_RADIUS = {WINDOW_RADIUS} pixels (480m half-width)\n")

    all_year_results = {}
    
    for year in YEARS:
        results = process_single_year(year)
        all_year_results[year] = results
    
    # 输出总结
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    for year, results in all_year_results.items():
        print(f"Year {year}: {len(results)} scenes processed")
    
    print("\nAll Done! ✓")


if __name__ == "__main__":
    main()