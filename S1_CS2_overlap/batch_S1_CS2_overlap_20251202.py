import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from pyproj import CRS
import warnings
warnings.filterwarnings('ignore')
import re

# -------------------- USER CONFIG --------------------
BASE_DIR = r"C:\Users\TJ002\Desktop\CS2_S1_result\filter"   # 下面有 2015/2016/.../2021/2022 等
MATCH_BASE_DIR = r"F:\NWP\CS2_S1_matched"                  # 里面有 time_match_2015_filter.csv 等
OUTPUT_DIR = r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap"
# 要处理的年份列表（先从 2021、2022 测试，后面想加哪年就在这里加）
YEARS = [2021, 2022]

# 测试模式：True 时，每个年份只处理前 TEST_LIMIT 个 pair，确认逻辑无误后改成 False 全量跑
TEST_MODE = False
TEST_LIMIT = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ICE_MATCH_OPTION:
#   1: ice -> S1 value 1 only
#   2: ice -> S1 value 1 or 3 (including refrozen)
ICE_MATCH_OPTION = 1

# LEAD_MATCH_OPTION:
#   1: lead -> S1 value 2 only
#   2: lead -> S1 value 2 or 3 (including refrozen)
LEAD_MATCH_OPTION = 2

COLORS = {
    'ice': '#6BAED6', 'lead': '#FD8D3C', 'refrozen': '#BCBDDC',
    'background': '#F0F0F0', 'match_cmap': 'Greens', 'mismatch_cmap': 'Reds'
}

# -------------------- HELPER FUNCTIONS --------------------

# --- MODIFIED --- This function now accepts an optional 'extension' argument for more specific searches
def find_file_by_pattern(folder, pattern, extension=None):
    """Find file in folder that contains the pattern and optionally ends with a specific extension."""
    if not os.path.exists(folder):
        return None
    for file in os.listdir(folder):
        # Check if the main part of the name matches
        pattern_match = pattern in file
        
        # Check for the file extension if one is provided
        extension_match = True  # Assume it matches if no extension is specified
        if extension:
            extension_match = file.lower().endswith(extension.lower())
        
        if pattern_match and extension_match:
            return os.path.join(folder, file)
    return None
def extract_cs2_timestamp_key_from_nc(cs2_nc_path: str):
    """
    从 CSV 中的 cs2_path（.nc 文件）提取关键时间串：
    例如：CS_LTA__SIR_SIN_2__20150203T173542_20150203T173644_E001.nc
    → 返回 '20150203T173542_20150203T173644'
    """
    fname = Path(cs2_nc_path).name
    m = re.search(r'(\d{8}T\d{6}_\d{8}T\d{6})', fname)
    return m.group(1) if m else None


def find_s1_and_cs2_files_for_row(year_s1_dir, year_cs2_dir, row):
    """
    对某一行 match_csv 的记录，在该年份的 tif/ 与 gpkg/ 中找到对应文件。
    依赖：
      - sceneName → S1 文件前缀
      - cs2_path (.nc) → 提取时间 key 再匹配 .gpkg
    """
    scene_name = row['sceneName']
    cs2_nc_path = row['cs2_path']

    # S1: 用 sceneName 在 tif/ 中匹配 .tif
    s1_file = find_file_by_pattern(year_s1_dir, scene_name, extension=".tif")

    # CS2: 从 nc 中抽时间段 key，再在 gpkg/ 中匹配
    ts_key = extract_cs2_timestamp_key_from_nc(cs2_nc_path)
    cs2_file = None
    if ts_key:
        cs2_file = find_file_by_pattern(year_cs2_dir, ts_key, extension=".gpkg")

    return s1_file, cs2_file

def check_and_reproject_crs(gdf, target_crs):
    """Check CRS and reproject if necessary."""
    if gdf.crs is None: raise ValueError("Input GeoDataFrame has no CRS defined.")
    if gdf.crs != target_crs:
        print(f"  Reprojecting from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)
    return gdf

def extract_cs2_values_at_points(cs2_gdf, s1_path):
    """Extract S1 raster values at CS2 point locations."""
    with rasterio.open(s1_path) as src:
        coords = [(x, y) for x, y in zip(cs2_gdf.geometry.x, cs2_gdf.geometry.y)]
        s1_values = [list(src.sample([coord]))[0][0] for coord in coords]
        return np.array(s1_values)

def classify_match(cs2_class, s1_value, ice_option=1, lead_option=1):
    """Determine if CS2 and S1 classifications match."""
    if pd.isna(s1_value) or s1_value == 0: return ('no_data', False)
    cs2_class_lower = str(cs2_class).lower().strip()
    
    if cs2_class_lower == 'ice':
        match_values = [1] if ice_option == 1 else [1, 3]
        is_correct = (s1_value in match_values)
        match_type = 'ice_match' if is_correct else 'ice_mismatch'
    elif cs2_class_lower == 'lead':
        match_values = [2] if lead_option == 1 else [2, 3]
        is_correct = (s1_value in match_values)
        match_type = 'lead_match' if is_correct else 'lead_mismatch'
    else: return ('unknown', False)
    return (match_type, is_correct)

# --- MODIFIED --- This version fixes the visualization KeyError
def analyze_pair(cs2_path, s1_path, ice_option=1, lead_option=1):
    """Analyze a single CS2-S1 pair."""
    try:
        cs2_gdf = gpd.read_file(cs2_path)
        class_col = next((c for c in ['class', 'Class', 'CLASS', 'classification'] if c in cs2_gdf.columns), None)
        if not class_col:
            raise ValueError("Classification column not found in CS2 file.")

        with rasterio.open(s1_path) as src:
            s1_crs, s1_bounds = src.crs, src.bounds

        cs2_gdf = check_and_reproject_crs(cs2_gdf, s1_crs)
        cs2_gdf = cs2_gdf.cx[s1_bounds.left:s1_bounds.right, s1_bounds.bottom:s1_bounds.top].reset_index(drop=True)

        if cs2_gdf.empty:
            print("  Warning: No CS2 points within S1 bounds")
            return None

        print(f"  CS2 points in S1 region: {len(cs2_gdf)}")

        cs2_gdf['s1_value'] = extract_cs2_values_at_points(cs2_gdf, s1_path)

        # --- FIX --- Unpack results into two lists before assignment
        # This is a more robust method that prevents pandas data type errors.
        match_types = []
        is_correct_list = []
        for _, row in cs2_gdf.iterrows():
            match_type, is_correct = classify_match(row[class_col], row['s1_value'], ice_option, lead_option)
            match_types.append(match_type)
            is_correct_list.append(is_correct)
        
        cs2_gdf['match_type'] = match_types
        cs2_gdf['is_correct'] = is_correct_list
        # --- END FIX ---

        valid_gdf = cs2_gdf[cs2_gdf['match_type'] != 'no_data'].copy()
        valid_gdf['class_lower'] = valid_gdf[class_col].str.lower()

        stats = {
            'total_points': len(cs2_gdf), 'valid_points': len(valid_gdf),
            'correct_points': int(valid_gdf['is_correct'].sum()),
        }
        stats['accuracy'] = stats['correct_points'] / stats['valid_points'] if stats['valid_points'] > 0 else 0

        # Calculate confusion matrix elements and metrics for both classes
        for class_name in ['lead', 'ice']:
            if class_name == 'lead':
                s1_positive_values = [2] if lead_option == 1 else [2, 3]
            else:  # class_name == 'ice'
                s1_positive_values = [1] if ice_option == 1 else [1, 3]

            tp = len(valid_gdf[(valid_gdf['class_lower'] == class_name) & (valid_gdf['is_correct'])])
            fp = len(valid_gdf[(valid_gdf['class_lower'] != class_name) & (valid_gdf['s1_value'].isin(s1_positive_values))])
            fn = len(valid_gdf[(valid_gdf['class_lower'] == class_name) & (~valid_gdf['is_correct'])])

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            stats[f'{class_name}_tp'] = tp
            stats[f'{class_name}_fp'] = fp
            stats[f'{class_name}_fn'] = fn
            stats[f'{class_name}_precision'] = precision
            stats[f'{class_name}_recall'] = recall
            stats[f'{class_name}_f1_score'] = f1_score

        stats['results_gdf'] = cs2_gdf
        return stats

    except Exception as e:
        print(f"  Error analyzing pair: {e}")
        return None
# --- MODIFIED --- Visualization text now shows stats for both ice and lead
def create_visualization(s1_path, stats, output_path):
    """Create visualization of CS2-S1 overlap using hexbin density plots."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1.2]})
        
        with rasterio.open(s1_path) as src:
            s1_data, s1_extent = src.read(1), [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        # Left subplot: S1 Classification
        cmap_s1 = plt.matplotlib.colors.ListedColormap([COLORS[c] for c in ['background', 'ice', 'lead', 'refrozen']])
        ax1.imshow(s1_data, extent=s1_extent, cmap=cmap_s1, vmin=0, vmax=3, interpolation='nearest')
        ax1.set_title('S1 RF Classification', fontsize=12, fontweight='bold')
        legend_s1 = [mpatches.Patch(color=c, label=l) for l, c in 
                     {'Ice (1)': COLORS['ice'], 'Lead (2)': COLORS['lead'], 'Refrozen (3)': COLORS['refrozen']}.items()]
        ax1.legend(handles=legend_s1, loc='upper right', fontsize=9)
        
        # Right subplot: Hexbin Overlay
        ax2.imshow(s1_data, extent=s1_extent, cmap=cmap_s1, vmin=0, vmax=3, interpolation='nearest', alpha=0.4)
        valid_gdf = stats['results_gdf'][stats['results_gdf']['match_type'] != 'no_data']
        correct_gdf = valid_gdf[valid_gdf['is_correct']]
        incorrect_gdf = valid_gdf[~valid_gdf['is_correct']]
        gridsize = 50 
        
        if not incorrect_gdf.empty:
            hb_mis = ax2.hexbin(incorrect_gdf.geometry.x, incorrect_gdf.geometry.y, gridsize=gridsize, cmap=COLORS['mismatch_cmap'], alpha=0.7, mincnt=1)
            cb1 = fig.colorbar(hb_mis, ax=ax2, shrink=0.6, pad=0.02); cb1.set_label('Mismatch Density', fontsize=9)
        if not correct_gdf.empty:
            hb_mat = ax2.hexbin(correct_gdf.geometry.x, correct_gdf.geometry.y, gridsize=gridsize, cmap=COLORS['match_cmap'], alpha=0.7, mincnt=1)
            cb2 = fig.colorbar(hb_mat, ax=ax2, shrink=0.6, pad=0.02); cb2.set_label('Match Density', fontsize=9)

        ax2.set_title(f"CS2-S1 Overlap Density (Overall Accuracy: {stats['accuracy']:.1%})", fontsize=12, fontweight='bold')
        
        # --- MODIFIED --- Stats text now includes both classes
        stats_text = (f"--- Lead Class ---\n"
                      f"  Precision: {stats['lead_precision']:.2f}\n"
                      f"  Recall:    : {stats['lead_recall']:.2f}\n"
                      f"  F1-Score  : {stats['lead_f1_score']:.2f}\n\n"
                      f"--- Ice Class ---\n"
                      f"  Precision: {stats['ice_precision']:.2f}\n"
                      f"  Recall     : {stats['ice_recall']:.2f}\n"
                      f"  F1-Score   : {stats['ice_f1_score']:.2f}")
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
        
        plt.tight_layout(); plt.savefig(output_path, dpi=200, bbox_inches='tight'); plt.close()
        print(f"  Visualization saved: {output_path}")
    except Exception as e: print(f"  Error creating visualization: {e}")

# -------------------- MULTI-YEAR MAIN PROCESSING --------------------

def extract_year_from_scene(scene_name):
    """
    从 S1 sceneName 中提取年份 (前两个 timestamp 的前4位)
    示例：S1A_EW_GRDM_1SDH_20150203T190336_... → 2015
    """
    match = re.search(r'(\d{8})T', scene_name)
    if match:
        return int(match.group(1)[:4])
    return None


def extract_cs2_timestamp_key(cs2_nc_path):
    """
    从 CSV 中的 cs2_path (nc 文件) 提取关键时间段：
    E.g., CS_LTA__SIR_SIN_2__20150203T173542_20150203T173644_E001.nc
    → '20150203T173542_20150203T173644'
    """
    fname = Path(cs2_nc_path).name
    match = re.search(r'(\d{8}T\d{6}_\d{8}T\d{6})', fname)
    return match.group(1) if match else None


def find_s1_file_in_year(year_tif_dir, scene_name):
    """
    在该年份的 tif/ 中查找以 sceneName 开头的 S1 文件
    """
    for f in os.listdir(year_tif_dir):
        if f.startswith(scene_name) and f.lower().endswith(".tif"):
            return os.path.join(year_tif_dir, f)
    return None


def find_cs2_file_in_year(year_gpkg_dir, cs2_timestamp_key):
    """
    在该年份 gpkg/ 中根据时间段关键字匹配 CS2 GPKG 文件
    """
    for f in os.listdir(year_gpkg_dir):
        if cs2_timestamp_key in f and f.lower().endswith(".gpkg"):
            return os.path.join(year_gpkg_dir, f)
    return None


# -------------------- MULTI-YEAR MAIN PROCESSING --------------------

def main():
    print("=" * 100)
    print("Multi-Year CS2–S1 Overlap Analysis")
    print("=" * 100)
    print(f"Years to process: {YEARS}")
    print(f"Ice match option : {ICE_MATCH_OPTION} ({'1' if ICE_MATCH_OPTION == 1 else '1,3'})")
    print(f"Lead match option: {LEAD_MATCH_OPTION} ({'2' if LEAD_MATCH_OPTION == 1 else '2,3'})")
    print(f"TEST_MODE = {TEST_MODE}, TEST_LIMIT = {TEST_LIMIT}")
    print("=" * 100)

    for year in YEARS:
        print(f"\n\n==================== Year {year} ====================")

        # 1. 按年份组织路径
        year_str = str(year)
        year_s1_dir = os.path.join(BASE_DIR, year_str, "tif")
        year_cs2_dir = os.path.join(BASE_DIR, year_str, "gpkg")
        year_out_dir = os.path.join(OUTPUT_DIR, year_str)
        os.makedirs(year_out_dir, exist_ok=True)

        # 对应年份的 match_csv：优先用 *_filter.csv，若没有则退到 time_match_YYYY.csv
        csv_filter = os.path.join(MATCH_BASE_DIR, f"time_match_{year_str}_filter.csv")
        csv_raw    = os.path.join(MATCH_BASE_DIR, f"time_match_{year_str}.csv")

        if os.path.exists(csv_filter):
            match_csv = csv_filter
        elif os.path.exists(csv_raw):
            match_csv = csv_raw
        else:
            print(f"  ✗ No match CSV found for {year} (expected {csv_filter} or {csv_raw})")
            continue

        if not os.path.exists(year_s1_dir):
            print(f"  ✗ S1 directory not found: {year_s1_dir}")
            continue
        if not os.path.exists(year_cs2_dir):
            print(f"  ✗ CS2 directory not found: {year_cs2_dir}")
            continue

        print(f"  Using CSV : {match_csv}")
        print(f"  S1 folder : {year_s1_dir}")
        print(f"  CS2 folder: {year_cs2_dir}")
        print(f"  Output to : {year_out_dir}")

        # 2. 读取该年份的 match_csv
        match_df = pd.read_csv(match_csv)

        # 测试模式：只处理前 TEST_LIMIT 条
        if TEST_MODE:
            match_df = match_df.head(TEST_LIMIT)
            print(f"  TEST_MODE ON → Only first {len(match_df)} pairs will be processed")

        if match_df.empty:
            print("  ✗ No pairs in CSV after filtering.")
            continue

        all_stats = []

        # 3. 遍历该年份的每一条匹配记录
        for idx, row in match_df.iterrows():
            scene_name = row['sceneName']
            print(f"\n  [{idx + 1}/{len(match_df)}] Scene: {scene_name}")

            # 在对应年份的 tif/gpkg 中找到实际文件
            s1_file, cs2_file = find_s1_and_cs2_files_for_row(year_s1_dir, year_cs2_dir, row)

            if not s1_file:
                print("    ✗ S1 tif file not found for this scene.")
                continue
            if not cs2_file:
                print("    ✗ CS2 gpkg file not found for this scene.")
                continue

            print(f"    ✓ S1 : {Path(s1_file).name}")
            print(f"    ✓ CS2: {Path(cs2_file).name}")

            # 4. 分析单个 pair
            stats = analyze_pair(cs2_file, s1_file, ICE_MATCH_OPTION, LEAD_MATCH_OPTION)
            if not stats:
                print("    ✗ analyze_pair failed, skip.")
                continue

            # 5. 生成可视化
            fig_name = f"{Path(s1_file).stem}_overlap.png"
            fig_path = os.path.join(year_out_dir, fig_name)
            create_visualization(s1_file, stats, fig_path)

            # 6. 保存文本统计（和你原来的类似，但按年份目录区分）
            stats_path = os.path.join(year_out_dir, f"{Path(s1_file).stem}_statistics.txt")
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write(f"Pair Statistics (Year {year}): {scene_name}\n" + "="*60 + "\n")
                f.write(f"Overall Accuracy: {stats['accuracy']:.2%}\n")
                f.write(f"Valid Points: {stats['valid_points']}/{stats['total_points']}\n\n")
                for class_name in ['ice', 'lead']:
                    f.write(f"--- {class_name.capitalize()} Class Performance ---\n")
                    f.write(f"Precision: {stats[f'{class_name}_precision']:.4f}\n")
                    f.write(f"Recall (TPR): {stats[f'{class_name}_recall']:.4f}\n")
                    f.write(f"F1-Score: {stats[f'{class_name}_f1_score']:.4f}\n")
                    f.write(f"  TP: {stats[f'{class_name}_tp']} | FP: {stats[f'{class_name}_fp']} | FN: {stats[f'{class_name}_fn']}\n\n")

            print(f"    ✓ Visualization saved: {Path(fig_path).name}")
            print(f"    ✓ Statistics saved: {Path(stats_path).name}")

            # 7. 汇总信息
            stats['scene_name'] = scene_name
            del stats['results_gdf']
            all_stats.append(stats)
            print(f"    ✓ Pair done - Accuracy: {stats['accuracy']:.1%}")

        # 8. 该年份的 overall summary
        if not all_stats:
            print(f"\n  ✗ No valid pairs processed for {year}.")
            continue

        summary_df = pd.DataFrame(all_stats)
        summary_csv = os.path.join(year_out_dir, f"overall_summary_{year}.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n  ✓ Year {year} summary saved: {summary_csv}")

        # 打印总体指标
        total_valid = summary_df['valid_points'].sum()
        total_correct = summary_df['correct_points'].sum()
        overall_accuracy = total_correct / total_valid if total_valid > 0 else 0
        print(f"  Year {year} Overall Accuracy: {overall_accuracy:.2%}")

        for class_name in ['ice', 'lead']:
            total_tp = summary_df[f'{class_name}_tp'].sum()
            total_fp = summary_df[f'{class_name}_fp'].sum()
            total_fn = summary_df[f'{class_name}_fn'].sum()

            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"    ├─ {class_name.capitalize()} Precision: {precision:.4f}")
            print(f"    ├─ {class_name.capitalize()} Recall   : {recall:.4f}")
            print(f"    └─ {class_name.capitalize()} F1-Score : {f1:.4f}")

    print("\nAll years finished. 🎉")


if __name__ == "__main__":
    main()
