"""
CS2–S1 Overlap Density  ▸  scene × SIC bin  (Multi-year batch)
================================================================
核心设计（与旧版的关键区别）：
  旧：CS2点 → SIC tag → 分bin驱动S1采样 → S1像素再做bin过滤（双重限制，大量数据丢失）
  新：CS2点 → 驱动S1窗口采样（不做任何SIC过滤）
        → 读该窗口内 sic_aligned 均值 → 按均值分bin
        → 统计窗口内S1像素分类密度
        → 同一窗口的CS2点分类密度
        → Δdensity 挂在 S1-SIC-bin 上

SIC 数值：0–100 整数（非 0–1 浮点）
SIC bin：10 个等宽窗口 [0,10), [10,20), ..., [90,100]
"""

import os
import re
import glob
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
import numpy as np
import warnings
from datetime import datetime
from shapely.geometry import Point
from rasterio.transform import rowcol
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ========================= USER CONFIG =========================

YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,2023]
# YEARS = [2015]
WINDOW_RADIUS = 12

S1_BACKGROUND = 0
S1_ICE        = 1
S1_LEAD       = 2
S1_REFR       = 3

# ---------- SIC 配置 ----------
# SIC tif 数值范围：0–100 整数
# 按年份存放：D:\S1_CS2_data\SIC\{year}\n6250
# 示例文件名：asi-AMSR2-n6250-20230329-v5.tif
SIC_FOLDER_TEMPLATE = r"D:\S1_CS2_data\SIC\{year}\n6250"
SEARCH_NEAREST_DAYS = 1          # 找不到精确日期时，允许 ±N 天
SIC_RESAMPLING      = Resampling.nearest

# ---------- SIC bin 配置（10 个等宽窗口，0–100 整数）----------
# [0,10), [10,20), ..., [80,90), [90,100]
# SIC_BIN_EDGES  = list(range(0, 101, 10))          # [0,10,20,...,100]
# SIC_BIN_LABELS = [f"{SIC_BIN_EDGES[i]}-{SIC_BIN_EDGES[i+1]}"
#                   for i in range(len(SIC_BIN_EDGES) - 1)]
# → ["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90","90-100"]
SIC_BIN_EDGES   = [0, 60, 70, 80, 90, 100]          # 边界
SIC_BIN_LABELS = ["0-60", "60-70", "70-80", "80-90", "90-100"]   # 5 个 bin
# ---------- 输出路径 ----------
OUTPUT_ROOT = r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\201604021"

VIS_COLORS = {
    'background': 'lightgray',
    'ice':        '#1f77b4',
    'lead':       '#d62728',
    'refrozen':   '#ff7f0e',
    'hexbin_map': 'viridis'
}


# ========================= SIC 辅助函数 =========================

def get_sic_folder(year: int) -> str:
    return SIC_FOLDER_TEMPLATE.format(year=year)


def extract_date_from_name(name: str):
    """从文件名中解析第一个 YYYYMMDD 日期。"""
    base = Path(name).stem
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", base)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d).date()
        except ValueError:
            pass
    return None


def index_sic_by_date(sic_folder: str) -> dict:
    """扫描 SIC 文件夹，建立 {date: [path, ...]} 索引。"""
    idx = {}
    for tif in glob.glob(os.path.join(sic_folder, "*.tif*")):
        d = extract_date_from_name(tif)
        if d:
            idx.setdefault(d, []).append(tif)
    for k in idx:
        idx[k].sort()
    return idx


def pick_sic_for_date(sic_index: dict, target_date):
    """找与 target_date 最近的 SIC 文件（在 SEARCH_NEAREST_DAYS 范围内）。"""
    if target_date in sic_index:
        return sic_index[target_date][0]
    if SEARCH_NEAREST_DAYS > 0:
        best_diff, best_path = float("inf"), None
        for dt, paths in sic_index.items():
            diff = abs((dt - target_date).days)
            if diff <= SEARCH_NEAREST_DAYS and diff < best_diff:
                best_diff, best_path = diff, paths[0]
        return best_path
    return None


def align_sic_to_s1(sic_path: str, s1_profile: dict) -> np.ndarray:
    """
    将 SIC 栅格重投影 + 重采样到 S1 的空间参考和分辨率。
    返回与 S1 同尺寸的 float32 数组（0–100 整数值，无效→NaN）。
    """
    dst = np.full(
        (s1_profile["height"], s1_profile["width"]), np.nan, dtype=np.float32
    )
    with rasterio.open(sic_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=s1_profile["transform"],
            dst_crs=s1_profile["crs"],
            dst_nodata=np.nan,
            resampling=SIC_RESAMPLING,
        )
    return dst


def sic_value_to_bin_label(sic_val: float) -> str:
    """
    将 SIC 均值（0–100 整数尺度）映射到 bin 标签。
    NaN → "nan"；超出 [0,100] → 最近边界 bin。
    """
    if np.isnan(sic_val):
        return "nan"
    v = max(0.0, min(100.0, float(sic_val)))
    for i in range(len(SIC_BIN_EDGES) - 1):
        lo = SIC_BIN_EDGES[i]
        hi = SIC_BIN_EDGES[i + 1]
        if i == len(SIC_BIN_EDGES) - 2:   # 最后一个 bin 右闭
            if lo <= v <= hi:
                return SIC_BIN_LABELS[i]
        else:
            if lo <= v < hi:
                return SIC_BIN_LABELS[i]
    return SIC_BIN_LABELS[-1]


# ========================= 辅助函数 =========================

def find_file_by_pattern(folder, pattern, extension=None):
    if not os.path.exists(folder):
        return None
    for file in os.listdir(folder):
        if pattern in file and (
            extension is None or file.lower().endswith(extension.lower())
        ):
            return os.path.join(folder, file)
    return None


def detect_class_column(gdf):
    for c in ["class", "Class", "CLASS", "classification"]:
        if c in gdf.columns:
            return c
    return None


def plot_overlap_density_validation(
    s1_data, s1_transform, cs2_gdf_proj, scene_name, output_dir
):
    rows, cols = s1_data.shape
    xmin, ymax = s1_transform * (0, 0)
    xmax, ymin = s1_transform * (cols, rows)
    s1_extent = [xmin, xmax, ymin, ymax]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    cmap_s1 = mcolors.ListedColormap(
        [VIS_COLORS["background"], VIS_COLORS["ice"],
         VIS_COLORS["lead"], VIS_COLORS["refrozen"]]
    )
    norm_s1 = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_s1.N)
    ax.imshow(
        s1_data, extent=s1_extent, cmap=cmap_s1,
        norm=norm_s1, interpolation="nearest", alpha=0.5,
    )
    if not cs2_gdf_proj.empty:
        hb = ax.hexbin(
            cs2_gdf_proj.geometry.x, cs2_gdf_proj.geometry.y,
            gridsize=50, cmap=VIS_COLORS["hexbin_map"],
            alpha=0.7, mincnt=1, extent=s1_extent,
        )
        cb = fig.colorbar(hb, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label("CS2 Overlap Point Density (Count)", fontsize=9)
        ax.scatter(
            cs2_gdf_proj.geometry.x, cs2_gdf_proj.geometry.y,
            marker="x", color="yellow", s=5,
            label="CS2 Overlap Points", zorder=10,
        )
    ax.set_title(
        f"S1 Classification and CS2 Overlap Density\nScene: {scene_name}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    legend_s1 = [
        mpatches.Patch(color=VIS_COLORS[c], label=f"{l} ({i})")
        for i, c, l in [
            (S1_ICE, "ice", "Ice"),
            (S1_LEAD, "lead", "Lead"),
            (S1_REFR, "refrozen", "Refrozen"),
        ]
    ]
    ax.legend(
        handles=legend_s1, loc="upper left", fontsize=8,
        title="S1 Classification", framealpha=0.8,
    )
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"overlap_density_{scene_name}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"    ✓ Plot saved: {Path(plot_path).name}")


# ========================= 核心密度计算 =========================

def compute_density_per_point(
    cs2_proj_overlapped: gpd.GeoDataFrame,
    class_col: str,
    s1_array: np.ndarray,
    sic_aligned: np.ndarray,
    s1_transform,
    win_radius: int,
) -> list:
    """
    ✅ 新核心逻辑：以 CS2 点为驱动，窗口采样 S1。
    用窗口内 SIC 均值确定 bin，不对 CS2 点本身做 SIC 过滤。

    返回：每个 CS2 点一条 record，后续 groupby sic_bin 聚合。
    """
    rows_n, cols_n = s1_array.shape
    inv_tf = ~s1_transform

    records = []
    class_series = cs2_proj_overlapped[class_col].str.lower().str.strip()

    for pt, cs2_class in zip(cs2_proj_overlapped.geometry, class_series):
        if not isinstance(pt, Point):
            continue

        # 窗口边界
        c_f, r_f = inv_tf * (pt.x, pt.y)
        r_c, c_c = int(r_f), int(c_f)
        r1 = max(r_c - win_radius, 0);  r2 = min(r_c + win_radius, rows_n)
        c1 = max(c_c - win_radius, 0);  c2 = min(c_c + win_radius, cols_n)

        if r2 <= r1 or c2 <= c1:
            continue

        s1_win  = s1_array[r1:r2, c1:c2]
        sic_win = sic_aligned[r1:r2, c1:c2]

        # ✅ SIC 均值 → bin（S1驱动，CS2不额外过滤）
        sic_valid = sic_win[~np.isnan(sic_win)]
        sic_mean  = float(np.mean(sic_valid)) if sic_valid.size > 0 else np.nan
        sic_bin   = sic_value_to_bin_label(sic_mean)

        # S1 像素统计（窗口内所有非背景像素）
        lead_px  = int(np.sum(s1_win == S1_LEAD))
        ice_px   = int(np.sum(s1_win == S1_ICE))
        refr_px  = int(np.sum(s1_win == S1_REFR))
        total_px = lead_px + ice_px + refr_px

        records.append({
            "sic_bin":     sic_bin,
            "sic_mean":    round(sic_mean, 2) if not np.isnan(sic_mean) else np.nan,
            "cs2_class":   cs2_class,
            "S1_lead_px":  lead_px,
            "S1_ice_px":   ice_px,
            "S1_refr_px":  refr_px,
            "S1_total_px": total_px,
        })

    return records


def aggregate_by_bin(records: list) -> dict:
    """
    将 per-point records 按 sic_bin 聚合：
      - CS2 密度分母 = 该 bin 内所有 CS2 点数
      - S1 密度分母 = 累加 S1_total_px（非背景像素总数）
      - 无数据的 bin 不出现在结果中（不显示）
    返回：{bin_label: metrics_dict}
    """
    if not records:
        return {}

    df = pd.DataFrame(records)
    bin_results = {}

    for bin_label, grp in df.groupby("sic_bin"):
        if bin_label == "nan":
            continue

        # CS2
        n_pts          = len(grp)
        count_lead     = int((grp["cs2_class"] == "lead").sum())
        count_ice      = int((grp["cs2_class"] == "ice").sum())
        count_refrozen = int((grp["cs2_class"] == "refrozen").sum())
        count_ambi     = int((grp["cs2_class"] == "ambiguous").sum())
        total_cs2      = count_lead + count_ice + count_refrozen + count_ambi

        if total_cs2 == 0:
            continue

        # S1（只用窗口有效像素的点）
        grp_valid = grp[grp["S1_total_px"] > 0]
        if grp_valid.empty:
            continue

        S1_lead  = int(grp_valid["S1_lead_px"].sum())
        S1_ice   = int(grp_valid["S1_ice_px"].sum())
        S1_refr  = int(grp_valid["S1_refr_px"].sum())
        S1_total = int(grp_valid["S1_total_px"].sum())

        if S1_total == 0:
            continue

        bin_results[bin_label] = {
            "n_CS2_points":       n_pts,
            "count_CS2_lead":     count_lead,
            "count_CS2_ice":      count_ice,
            "count_CS2_refrozen": count_refrozen,
            "count_CS2_ambi":     count_ambi,
            "total_CS2":          total_cs2,
            "density_CS2_lead_only":  count_lead / total_cs2,
            "density_CS2_leadref":    (count_lead + count_refrozen) / total_cs2,
            "density_CS2_floe":       count_ice  / total_cs2,
            "density_CS2_floeref":    (count_ice  + count_refrozen) / total_cs2,
            "density_CS2_ambiguous":  count_ambi / total_cs2,
            "S1_lead_pixels":     S1_lead,
            "S1_ice_pixels":      S1_ice,
            "S1_refrozen_pixels": S1_refr,
            "S1_total_pixels":    S1_total,
            "density_S1_lead_only":  S1_lead / S1_total,
            "density_S1_leadref":    (S1_lead + S1_refr) / S1_total,
            "density_S1_floe":       S1_ice  / S1_total,
            "density_S1_floeref":    (S1_ice  + S1_refr) / S1_total,
        }

    return bin_results


# ========================= 单年份处理 =========================

def process_single_year(year: int, sic_index: dict) -> list:
    print("\n" + "=" * 80)
    print(f"PROCESSING YEAR: {year}")
    print("=" * 80)

    CS2_DIR    = rf"C:\Users\TJ002\Desktop\CS2_S1_result\filter1\{year}\gpkg"
    S1_DIR     = rf"C:\Users\TJ002\Desktop\CS2_S1_result\filter1\{year}\tif"
    MATCH_CSV  = rf"E:\NWP\CS2_S1_matched\time_match_{year}_filter.csv"
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f"{year}_density_SIC_binned")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MATCH_CSV):
        print(f"  ✗ Match CSV not found: {MATCH_CSV}")
        return []

    match_df = pd.read_csv(MATCH_CSV)
    results  = []

    for idx, row in match_df.iterrows():
        scene_name = row["sceneName"]
        print(f"\n  [{idx+1}/{len(match_df)}] {scene_name}")

        # ---- 文件定位 ----
        cs2_key = re.search(
            r"(\d{8}T\d{6}_\d{8}T\d{6})", Path(row["cs2_path"]).name
        )
        if not cs2_key:
            print("    ✗ Cannot parse CS2 key.")
            continue

        cs2_file = find_file_by_pattern(CS2_DIR, cs2_key.group(1), ".gpkg")
        s1_file  = find_file_by_pattern(S1_DIR,  scene_name,        ".tif")

        if not cs2_file or not s1_file:
            print("    ✗ Required files not found.")
            continue

        print(f"    CS2: {Path(cs2_file).name}")
        print(f"    S1 : {Path(s1_file).name}")

        # ---- 读取 CS2 ----
        try:
            cs2_gdf = gpd.read_file(cs2_file)
        except Exception as e:
            print(f"    ✗ CS2 read error: {e}")
            continue

        class_col = detect_class_column(cs2_gdf)
        if not class_col:
            print("    ✗ No classification column found.")
            continue

        # ---- 读取 S1 ----
        try:
            with rasterio.open(s1_file) as src:
                s1_data      = src.read(1)
                s1_crs       = src.crs
                s1_transform = src.transform
                s1_rows, s1_cols = src.shape
                s1_profile = {
                    "crs": s1_crs, "transform": s1_transform,
                    "height": s1_rows, "width": s1_cols,
                }
        except Exception as e:
            print(f"    ✗ S1 read error: {e}")
            continue

        # ---- 匹配 SIC ----
        scene_date  = extract_date_from_name(scene_name)
        sic_aligned = np.full((s1_rows, s1_cols), np.nan, dtype=np.float32)

        if scene_date is None:
            print("    ⚠ Cannot parse date from scene name — scene skipped.")
            continue

        sic_path = pick_sic_for_date(sic_index, scene_date)
        if sic_path is None:
            print("    ⚠ No SIC file matched — scene skipped.")
            continue

        print(f"    SIC: {Path(sic_path).name}")
        sic_aligned = align_sic_to_s1(sic_path, s1_profile)

        # ---- CS2 投影 + 重叠过滤 ----
        cs2_proj = cs2_gdf.to_crs(s1_crs)
        xs = cs2_proj.geometry.x.to_numpy()
        ys = cs2_proj.geometry.y.to_numpy()
        r_arr, c_arr = rowcol(s1_transform, xs, ys)
        r_arr = np.array(r_arr);  c_arr = np.array(c_arr)

        rect_mask = (
            (r_arr >= 0) & (r_arr < s1_rows) &
            (c_arr >= 0) & (c_arr < s1_cols)
        )
        s1_vals = np.full(len(cs2_gdf), S1_BACKGROUND, dtype=s1_data.dtype)
        vi = np.where(rect_mask)[0]
        if vi.size > 0:
            s1_vals[vi] = s1_data[r_arr[vi], c_arr[vi]]

        final_mask         = rect_mask & (s1_vals != S1_BACKGROUND)
        cs2_gdf_overlapped = cs2_gdf[final_mask].copy()
        n_overlap          = len(cs2_gdf_overlapped)
        print(f"    CS2 overlap points: {n_overlap}")

        if n_overlap == 0:
            print("    ✗ No CS2 points within S1 valid area.")
            continue

        cs2_proj_overlapped = cs2_gdf_overlapped.to_crs(s1_crs)

        # =====================================================
        # ✅ 新核心：per-point 采样 → aggregate by S1-SIC bin
        # =====================================================
        point_records = compute_density_per_point(
            cs2_proj_overlapped = cs2_proj_overlapped,
            class_col           = class_col,
            s1_array            = s1_data,
            sic_aligned         = sic_aligned,
            s1_transform        = s1_transform,
            win_radius          = WINDOW_RADIUS,
        )

        bin_metrics = aggregate_by_bin(point_records)

        if not bin_metrics:
            print("    ✗ No valid bins after aggregation.")
            continue

        scene_year  = scene_date.year
        scene_month = scene_date.month
        bins_found  = []

        for bin_label, m in bin_metrics.items():
            results.append({
                # 索引
                "scene_name":        scene_name,
                "year":              scene_year,
                "month":             scene_month,
                "sic_bin":           bin_label,
                "window_radius_px":  WINDOW_RADIUS,
                # 样本量
                "n_CS2_points":      m["n_CS2_points"],
                # CS2 counts
                "count_CS2_lead":     m["count_CS2_lead"],
                "count_CS2_ice":      m["count_CS2_ice"],
                "count_CS2_refrozen": m["count_CS2_refrozen"],
                "count_CS2_ambi":     m["count_CS2_ambi"],
                "total_CS2_overlap":  m["total_CS2"],
                # CS2 density
                "density_CS2_lead_only": m["density_CS2_lead_only"],
                "density_CS2_leadref":   m["density_CS2_leadref"],
                "density_CS2_floe":      m["density_CS2_floe"],
                "density_CS2_floeref":   m["density_CS2_floeref"],
                "density_CS2_ambiguous": m["density_CS2_ambiguous"],
                # S1 pixels
                "S1_lead_pixels":     m["S1_lead_pixels"],
                "S1_ice_pixels":      m["S1_ice_pixels"],
                "S1_refrozen_pixels": m["S1_refrozen_pixels"],
                "S1_total_pixels":    m["S1_total_pixels"],
                # S1 density
                "density_S1_lead_only": m["density_S1_lead_only"],
                "density_S1_leadref":   m["density_S1_leadref"],
                "density_S1_floe":      m["density_S1_floe"],
                "density_S1_floeref":   m["density_S1_floeref"],
                # Δdensity
                "diff_lead_density":    m["density_CS2_lead_only"] - m["density_S1_lead_only"],
                "diff_leadref_density": m["density_CS2_leadref"]   - m["density_S1_leadref"],
                "diff_floe_density":    m["density_CS2_floe"]      - m["density_S1_floe"],
                "diff_floeref_density": m["density_CS2_floeref"]   - m["density_S1_floeref"],
            })
            bins_found.append(bin_label)

        print(f"    ✓ Bins: {sorted(bins_found, key=lambda b: int(b.split('-')[0]))}  "
              f"({len(point_records)} point-records)")

        plot_overlap_density_validation(
            s1_data, s1_transform, cs2_proj_overlapped, scene_name, OUTPUT_DIR
        )

    # ---- 保存本年度 ----
    if results:
        outcsv = os.path.join(OUTPUT_DIR, f"density_SIC_binned_{year}.csv")
        pd.DataFrame(results).to_csv(outcsv, index=False)
        print(f"\n  ✓ Year {year} saved → {outcsv}")
    else:
        print(f"\n  ✗ No valid results for year {year}.")

    return results


# ========================= MAIN =========================

def main():
    print("=" * 80)
    print("CS2–S1 DENSITY  ▸  scene × SIC bin  (S1-driven, 10 bins, 0-100 int)")
    print("=" * 80)
    print(f"Years         : {YEARS}")
    print(f"WINDOW_RADIUS : {WINDOW_RADIUS} px")
    print(f"SIC bins (×10): {SIC_BIN_LABELS}")
    print(f"SIC template  : {SIC_FOLDER_TEMPLATE}\n")

    # ---- 逐年扫描 SIC ----
    print("Indexing SIC files per year...")
    sic_index_by_year: dict = {}
    for year in YEARS:
        folder = get_sic_folder(year)
        if os.path.isdir(folder):
            idx = index_sic_by_date(folder)
            sic_index_by_year[year] = idx
            print(f"  {year}: {len(idx)} dates  ← {folder}")
        else:
            sic_index_by_year[year] = {}
            print(f"  {year}: ✗ folder not found: {folder}")
    print()

    # ---- 逐年处理 ----
    all_results = []
    for year in YEARS:
        yr = process_single_year(year, sic_index_by_year[year])
        all_results.extend(yr)

    if not all_results:
        print("\n✗ No results across all years.")
        return

    df_all = pd.DataFrame(all_results)

    # ---- 全年明细 CSV ----
    all_csv = os.path.join(OUTPUT_ROOT, "density_SIC_binned_ALL_YEARS.csv")
    df_all.to_csv(all_csv, index=False)
    print(f"\n✓ All-year detail saved: {all_csv}")

    # ---- 聚合：mean Δdensity(year, month, sic_bin) + n_scenes + n_CS2_points ----
    group_cols   = ["year", "month", "sic_bin"]
    diff_cols    = ["diff_lead_density", "diff_leadref_density",
                    "diff_floe_density", "diff_floeref_density"]
    density_cols = ["density_CS2_lead_only", "density_CS2_leadref",
                    "density_CS2_floe",      "density_CS2_floeref",
                    "density_S1_lead_only",  "density_S1_leadref",
                    "density_S1_floe",       "density_S1_floeref"]

    agg = (
        df_all.groupby(group_cols)
        .agg(
            n_scenes     = ("scene_name",   "nunique"),
            n_CS2_points = ("n_CS2_points", "sum"),
            **{c: (c, "mean") for c in diff_cols + density_cols},
        )
        .reset_index()
    )

    # bin 按数值排序
    agg["_sort"] = agg["sic_bin"].map(lambda b: int(b.split("-")[0]))
    agg = agg.sort_values(["year", "month", "_sort"]).drop(columns="_sort")

    agg_csv = os.path.join(OUTPUT_ROOT, "delta_density_by_sic_bin_month_year.csv")
    agg.to_csv(agg_csv, index=False)
    print(f"✓ Aggregated Δdensity saved: {agg_csv}")

    # ---- 样本量汇总 ----
    print("\n--- SIC bin sample summary (all years combined) ---")
    present_bins = [b for b in SIC_BIN_LABELS if b in df_all["sic_bin"].values]
    summary = (
        df_all.groupby("sic_bin")
        .agg(n_scenes=("scene_name", "nunique"),
             n_CS2_points=("n_CS2_points", "sum"))
        .reindex(present_bins)
    )
    print(summary.to_string())
    print("\nAll Done! ✓")


if __name__ == "__main__":
    main()
