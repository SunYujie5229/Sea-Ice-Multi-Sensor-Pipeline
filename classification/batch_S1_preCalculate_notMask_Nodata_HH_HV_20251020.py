# -*- coding: utf-8 -*-
# ==============================================================
# Batch Sentinel-1 preprocessing script (Final Version: nodata = -9999)
# --------------------------------------------------------------
# 功能：
#   1. 读取 Sentinel-1 HH/HV (dB) 与对应 SIC
#   2. 以 HH≠0 区域为有效区
#   3. 有效区保留原值，掩膜区赋值为 -9999
#   4. 输出含7波段 GeoTIFF (nodata = -9999)
# ==============================================================

import os
import re
import glob
from pathlib import Path
from datetime import datetime
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import matplotlib.pyplot as plt

# ===================== Configuration =====================
S1_FOLDER = r"E:\s1_gee\2018"
# SIC_FOLDER = r"E:\NWP\CS2_S1_matched\SIC\n6250"
SIC_FOLDER = r"D:\S1_CS2_data\SIC\2018\n6250"
# OUTPUT_DIR = r"F:\NWP\S1_processed_for_classification\2023_1"
OUTPUT_DIR = r"E:\pre_processing\2018"
S1_GLOB_PATTERNS = ["*.tif", "*.tiff"]
SEARCH_NEAREST_DAYS = 0
SIC_RESAMPLING = Resampling.bilinear
EPS = 1e-6
GENERATE_QUICKLOOK = True
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== Helper Functions =====================

def extract_date_from_name(name: str):
    base = Path(name).stem
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", base)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d).date()
        except ValueError:
            pass
    return None


def index_sic_by_date(sic_folder):
    idx = {}
    for tif in glob.glob(os.path.join(sic_folder, "*.tif*")):
        d = extract_date_from_name(tif)
        if d:
            idx.setdefault(d, []).append(tif)
    for k in idx:
        idx[k].sort()
    return idx


def pick_sic_for_date(sic_index, target_date):
    if target_date in sic_index:
        return sic_index[target_date][0]
    if SEARCH_NEAREST_DAYS > 0:
        best_diff = float("inf")
        best_path = None
        for dt, paths in sic_index.items():
            diff = abs((dt - target_date).days)
            if diff <= SEARCH_NEAREST_DAYS and diff < best_diff:
                best_diff = diff
                best_path = paths[0]
        return best_path
    return None


def align_sic_to_s1(sic_path, s1_profile):
    with rasterio.open(sic_path) as src:
        dst = np.full((s1_profile["height"], s1_profile["width"]), np.nan, dtype=np.float32)
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


def calculate_new_bands(hh, hv):
    with np.errstate(divide="ignore", invalid="ignore"):
        hh_mult_hv = hh * hv
        hv_safe = np.where(np.abs(hv) < EPS, np.nan, hv)
        hh_div_hv = hh / hv_safe
        hh_minus_hv = hh - hv
        hh_minus_hv_safe = np.where(np.abs(hh_minus_hv) < EPS, np.nan, hh_minus_hv)
        sum_div_diff = (hh + hv) / hh_minus_hv_safe
    return {
        "HH_mult_HV": hh_mult_hv.astype(np.float32),
        "HH_div_HV": hh_div_hv.astype(np.float32),
        "sum_div_diff": sum_div_diff.astype(np.float32),
        "HH_minus_HV": hh_minus_hv.astype(np.float32),
    }


def save_geotiff(out_path, data_stack, profile, band_names):
    profile.update(
        count=data_stack.shape[0],
        dtype="float32",
        compress="DEFLATE",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data_stack)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def generate_quicklook(out_png_path, hh_band, new_bands, nodata_value=-9999):
    """显示主 HH 及 3 个派生特征 quicklook"""
    try:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        def _imshow(ax, arr, title):
            arr = np.where(arr == nodata_value, np.nan, arr)
            vmin, vmax = np.nanpercentile(arr, [2, 98])
            im = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        _imshow(axes[0], hh_band, "HH (masked)")
        for ax, (name, band) in zip(axes[1:], list(new_bands.items())[:3]):
            _imshow(ax, band, name)

        plt.tight_layout()
        plt.savefig(out_png_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  [Quicklook Error] {e}")


# ===================== Main Workflow =====================

def main():
    print("Indexing SIC files by date...")
    sic_index = index_sic_by_date(SIC_FOLDER)
    if not sic_index:
        print(f"[ERROR] No SIC files found in: {SIC_FOLDER}")
        return

    print("Finding Sentinel-1 files to process...")
    s1_files = sorted(
        [f for p in S1_GLOB_PATTERNS for f in glob.glob(os.path.join(S1_FOLDER, p))]
    )
    if not s1_files:
        print(f"[ERROR] No Sentinel-1 files found in: {S1_FOLDER}")
        return

    print(f"Found {len(s1_files)} Sentinel-1 files. Starting batch processing...")
    processed_count = 0
    nodata_value = -9999  # ✅ classification-friendly

    for s1_path in s1_files:
        s1_name = Path(s1_path).name
        expected_output_filename = f"{Path(s1_path).stem}_processed.tif"
        expected_output_path = os.path.join(OUTPUT_DIR, expected_output_filename)
        if os.path.exists(expected_output_path):
            print(f"[SKIP] {expected_output_filename} already exists.")
            continue

        try:
            s1_date = extract_date_from_name(s1_name)
            if not s1_date:
                print(f"[SKIP] {s1_name}: cannot extract date.")
                continue
            sic_path = pick_sic_for_date(sic_index, s1_date)
            if not sic_path:
                print(f"[SKIP] {s1_name}: no matching SIC file.")
                continue

            print(f"\nProcessing {s1_name} with {Path(sic_path).name} ...")

            with rasterio.open(s1_path) as src:
                s1_profile = src.profile.copy()

                # === 1. Read HH/HV (GEE导出 dB×0.01) ===
                hh = src.read(1).astype(np.float32) * 0.01
                hv = src.read(2).astype(np.float32) * 0.01

                # === 2. 掩膜：HH=0 为无效区 ===
                s1_valid_mask = (hh != 0)

                # === 3. 应用掩膜 ===
                hv[~s1_valid_mask] = np.nan

                sic_aligned = align_sic_to_s1(sic_path, s1_profile)
                sic_aligned[~s1_valid_mask] = np.nan

                # === 4. 派生特征 ===
                new_bands = calculate_new_bands(hh, hv)

                # === 5. 将 NaN 替换为 -9999 ===
                for arr in [hh, hv, sic_aligned] + list(new_bands.values()):
                    arr[np.isnan(arr)] = nodata_value

                # === 6. 输出 ===
                output_bands = [
                    hh,
                    hv,
                    sic_aligned,
                    new_bands["HH_mult_HV"],
                    new_bands["HH_div_HV"],
                    new_bands["sum_div_diff"],
                    new_bands["HH_minus_HV"],
                ]
                output_band_names = [
                    "HH",
                    "HV",
                    "SIC",
                    "HH_mult_HV",
                    "HH_div_HV",
                    "sum_div_diff",
                    "HH_minus_HV",
                ]
                output_stack = np.stack(output_bands, axis=0)

                s1_profile.update(
                    count=len(output_bands),
                    dtype="float32",
                    nodata=nodata_value,
                    compress="DEFLATE",
                    tiled=True,
                    BIGTIFF="IF_SAFER",
                )
                save_geotiff(expected_output_path, output_stack, s1_profile, output_band_names)
                print(f"  [SUCCESS] Saved processed file to: {expected_output_filename}")

                if GENERATE_QUICKLOOK:
                    ql_path = os.path.join(OUTPUT_DIR, f"{Path(s1_path).stem}_quicklook.png")
                    generate_quicklook(ql_path, hh, new_bands, nodata_value)
                    print(f"  [SUCCESS] Saved quicklook: {Path(ql_path).name}")

                processed_count += 1

        except Exception as e:
            print(f"[FAIL] {s1_name}: {e}")

    print(f"\n--- Batch processing complete ---")
    print(f"Successfully processed {processed_count}/{len(s1_files)} files.")


# ===================== Run =====================
if __name__ == "__main__":
    main()
