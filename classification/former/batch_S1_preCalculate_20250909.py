# -*- coding: utf-8 -*-
# """
# Integrated script for batch processing of Sentinel-1 and SIC data.

# This script performs the following steps in a unified workflow:
# 1. Reads Sentinel-1 (HH, HV, ANGLE) and corresponding SIC (Sea Ice Concentration) images.
# 2. Finds corresponding pairs of Sentinel-1 and SIC images based on their dates.
# 3. Masks the Sentinel-1 image using a specified SIC threshold.
# 4. Calculates additional feature bands: HH*HV, HH/HV, and (HH+HV)/(HH-HV).
# 5. Exports the processed Sentinel-1 image with all the bands (original, SIC, and new features)
#    for subsequent classification procedures.
# """

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

# --- Input/Output Paths ---
S1_FOLDER = r"F:\NWP\sentinel1 gee\2023"
SIC_FOLDER = r"E:\NWP\CS2_S1_matched\SIC\n6250"
OUTPUT_DIR = r"F:\NWP\S1_processed_for_classification"

# --- File Matching ---
S1_GLOB_PATTERNS = ["*.tif", "*.tiff"]
SEARCH_NEAREST_DAYS = 0  # 0 = only same day

# --- SIC Masking Parameters ---
SIC_ICE_MIN = 15.0  # Minimum SIC value to be considered ice
SIC_ICE_MAX = 100.0 # Maximum SIC value for ice
SIC_WATER_MAX = 10.0 # Maximum SIC value to be considered water
MASK_MODE = "ice"  # "ice" or "water"
SIC_RESAMPLING = Resampling.bilinear

# --- Feature Calculation ---
# A small number to prevent division by zero
EPS = 1e-6

# --- Output Settings ---
GENERATE_QUICKLOOK = True
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== Helper Functions =====================

def extract_date_from_name(name: str):
    """Extracts a date from a filename."""
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
    """Creates an index of SIC files by date."""
    idx = {}
    for tif in glob.glob(os.path.join(sic_folder, "*.tif*")):
        d = extract_date_from_name(tif)
        if d:
            idx.setdefault(d, []).append(tif)
    for k in idx:
        idx[k].sort()
    return idx

def pick_sic_for_date(sic_index, target_date):
    """Finds the best matching SIC file for a given date."""
    if target_date in sic_index:
        return sic_index[target_date][0]
    if SEARCH_NEAREST_DAYS > 0:
        best_diff = float('inf')
        best_path = None
        for dt, paths in sic_index.items():
            diff = abs((dt - target_date).days)
            if diff <= SEARCH_NEAREST_DAYS and diff < best_diff:
                best_diff = diff
                best_path = paths[0]
        return best_path
    return None

def align_sic_to_s1(sic_path, s1_profile):
    """Reprojects and aligns a SIC raster to the S1 grid."""
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

def create_sic_mask(sic_arr):
    """Creates a boolean mask based on SIC thresholds."""
    sic_valid = (sic_arr >= 0.0) & (sic_arr <= 100.0)
    if MASK_MODE == "ice":
        keep = (sic_arr >= SIC_ICE_MIN) & (sic_arr <= SIC_ICE_MAX)
    elif MASK_MODE == "water":
        keep = sic_arr <= SIC_WATER_MAX
    else:
        raise ValueError("MASK_MODE must be 'ice' or 'water'")
    
    in_buffer = (sic_arr > SIC_WATER_MAX) & (sic_arr < SIC_ICE_MIN)
    return keep & sic_valid & (~in_buffer)

def calculate_new_bands(hh, hv):
    """Calculates new feature bands from HH and HV."""
    with np.errstate(divide='ignore', invalid='ignore'):
        hh_mult_hv = hh * hv
        
        # Ensure non-zero divisor for division
        hv_safe = np.where(np.abs(hv) < EPS, np.nan, hv)
        hh_div_hv = hh / hv_safe
        
        # Ensure non-zero divisor for sum/diff
        hh_minus_hv = hh - hv
        hh_minus_hv_safe = np.where(np.abs(hh_minus_hv) < EPS, np.nan, hh_minus_hv)
        sum_div_diff = (hh + hv) / hh_minus_hv_safe

    return {
        "HH_mult_HV": hh_mult_hv.astype(np.float32),
        "HH_div_HV": hh_div_hv.astype(np.float32),
        "sum_div_diff": sum_div_diff.astype(np.float32)
    }

def save_geotiff(out_path, data_stack, profile, band_names):
    """Saves a multi-band GeoTIFF."""
    profile.update(
        count=data_stack.shape[0],
        dtype="float32",
        nodata=np.nan,
        compress="DEFLATE",
        tiled=True,
        BIGTIFF="IF_SAFER"
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data_stack)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)

def generate_quicklook(out_png_path, hh_masked, new_bands):
    """Generates a quicklook PNG of the results."""
    try:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Display masked HH
        im0 = axes[0].imshow(hh_masked, cmap='gray', vmin=np.nanpercentile(hh_masked, 2), vmax=np.nanpercentile(hh_masked, 98))
        axes[0].set_title("Masked HH")
        axes[0].axis("off")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Display new bands
        for i, (name, band) in enumerate(new_bands.items(), 1):
            im = axes[i].imshow(band, cmap='viridis', vmin=np.nanpercentile(band, 2), vmax=np.nanpercentile(band, 98))
            axes[i].set_title(name)
            axes[i].axis("off")
            fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(out_png_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  [Quicklook Error] Could not generate quicklook: {e}")

# ===================== Main Processing Workflow =====================

def main():
    print("Indexing SIC files by date...")
    sic_index = index_sic_by_date(SIC_FOLDER)
    if not sic_index:
        print(f"[ERROR] No SIC files found in: {SIC_FOLDER}")
        return

    print("Finding Sentinel-1 files to process...")
    s1_files = sorted([f for p in S1_GLOB_PATTERNS for f in glob.glob(os.path.join(S1_FOLDER, p))])
    if not s1_files:
        print(f"[ERROR] No Sentinel-1 files found in: {S1_FOLDER}")
        return

    print(f"Found {len(s1_files)} Sentinel-1 files. Starting batch processing...")
    processed_count = 0

    for s1_path in s1_files:
        s1_name = Path(s1_path).name
        try:
            s1_date = extract_date_from_name(s1_name)
            if not s1_date:
                print(f"[SKIP] {s1_name}: Could not extract date from filename.")
                continue

            sic_path = pick_sic_for_date(sic_index, s1_date)
            if not sic_path:
                print(f"[SKIP] {s1_name}: No corresponding SIC file found for {s1_date}.")
                continue

            print(f"\nProcessing {s1_name} with {Path(sic_path).name}...")

            with rasterio.open(s1_path) as src:
                s1_profile = src.profile.copy()
                
                # Assuming bands are HH, HV, ANGLE
                hh = src.read(1).astype(np.float32) * 0.01 # GEE scale factor
                hv = src.read(2).astype(np.float32) * 0.01 # GEE scale factor
                angle = src.read(3).astype(np.float32) * 0.01

                # --- Masking ---
                sic_aligned = align_sic_to_s1(sic_path, s1_profile)
                valid_mask = create_sic_mask(sic_aligned)
                
                # Apply mask (set invalid areas to NaN)
                hh[~valid_mask] = np.nan
                hv[~valid_mask] = np.nan
                angle[~valid_mask] = np.nan
                sic_aligned[~valid_mask] = np.nan

                # --- Band Calculation ---
                new_bands = calculate_new_bands(hh, hv)
                
                # --- Prepare for Export ---
                output_bands = [
                    hh, 
                    hv, 
                    angle, 
                    sic_aligned, 
                    new_bands["HH_mult_HV"],
                    new_bands["HH_div_HV"],
                    new_bands["sum_div_diff"]
                ]
                output_band_names = [
                    "HH", "HV", "ANGLE", "SIC", 
                    "HH_mult_HV", "HH_div_HV", "sum_div_diff"
                ]

                output_stack = np.stack(output_bands, axis=0)

                # --- Save Output ---
                output_filename = f"{Path(s1_path).stem}_processed.tif"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                save_geotiff(output_path, output_stack, s1_profile, output_band_names)
                print(f"  [SUCCESS] Saved processed file to: {output_filename}")

                if GENERATE_QUICKLOOK:
                    quicklook_path = os.path.join(OUTPUT_DIR, f"{Path(s1_path).stem}_quicklook.png")
                    generate_quicklook(quicklook_path, hh, new_bands)
                    print(f"  [SUCCESS] Saved quicklook to: {Path(quicklook_path).name}")

                processed_count += 1

        except Exception as e:
            print(f"[FAIL] {s1_name}: An error occurred: {e}")

    print(f"\n--- Batch processing complete. ---")
    print(f"Successfully processed {processed_count}/{len(s1_files)} files.")


if __name__ == "__main__":
    main()