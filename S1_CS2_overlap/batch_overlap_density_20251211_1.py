import os
import re
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pathlib import Path
from shapely.ops import unary_union
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# -------------------- USER CONFIG --------------------
CS2_DIR = r"C:\Users\TJ002\Desktop\CS2_S1_result\filter\2023\gpkg"
S1_DIR  = r"C:\Users\TJ002\Desktop\CS2_S1_result\filter\2023\tif"
MATCH_CSV = r"F:\NWP\CS2_S1_matched\time_match_2023_filter.csv"

OUTPUT_DIR = r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\2023_lead_refrozen_density"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Neighborhood half-window size (pixels)
WINDOW_RADIUS = 20   # ≈600–800m footprint

# S1 encoding (fixed)
S1_BACKGROUND = 0
S1_ICE = 1
S1_LEAD = 2
S1_REFR = 3

# -------------------- FILE SEARCH --------------------
def find_file_by_pattern(folder, pattern, extension=None):
    """Find file in folder whose name contains `pattern` and optionally endswith(ext)."""
    if not os.path.exists(folder):
        return None
    for file in os.listdir(folder):
        if pattern in file:
            if extension is None or file.lower().endswith(extension.lower()):
                return os.path.join(folder, file)
    return None

def detect_class_column(gdf):
    for c in ["class", "Class", "CLASS", "classification"]:
        if c in gdf.columns:
            return c
    return None
# -------------------- COMPUTE CS2 DENSITY --------------------
def compute_cs2_density(cs2_gdf, class_col):
    """Compute CS2 track lead/ice/refrozen density."""
    cs2_gdf[class_col] = cs2_gdf[class_col].str.lower().str.strip()

    count_lead      = (cs2_gdf[class_col] == "lead").sum()
    count_ice       = (cs2_gdf[class_col] == "ice").sum()
    count_refrozen  = (cs2_gdf[class_col] == "refrozen").sum()

    total = count_lead + count_ice + count_refrozen
    if total == 0:
        return None

    density_lead_only  = count_lead / total
    density_lead_ref   = (count_lead + count_refrozen) / total
    density_floe       = count_ice / total

    return {
        "count_CS2_lead": int(count_lead),
        "count_CS2_ice": int(count_ice),
        "count_CS2_refrozen": int(count_refrozen),
        "total_CS2": int(total),
        "density_CS2_lead_only": density_lead_only,
        "density_CS2_leadref": density_lead_ref,
        "density_CS2_floe": density_floe
    }

# -------------------- S1 WINDOW SAMPLING --------------------
def extract_s1_neighborhoods(s1_array, transform, cs2_points, win_radius):
    """
    Fast local-window sampling for each CS2 point.
    Equivalent to footprint-like along-track sampling.
    """
    all_pixels = []

    for point in cs2_points:
        x, y = point.x, point.y
        col, row = ~transform * (x, y)
        row, col = int(row), int(col)

        r1, r2 = max(row - win_radius, 0), min(row + win_radius, s1_array.shape[0])
        c1, c2 = max(col - win_radius, 0), min(col + win_radius, s1_array.shape[1])

        patch = s1_array[r1:r2, c1:c2]
        all_pixels.append(patch.flatten())

    if len(all_pixels) == 0:
        return None

    return np.concatenate(all_pixels)


def compute_s1_density_from_pixels(valid_pixels):
    lead_count     = (valid_pixels == S1_LEAD).sum()
    ice_count      = (valid_pixels == S1_ICE).sum()
    refrozen_count = (valid_pixels == S1_REFR).sum()
    total          = len(valid_pixels)

    return {
        "S1_lead_pixels": int(lead_count),
        "S1_ice_pixels": int(ice_count),
        "S1_refrozen_pixels": int(refrozen_count),
        "S1_total_pixels": int(total),
        "density_S1_lead_only": lead_count / total,
        "density_S1_leadref": (lead_count + refrozen_count) / total,
        "density_S1_floe": ice_count / total
    }


def compute_s1_window_density_fast(s1_array, transform, cs2_points, win_radius):
    """
    Efficient version: does NOT store all pixels, only accumulates counts.
    Avoids MemoryError.
    """
    lead_total = 0
    ice_total = 0
    refr_total = 0

    for point in cs2_points:
        x, y = point.x, point.y
        col, row = ~transform * (x, y)
        row, col = int(row), int(col)

        r1, r2 = max(row - win_radius, 0), min(row + win_radius, s1_array.shape[0])
        c1, c2 = max(col - win_radius, 0), min(col + win_radius, s1_array.shape[1])

        patch = s1_array[r1:r2, c1:c2]
        if patch.size == 0:
            continue

        # accumulate counts (no need to store pixels)
        lead_total += np.sum(patch == S1_LEAD)
        ice_total  += np.sum(patch == S1_ICE)
        refr_total += np.sum(patch == S1_REFR)

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


# -------------------- MAIN --------------------
def main():
    print("="*80)
    print("FAST CS2–S1 ALONG-TRACK DENSITY (WINDOW SAMPLING)")
    print("="*80)
    print(f"WINDOW_RADIUS = {WINDOW_RADIUS} pixels\n")

    match_df = pd.read_csv(MATCH_CSV)
    results = []

    for idx, row in match_df.iterrows():
        print(f"\n[{idx+1}/{len(match_df)}] Scene: {row['sceneName']}")

        # --- find CS2 file ---
        cs2_key = re.search(r"(\d{8}T\d{6}_\d{8}T\d{6})", Path(row["cs2_path"]).name)
        if not cs2_key:
            print("  ✗ Cannot extract CS2 timestamp key.")
            continue

        cs2_file = find_file_by_pattern(CS2_DIR, cs2_key.group(1), extension=".gpkg")
        s1_file  = find_file_by_pattern(S1_DIR, row["sceneName"], extension=".tif")

        if not cs2_file or not s1_file:
            print("  ✗ Required files not found.")
            continue

        print("  ✓ CS2:", Path(cs2_file).name)
        print("  ✓ S1 :", Path(s1_file).name)

        # --- load CS2 ---
        cs2_gdf = gpd.read_file(cs2_file)
        class_col = detect_class_column(cs2_gdf)
        if not class_col:
            print("  ✗ No classification column found.")
            continue

        # --- CS2 density ---
        cs2_density = compute_cs2_density(cs2_gdf, class_col)
        if cs2_density is None:
            print("  ✗ No valid CS2 points.")
            continue

        # --- S1 window sampling ---
        with rasterio.open(s1_file) as src:
            s1_data = src.read(1)
            s1_crs = src.crs
            s1_transform = src.transform

        cs2_proj = cs2_gdf.to_crs(s1_crs)

        raw_pixels = extract_s1_neighborhoods(
            s1_array=s1_data,
            transform=s1_transform,
            cs2_points=cs2_proj.geometry,
            win_radius=WINDOW_RADIUS
        )

        if raw_pixels is None:
            print("  ✗ No S1 pixels extracted.")
            continue

        valid_pixels = raw_pixels[raw_pixels > 0]
        if len(valid_pixels) == 0:
            print("  ✗ No valid S1 pixels.")
            continue

        s1_density = compute_s1_window_density_fast(
            s1_array=s1_data,
            transform=s1_transform,
            cs2_points=cs2_proj.geometry,
            win_radius=WINDOW_RADIUS
        )

        # --- Differences ---
        combined = {
            "scene_name": row["sceneName"],
            **cs2_density,
            **s1_density,
            "diff_lead_density": cs2_density["density_CS2_lead_only"] - s1_density["density_S1_lead_only"],
            "diff_leadref_density": cs2_density["density_CS2_leadref"] - s1_density["density_S1_leadref"],
            "diff_floe_density": cs2_density["density_CS2_floe"] - s1_density["density_S1_floe"]
        }

        results.append(combined)
        print("  ✓ Density computed.")

    # save summary
    if results:
        outcsv = os.path.join(OUTPUT_DIR, "overall_density_summary_2023_fast.csv")
        pd.DataFrame(results).to_csv(outcsv, index=False)
        print("\nSaved:", outcsv)
    else:
        print("\n✗ No results.")

    print("\nDone.")


if __name__ == "__main__":
    main()