import os
import re
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from pathlib import Path
from shapely.ops import unary_union
import warnings
warnings.filterwarnings("ignore")

# -------------------- USER CONFIG --------------------
CS2_DIR = r"C:\Users\TJ002\Desktop\CS2_S1_result\filter\2023\gpkg\CS_OFFL_SIR_SIN_1B_20230424T135429_20230424T135644_E001_classified.gpkg"
S1_DIR  = r"C:\Users\TJ002\Desktop\CS2_S1_result\filter\2023\tif\S1A_EW_GRDM_1SDH_20230424T121731_20230424T121835_048238_05CCE4_C830_EW_HH_HV_int16x100_nodata-9999-0000000000-0000000000_processed_classified.tif"
MATCH_CSV = r"F:\NWP\CS2_S1_matched\time_match_2023_filter.csv"

OUTPUT_DIR = r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\2023_lead_refrozen_density"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# track buffer distance (meters) - adjustable
TRACK_BUFFER_METERS = 500

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


# -------------------- S1 TRACK DENSITY --------------------
def compute_s1_track_density(s1_path, cs2_gdf, buffer_meters=500):
    """Extract S1 pixels inside CS2 track buffer region, compute S1 densities."""
    with rasterio.open(s1_path) as src:
        s1_crs = src.crs

        # Project CS2 onto S1 CRS
        cs2_proj = cs2_gdf.to_crs(s1_crs)

        # Build CS2 track line
        track_line = unary_union(cs2_proj.geometry)

        # Buffer (meters)
        track_buffer = track_line.buffer(buffer_meters)

        # Mask S1
        s1_clip, _ = mask(src, [track_buffer], nodata=0, crop=True)
        data = s1_clip[0]

    # Valid pixels > 0
    valid = data[data > 0]
    if len(valid) == 0:
        return None

    lead_count     = (valid == S1_LEAD).sum()
    ice_count      = (valid == S1_ICE).sum()
    refrozen_count = (valid == S1_REFR).sum()
    total          = len(valid)

    density_lead_only = lead_count / total
    density_leadref   = (lead_count + refrozen_count) / total
    density_floe      = ice_count / total

    return {
        "S1_lead_pixels": int(lead_count),
        "S1_ice_pixels": int(ice_count),
        "S1_refrozen_pixels": int(refrozen_count),
        "S1_total_pixels": int(total),
        "density_S1_lead_only": density_lead_only,
        "density_S1_leadref": density_leadref,
        "density_S1_floe": density_floe
    }


# -------------------- MAIN --------------------
def main():
    print("="*80)
    print("Computing CS2–S1 Along-Track Density (Lead / Lead+Refrozen / Floe)")
    print("="*80)
    print(f"TRACK_BUFFER_METERS = {TRACK_BUFFER_METERS} m")

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

        print(f"  ✓ CS2 file: {Path(cs2_file).name}")
        print(f"  ✓ S1  file: {Path(s1_file).name}")

        # --- load CS2 ---
        cs2_gdf = gpd.read_file(cs2_file)
        class_col = next((c for c in ["class", "Class", "CLASS", "classification"] if c in cs2_gdf.columns), None)
        if not class_col:
            print("  ✗ No classification column found.")
            continue

        # --- compute CS2 density ---
        cs2_density = compute_cs2_density(cs2_gdf, class_col)
        if cs2_density is None:
            print("  ✗ No valid CS2 points.")
            continue

        # --- compute S1 track density ---
        s1_density = compute_s1_track_density(s1_file, cs2_gdf, buffer_meters=TRACK_BUFFER_METERS)
        if s1_density is None:
            print("  ✗ No valid S1 pixels in track region.")
            continue

        # --- combine results ---
        combined = {
            "scene_name": row["sceneName"],
            **cs2_density,
            **s1_density,

            # differences
            "diff_lead_density": cs2_density["density_CS2_lead_only"] - s1_density["density_S1_lead_only"],
            "diff_leadref_density": cs2_density["density_CS2_leadref"] - s1_density["density_S1_leadref"],
            "diff_floe_density": cs2_density["density_CS2_floe"] - s1_density["density_S1_floe"]
        }

        results.append(combined)
        print(f"  ✓ Density computed successfully.")

    # --- Save summary ---
    if results:
        outcsv = os.path.join(
            OUTPUT_DIR,
            "overall_density_summary_2023.csv"
        )
        pd.DataFrame(results).to_csv(outcsv, index=False)
        print("\nSaved:", outcsv)
    else:
        print("\n✗ No valid density results.")

    print("\nDone.")


if __name__ == "__main__":
    main()
