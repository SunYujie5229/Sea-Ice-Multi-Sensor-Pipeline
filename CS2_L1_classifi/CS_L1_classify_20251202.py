import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime

# =============================================================================
# HELPER FUNCTION
# =============================================================================
def calculate_pulse_peakiness(waveform: np.ndarray) -> float:
    """
    Compute pulse peakiness (PP) from a waveform array of power values.
    """
    if not isinstance(waveform, np.ndarray):
        waveform = np.array(waveform)

    if waveform.size == 0 or np.all(waveform == 0):
        return np.nan

    b_max = np.argmax(waveform)
    start_index = b_max - 50
    end_index = b_max + 77

    cropped_waveform = np.zeros(128)
    src_start = max(0, start_index)
    src_end = min(waveform.size, end_index + 1)

    dest_start = max(0, -start_index)
    dest_end = dest_start + (src_end - src_start)

    cropped_waveform[dest_start:dest_end] = waveform[src_start:src_end]

    p_max = np.max(cropped_waveform)
    p_mean = np.mean(cropped_waveform)

    if p_mean == 0:
        return np.nan

    return p_max / p_mean


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================
def process_cs2_folder_merged_output(input_folder, output_folder):
    """
    Process CryoSat-2 OFFL + LTA L1B products and save unified classified GPKG.
    """

    print("\nSearching for CryoSat-2 L1B files...")

    # Support OFFL + LTA (双下划线!!) + NRT
    search_patterns = [
        "CS_OFFL_SIR_SIN_1B_*.nc",
        "CS_LTA__SIR_SIN_1B_*.nc",     # LTA has two underscores
        "CS_NRT__SIR_SIN_1B_*.nc"      # Optional
    ]

    nc_files = []
    for pattern in search_patterns:
        files = glob.glob(os.path.join(input_folder, pattern))
        nc_files.extend(files)

    if not nc_files:
        print(f"❌ No compatible CS2 files found in: {input_folder}")
        return

    print(f"✅ Found {len(nc_files)} files to process.")
    os.makedirs(output_folder, exist_ok=True)

    # ========================================================
    # LOOP OVER FILES
    # ========================================================
    for file_path in nc_files:
        file_name = os.path.basename(file_path)
        print(f"\n----------------------------------------------------------")
        print(f"Processing: {file_name}")
        print(f"----------------------------------------------------------")

        try:
            with xr.open_dataset(file_path) as ds:

                # Extract necessary variables
                lat, lon, time_raw, waveforms, stack_std, stack_kurt = [
                    ds[var].values for var in
                    [
                        "lat_20_ku", "lon_20_ku", "time_20_ku",
                        "pwr_waveform_20_ku", "stack_std_20_ku", "stack_kurtosis_20_ku"
                    ]
                ]

                # Basic filtering
                valid_mask = (lat < 90) & (lon > -181)
                if not np.any(valid_mask):
                    print("  -> No valid geolocation points, skipping.")
                    continue

                lat = lat[valid_mask]
                lon = lon[valid_mask]
                time_raw = time_raw[valid_mask]
                waveforms = waveforms[valid_mask]
                stack_std = stack_std[valid_mask]
                stack_kurt = stack_kurt[valid_mask]

                # ========================================================
                # TIME HANDLING (OFFL & LTA unified)
                # ========================================================
                if np.issubdtype(time_raw.dtype, np.datetime64):
                    utc_time = time_raw
                else:
                    utc_time = pd.to_datetime(
                        time_raw,
                        origin=datetime(2000, 1, 1),
                        unit="s"
                    )

                # ========================================================
                # CLASSIFICATION
                # ========================================================
                pp_vals = np.array([calculate_pulse_peakiness(wf) for wf in waveforms])

                df_classify = pd.DataFrame({
                    "pp": pp_vals,
                    "std": stack_std
                })

                df_classify["type"] = np.nan

                # Lead & Ice rules
                mask_lead = (df_classify["pp"] >= 18) & (df_classify["std"] <= 4.62)
                mask_ice  = (df_classify["pp"] < 9) & (df_classify["std"] > 4)

                mask_combined = mask_lead | mask_ice
                df_classify.loc[mask_combined, "type"] = np.where(
                    df_classify.loc[mask_combined, "pp"] >= 40,
                    "lead",
                    "ice"
                )

                # Build final DataFrame
                df = pd.DataFrame({
                    "latitude": lat,
                    "longitude": lon,
                    "utc_time": utc_time,
                    "pp": pp_vals,
                    "std": stack_std,
                    "kurtosis": stack_kurt,
                    "class": df_classify["type"]
                })

                df.dropna(subset=["class"], inplace=True)

                if df.empty:
                    print("  -> All points unclassified. Skipping export.")
                    continue

                # Convert to GeoDataFrame
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df.longitude, df.latitude),
                    crs="EPSG:4326"
                )

                print(f"  -> Classified points: {len(gdf)}")
                print(f"     Leads: {len(gdf[gdf['class'] == 'lead'])}")
                print(f"     Ice:   {len(gdf[gdf['class'] == 'ice'])}")

                # ========================================================
                # EXPORT
                # ========================================================
                base_name = os.path.splitext(file_name)[0]
                output_path = os.path.join(output_folder, f"{base_name}_classified.gpkg")
                gdf.to_file(output_path, driver="GPKG")

                print(f"  -> Saved GPKG: {output_path}")

        except Exception as e:
            print(f"❌ ERROR processing {file_name}: {e}")

    print("\n🎉 All files processed successfully!")
    print("==========================================")


# =============================================================================
# MAIN ENTRY
# =============================================================================
if __name__ == "__main__":
    INPUT_DATA_FOLDER = r"F:\NWP\CS2_L1\2016\CS_LTA__SIR_SIN_1B_20160930T124302_20160930T124517_E001.nc"
    OUTPUT_FOLDER = r"F:\NWP\CS2_S1_matched\CS2 File\CS2 class point\2022"

    process_cs2_folder_merged_output(INPUT_DATA_FOLDER, OUTPUT_FOLDER)
