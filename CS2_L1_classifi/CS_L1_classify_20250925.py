import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime

# =============================================================================
# HELPER FUNCTION (Unchanged)
# =============================================================================
def calculate_pulse_peakiness(waveform: np.ndarray) -> float:
    # ... (function code is the same)
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
    pp = p_max / p_mean
    return pp

# =============================================================================
# MAIN SCRIPT
# =============================================================================
def process_cs2_folder_merged_output(input_folder, output_folder):
    """
    Processes each NetCDF file and exports a SINGLE, merged geospatial file
    containing all classified points.
    """
    search_path = os.path.join(input_folder, 'CS_OFFL_SIR_SIN_1B_*.nc')
    nc_files = glob.glob(search_path)
    
    if not nc_files:
        print(f"No NetCDF files found in: {input_folder}")
        return

    print(f"Found {len(nc_files)} files to process...")
    os.makedirs(output_folder, exist_ok=True)
    
    for file_path in nc_files:
        print(f"\nProcessing: {os.path.basename(file_path)}")
        try:
            with xr.open_dataset(file_path) as ds:
                lat, lon, time_raw, waveforms, stack_std, stack_kurtosis = [
                    ds[var].values for var in 
                    ['lat_20_ku', 'lon_20_ku', 'time_20_ku', 'pwr_waveform_20_ku', 
                     'stack_std_20_ku', 'stack_kurtosis_20_ku']
                ]
                valid_mask = (lat < 90) & (lon > -181);
                if not np.any(valid_mask):
                    print("  -> No valid data points. Skipping."); continue
                lat, lon, time_raw, waveforms, stack_std, stack_kurtosis = [arr[valid_mask] for arr in [lat, lon, time_raw, waveforms, stack_std, stack_kurtosis]]
                
                # --- START: MODIFIED SECTION ---
                # This block now robustly handles time conversion.
                
                # Check if xarray already converted the time to datetime objects
                if np.issubdtype(time_raw.dtype, np.datetime64):
                    # If yes, the data is already in the correct format
                    utc_time = time_raw
                else:
                    # If it's numeric, convert it from seconds since the TAI epoch
                    tai_epoch = datetime(2000, 1, 1, 0, 0, 0)
                    utc_time = pd.to_datetime(time_raw, origin=tai_epoch, unit='s')
                
                # --- END: MODIFIED SECTION ---

                pp_calculated = np.array([calculate_pulse_peakiness(wf) for wf in waveforms])
                df_classify = pd.DataFrame({'pp_calculated': pp_calculated, 'std': stack_std});
                df_classify['type'] = np.nan
                mask_lead = (df_classify['pp_calculated'] >= 18) & (df_classify['std'] <= 4.62)
                mask_ice = (df_classify['pp_calculated'] < 9) & (df_classify['std'] > 4)
                combined_mask = mask_lead | mask_ice
                df_classify.loc[combined_mask, 'type'] = np.where(df_classify.loc[combined_mask, 'pp_calculated'] >= 40, 'lead', 'ice')
                
                # Create the final DataFrame using the correctly formatted utc_time
                df_processed = pd.DataFrame({
                    'latitude': lat,
                    'longitude': lon,
                    'utc_time': utc_time, # <-- Use the processed time variable
                    'pp_calc': pp_calculated,
                    'std': stack_std,
                    'kurtosis': stack_kurtosis,
                    'class': df_classify['type']
                })
                df_processed.dropna(subset=['class'], inplace=True)

                if df_processed.empty:
                    print("  -> No points were classified in this file. Skipping export.")
                    continue

                gdf = gpd.GeoDataFrame(
                    df_processed,
                    geometry=gpd.points_from_xy(df_processed.longitude, df_processed.latitude),
                    crs="EPSG:4326"
                )
                
                print(f"  -> Classified points: {len(gdf)} (Leads: {len(gdf[gdf['class'] == 'lead'])}, Ice: {len(gdf[gdf['class'] == 'ice'])})")

                base_filename = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(output_folder, f"{base_filename}_classified.gpkg")
                
                gdf.to_file(output_path, driver='GPKG')
                print(f"     -> Saved all points to: {os.path.basename(output_path)}")
                
        except Exception as e:
            print(f"  -> ERROR processing file {os.path.basename(file_path)}: {e}")

    print("\nAll files processed.")

if __name__ == '__main__':
    INPUT_DATA_FOLDER = r"F:\NWP\CS2_L1\2015"
    OUTPUT_FOLDER = r"F:\NWP\CS2_S1_matched\CS2 File\CS2 class point\2015"
    process_cs2_folder_merged_output(INPUT_DATA_FOLDER, OUTPUT_FOLDER)