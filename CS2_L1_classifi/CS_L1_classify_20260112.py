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
    """按照 Swiggs et al. (2024) 计算 PP: Pmax / Pmean [cite: 123, 125]"""
    if not isinstance(waveform, np.ndarray):
        waveform = np.array(waveform)
    if waveform.size == 0 or np.all(waveform == 0):
        return np.nan

    b_max = np.argmax(waveform)
    start_index, end_index = b_max - 50, b_max + 77
    cropped_waveform = np.zeros(128)
    src_start, src_end = max(0, start_index), min(waveform.size, end_index + 1)
    dest_start = max(0, -start_index)
    dest_end = dest_start + (src_end - src_start)
    cropped_waveform[dest_start:dest_end] = waveform[src_start:src_end]

    p_max = np.max(cropped_waveform)
    p_mean = np.mean(cropped_waveform)
    return p_max / p_mean if p_mean != 0 else np.nan

# =============================================================================
# BATCH PROCESSING FUNCTION
# =============================================================================
def process_all_years_cs2(root_input, root_output):
    """遍历所有年份文件夹并执行分类"""
    
    # 查找输入根目录下所有的年份子文件夹 
    year_folders = [f for f in os.listdir(root_input) if os.path.isdir(os.path.join(root_input, f))]
    print(f"🚀 找到待处理年份: {year_folders}")

    search_patterns = ["CS_OFFL_SIR_SIN_1B_*.nc", "CS_LTA__SIR_SIN_1B_*.nc", "CS_NRT__SIR_SIN_1B_*.nc"]

    for year in sorted(year_folders):
        input_year_path = os.path.join(root_input, year)
        output_year_path = os.path.join(root_output, year)
        
        # 获取该年份下所有 NC 文件
        nc_files = []
        for pattern in search_patterns:
            nc_files.extend(glob.glob(os.path.join(input_year_path, pattern)))

        if not nc_files:
            print(f"⚠️ 跳过年份 {year}: 未找到符合条件的 .nc 文件")
            continue

        os.makedirs(output_year_path, exist_ok=True)
        print(f"\n📅 正在处理年份 [{year}]，共 {len(nc_files)} 个文件...")

        for file_path in nc_files:
            file_name = os.path.basename(file_path)
            try:
                with xr.open_dataset(file_path) as ds:
                    # 提取数据
                    lat, lon = ds["lat_20_ku"].values, ds["lon_20_ku"].values
                    time_raw = ds["time_20_ku"].values
                    waveforms = ds["pwr_waveform_20_ku"].values
                    stack_std = ds["stack_std_20_ku"].values

                    # 时间解析兼容性修复
                    if np.issubdtype(time_raw.dtype, np.datetime64):
                        utc_time = pd.to_datetime(time_raw)
                    else:
                        utc_time = pd.to_datetime(time_raw, origin=datetime(2000, 1, 1), unit="s")

                    # 分类逻辑 
                    pp_vals = np.array([calculate_pulse_peakiness(wf) for wf in waveforms])
                    labels = np.full(pp_vals.shape, "ambiguous", dtype='<U10') # 默认保留所有 ambiguous [cite: 174]

                    mask_lead = (pp_vals > 18) & (stack_std < 4.62) 
                    mask_ice  = (pp_vals < 9) & (stack_std > 4.62) 

                    labels[mask_lead] = "lead"
                    labels[mask_ice] = "ice"

                    # 导出结果
                    df = pd.DataFrame({
                        "lat": lat, "lon": lon, 
                        "time": utc_time.astype(str), 
                        "pp": pp_vals, "ssd": stack_std, 
                        "class": labels
                    })

                    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
                    
                    base_name = os.path.splitext(file_name)[0]
                    output_file = os.path.join(output_year_path, f"{base_name}_class.gpkg")
                    gdf.to_file(output_file, driver="GPKG")

            except Exception as e:
                print(f"❌ 文件错误 {file_name}: {e}")

    print("\n✅ 所有年份处理完成！")

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    ROOT_L1_DIR = r"F:\NWP\CS2_L1"
    ROOT_OUT_DIR = r"F:\NWP\CS2_S1_matched\CS2 File\CS2 class point_20260112"

    process_all_years_cs2(ROOT_L1_DIR, ROOT_OUT_DIR)