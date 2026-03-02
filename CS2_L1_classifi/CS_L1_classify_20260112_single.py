import os
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime

def calculate_pulse_peakiness(waveform: np.ndarray) -> float:
    """计算脉冲峰值 (PP): Pmax / Pmean"""
    if not isinstance(waveform, np.ndarray):
        waveform = np.array(waveform)
    if waveform.size == 0 or np.all(waveform == 0):
        return np.nan

    # 裁剪波形至 128 bins (模拟标准处理)
    b_max = np.argmax(waveform)
    start_index = b_max - 50
    end_index = b_max + 77
    cropped_waveform = np.zeros(128)
    src_start, src_end = max(0, start_index), min(waveform.size, end_index + 1)
    dest_start = max(0, -start_index)
    dest_end = dest_start + (src_end - src_start)
    cropped_waveform[dest_start:dest_end] = waveform[src_start:src_end]

    p_max = np.max(cropped_waveform)
    p_mean = np.mean(cropped_waveform)
    return p_max / p_mean if p_mean != 0 else np.nan

def process_single_cs2_file(file_path, output_folder):
    """处理单个 CS2 文件，保留 Ambiguous 类别并导出为 SHP"""
    file_name = os.path.basename(file_path)
    print(f"正在处理: {file_name}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        with xr.open_dataset(file_path) as ds:
            # 1. 提取基础数据
            lat = ds["lat_20_ku"].values
            lon = ds["lon_20_ku"].values
            time_raw = ds["time_20_ku"].values
            waveforms = ds["pwr_waveform_20_ku"].values
            stack_std = ds["stack_std_20_ku"].values

            # 2. 修复时间转换逻辑
            # 如果时间已经是 datetime64 类型，直接转换；否则使用 numeric 转换
            if np.issubdtype(time_raw.dtype, np.datetime64):
                utc_str = pd.to_datetime(time_raw).astype(str)
            else:
                try:
                    # 尝试按秒转换（CS2 常用 2000-01-01 起始的秒数）
                    utc_str = pd.to_datetime(time_raw, origin=datetime(2000, 1, 1), unit="s").astype(str)
                except:
                    # 最终兜底方案
                    utc_str = pd.to_datetime(time_raw).astype(str)

            # 3. 计算 PP
            pp_vals = np.array([calculate_pulse_peakiness(wf) for wf in waveforms])

            # 4. 分类逻辑 (使用您指定的阈值)
            # 初始全部标记为 'ambiguous'
            labels = np.full(pp_vals.shape, "ambiguous", dtype='<U10')

            # Lead 掩码: pp >= 18 且 ssd <= 4.62
            mask_lead = (pp_vals >= 18) & (stack_std <= 4.62)
            
            # Ice 掩码: pp < 9 且 ssd > 4
            mask_ice = (pp_vals < 9) & (stack_std > 4)

            # 执行赋值
            labels[mask_lead] = "lead"
            labels[mask_ice] = "ice"

            # 5. 构建 GeoDataFrame
            df = pd.DataFrame({
                "lat": lat,
                "lon": lon,
                "time": utc_str,
                "pp": pp_vals,
                "ssd": stack_std,
                "class": labels
            })

            # 转为地理空间对象 (WGS84)
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

            # 6. 导出为 SHP
            # 注意：SHP 字段名长度限制为10个字符
            base_name = os.path.splitext(file_name)[0]
            output_path = os.path.join(output_folder, f"{base_name}_class.shp")
            gdf.to_file(output_path, driver="ESRI Shapefile")

            print(f"--- 处理成功 ---")
            print(f"结果路径: {output_path}")
            print(f"分类统计:")
            print(f"  - Lead:      {np.sum(labels == 'lead')}")
            print(f"  - Ice:       {np.sum(labels == 'ice')}")
            print(f"  - Ambiguous: {np.sum(labels == 'ambiguous')}")

    except Exception as e:
        print(f"❌ 处理失败: {e}")

# =============================================================================
# 使用示例
# =============================================================================
if __name__ == "__main__":
    # 请修改为您具体的文件路径
    SINGLE_FILE = r"F:\NWP\CS2_L1\2018\CS_LTA__SIR_SIN_1B_20180129T124408_20180129T124622_E001.nc"
    SAVE_PATH = r"E:\Manuscript\CS2_S1"
    
    process_single_cs2_file(SINGLE_FILE, SAVE_PATH)