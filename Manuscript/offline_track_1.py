import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from datetime import datetime

def calculate_pulse_peakiness(waveform: np.ndarray) -> float:
    """计算脉冲峰值 (PP)"""
    if not isinstance(waveform, np.ndarray):
        waveform = np.array(waveform)
    if waveform.size == 0 or np.all(waveform == 0):
        return np.nan
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

def sample_s1_multiple(gdf, search_pattern, col_prefix):
    """
    寻找匹配日期的所有 S1 文件并分别采样到不同字段。
    """
    files = sorted(glob.glob(search_pattern))
    if not files:
        print(f"⚠️ 未找到匹配的 S1 栅格文件")
        gdf[f"{col_prefix}1"] = np.nan
        return gdf
    
    for i, raster_path in enumerate(files, 1):
        # SHP 字段名限制10字符，s1_c_1, s1_c_2...
        col_name = f"{col_prefix}{i}"
        print(f"   - 正在采样 S1 ({i}/{len(files)}): {os.path.basename(raster_path)}")
        try:
            with rasterio.open(raster_path) as src:
                coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
                gdf[col_name] = [val[0] for val in src.sample(coords)]
        except Exception as e:
            print(f"   ❌ 采样失败 {os.path.basename(raster_path)}: {e}")
            gdf[col_name] = np.nan
    return gdf

def sample_single_raster(gdf, search_pattern, col_name):
    """通用单文件采样函数 (用于 SIC)"""
    files = glob.glob(search_pattern)
    if not files:
        gdf[col_name] = np.nan
        return gdf
    with rasterio.open(files[0]) as src:
        coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
        gdf[col_name] = [val[0] for val in src.sample(coords)]
    return gdf

def process_cs2_visual(file_path, output_folder, sic_root, s1_root):
    file_name = os.path.basename(file_path)
    print(f"\n🚀 开始处理: {file_name}")

    try:
        with xr.open_dataset(file_path) as ds:
            # 1. 提取基础数据
            lat, lon = ds["lat_20_ku"].values, ds["lon_20_ku"].values
            time_raw = ds["time_20_ku"].values
            waveforms = ds["pwr_waveform_20_ku"].values
            stack_std = ds["stack_std_20_ku"].values

            # 2. 时间解析
            if np.issubdtype(time_raw.dtype, np.datetime64):
                dt_objs = pd.to_datetime(time_raw)
            else:
                dt_objs = pd.to_datetime(time_raw, origin=datetime(2000, 1, 1), unit="s")
            date_str, year_str = dt_objs[0].strftime("%Y%m%d"), dt_objs[0].strftime("%Y")
            
            # 3. 计算 PP 并构建基础 DF
            pp_vals = np.array([calculate_pulse_peakiness(wf) for wf in waveforms])
            df = pd.DataFrame({
                "lat": lat, "lon": lon, "time": dt_objs.astype(str),
                "pp": pp_vals, "ssd": stack_std
            })

            # --- 增加可视化分类字段 ---
            # PP 分类: Lead(>18), Floe(<9), Ambiguous(9-18)
            df['pp_v'] = 'Ambiguous'
            df.loc[df['pp'] > 18, 'pp_v'] = 'Lead'
            df.loc[df['pp'] < 9, 'pp_v'] = 'Floe'
            
            # SSD 分类: Lead(<4.62), Floe(>4.62)
            df['ssd_v'] = np.where(df['ssd'] < 4.62, 'Lead', 'Floe')

            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

            # 4. SIC 采样与分类
            sic_pattern = os.path.join(sic_root, year_str, "n6250", f"*{date_str}*.tif")
            gdf = sample_single_raster(gdf, sic_pattern, "sic_raw")
            # 4. SIC 采样与三分类 (修复点)
            sic_pattern = os.path.join(sic_root, year_str, "n6250", f"*{date_str}*.tif")
            gdf = sample_single_raster(gdf, sic_pattern, "sic_raw")
            
            # 使用 np.select 处理多条件分类
            conditions = [
                (gdf["sic_raw"].isna()),
                (gdf["sic_raw"] <= 75),
                (gdf["sic_raw"] > 75) & (gdf["sic_raw"] <= 100),
                (gdf["sic_raw"] > 100)
            ]
            choices = ['None', '0-75', '75-100', '>100']
            gdf["sic_v"] = np.select(conditions, choices, default='None')

            # 5. Sentinel-1 多文件采样
            s1_pattern = os.path.join(s1_root, year_str, f"*{date_str}*.tif")
            gdf = sample_s1_multiple(gdf, s1_pattern, "s1_c_")

            # 6. 导出偏移 SHP (偏移量增大至 0.08)
            offsets = {
                "SIC_L08": -0.05,  # 显示 sic_v
                "PP_000": 0.0,     # 显示 pp_v (中心)
                "SSD_R08": 0.05,   # 显示 ssd_v
                "S1_R16": 0.10     # 显示 s1_c_1, s1_c_2...
            }

            if not os.path.exists(output_folder): os.makedirs(output_folder)
            base_name = os.path.splitext(file_name)[0]

            for suffix, dx in offsets.items():
                temp_gdf = gdf.copy()
                temp_gdf.geometry = temp_gdf.geometry.translate(xoff=dx)
                out_path = os.path.join(output_folder, f"{base_name}_{suffix}.shp")
                temp_gdf.to_file(out_path)
                print(f"   ✅ 导出成功: {suffix}")

    except Exception as e:
        print(f"❌ 处理失败: {e}")



# =============================================================================
if __name__ == "__main__":
    SINGLE_FILE = r"F:\NWP\CS2_L1\2022\CS_OFFL_SIR_SIN_1B_20220115T130025_20220115T130239_E001.nc"
    SAVE_PATH = r"E:\Manuscript\CS2_S1\offline_track"
    
    SIC_DIR = r"D:\S1_CS2_data\SIC"
    S1_DIR = r"C:\Users\TJ002\Desktop\classification result" 
    process_cs2_visual(SINGLE_FILE, SAVE_PATH, SIC_DIR, S1_DIR)