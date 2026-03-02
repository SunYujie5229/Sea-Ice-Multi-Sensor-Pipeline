import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from datetime import datetime
from shapely.geometry import Point

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

def sample_raster_by_date(gdf, search_path_pattern, date_str, col_name):
    """通用采样函数"""
    files = glob.glob(search_path_pattern)
    if not files:
        print(f"⚠️ 未找到日期 {date_str} 的栅格文件")
        gdf[col_name] = np.nan
        return gdf
    raster_path = files[0]
    print(f"   - 正在采样 {col_name}: {os.path.basename(raster_path)}")
    with rasterio.open(raster_path) as src:
        coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
        gdf[col_name] = [val[0] for val in src.sample(coords)]
    return gdf

def process_cs2_visual(file_path, output_folder, sic_root, s1_root):
    file_name = os.path.basename(file_path)
    print(f"\n🚀 开始执行可视化导出: {file_name}")

    try:
        with xr.open_dataset(file_path) as ds:
            # 1. 提取基础数据
            lat = ds["lat_20_ku"].values
            lon = ds["lon_20_ku"].values
            time_raw = ds["time_20_ku"].values
            waveforms = ds["pwr_waveform_20_ku"].values
            stack_std = ds["stack_std_20_ku"].values

            # 2. 时间解析
            if np.issubdtype(time_raw.dtype, np.datetime64):
                dt_objs = pd.to_datetime(time_raw)
            else:
                dt_objs = pd.to_datetime(time_raw, origin=datetime(2000, 1, 1), unit="s")
            date_str = dt_objs[0].strftime("%Y%m%d")
            year_str = dt_objs[0].strftime("%Y")
            
            # 3. 计算 PP
            pp_vals = np.array([calculate_pulse_peakiness(wf) for wf in waveforms])

            # 4. 构建属性字段（严格匹配图 5 设定）
            df = pd.DataFrame({
                "lat": lat, "lon": lon, 
                "time": dt_objs.astype(str),
                "pp": pp_vals, 
                "ssd": stack_std
            })
            
            # 增加 PP 分类字段 (用于中心轨道)
            # <9: Floe, 9-18: Ambiguous, >18: Lead
            df['pp_cat'] = 'Ambiguous'
            df.loc[df['pp'] < 9, 'pp_cat'] = 'Floe'
            df.loc[df['pp'] > 18, 'pp_cat'] = 'Lead'
            
            # 增加 SSD 分类字段 (用于偏移轨道)
            # <4.62: Lead, >4.62: Floe
            df['ssd_cat'] = np.where(df['ssd'] < 4.62, 'Lead', 'Floe')

            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

            # 5. 采样 SIC 并分类
            sic_pattern = os.path.join(sic_root, year_str, "n6250", f"*{date_str}*.tif")
            gdf = sample_raster_by_date(gdf, sic_pattern, date_str, "sic_raw")
            # 0-75: Open Ocean, 75-100: Floe
            gdf["sic_cat"] = np.where(gdf["sic_raw"] <= 75, 'Open Ocean', 'Floe')

            # 6. 采样 Sentinel-1
            s1_pattern = os.path.join(s1_root, year_str, f"*{date_str}*.tif")
            gdf = sample_raster_by_date(gdf, s1_pattern, date_str, "s1_class")

            # 7. 导出偏移 SHP
            # 稍微加大了偏移量 (0.05间隔) 以便在小比例尺下也能看清
            offsets = {
                "SIC_L05": -0.05,  # 采样自 SIC
                "PP_000": 0.0,     # 中心轨道，显示 PP 分类
                "SSD_R05": 0.05,   # 显示 SSD 分类
                "S1_R10": 0.10     # 显示 S1 采样结果
            }

            if not os.path.exists(output_folder): os.makedirs(output_folder)
            base_name = os.path.splitext(file_name)[0]

            for suffix, dx in offsets.items():
                temp_gdf = gdf.copy()
                temp_gdf.geometry = temp_gdf.geometry.translate(xoff=dx)
                out_path = os.path.join(output_folder, f"{base_name}_{suffix}.shp")
                temp_gdf.to_file(out_path)
                print(f"   ✅ 导出层: {suffix}")

    except Exception as e:
        print(f"❌ 处理失败: {e}")


# =============================================================================
if __name__ == "__main__":
    SINGLE_FILE = r"F:\NWP\CS2_L1\2016\CS_LTA__SIR_SIN_1B_20161222T215303_20161222T215423_E001.nc"
    SAVE_PATH = r"E:\Manuscript\CS2_S1\offline_track"
    
    SIC_DIR = r"D:\S1_CS2_data\SIC"
    S1_DIR = r"C:\Users\TJ002\Desktop\classification result"
    
    process_cs2_visual(SINGLE_FILE, SAVE_PATH, SIC_DIR, S1_DIR)