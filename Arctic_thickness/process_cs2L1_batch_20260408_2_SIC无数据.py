import os
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal
import scipy.ndimage
import multiprocessing as mp
import rasterio
from pyproj import Transformer
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator, interp1d, griddata
######################################
#增加SIC采样和文件遍历
#######################################


# 忽略数值计算产生的警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=scipy.optimize.OptimizeWarning)

# =====================================================================
# 模块 1: 参数配置与工具函数
# =====================================================================
_INTERP_CACHE = {} 
_SNOW_CACHE = {}
_SIC_INDEX = {}  # 缓存 SIC 文件索引

def get_processing_params(file_path):
    filename = os.path.basename(file_path)
    mode = "SARIn" if "SIR_SIN" in filename else "SAR"
    return {
        "mode": mode,
        "c": 299792458.0,
        "bin_size": 0.2342,
        "bn": 512 if mode == "SARIn" else 128,#bn 是波形中心点在窗口中的位置，SARIn 模式下为 512，SAR 模式下为 128
        "crop_left": 50,
        "crop_right": 77,
        "lead_retracker_mode": "threshold",
        "ice_retrack_threshold": 0.70,
        "lead_retrack_threshold": 0.50,
        "pp_lead_threshold": 18,
        "pp_ice_threshold": 9,
        "ssd_threshold": 4.62 if mode == "SARIn" else 6.29,
        "needs_smoothing": True if mode == "SARIn" else False
    }

def extract_date_from_sic_name(filename):
    """
    根据实际SIC文件名格式调整解析逻辑
    假设格式包含类似 20240101 的 8 位日期
    """
    import re
    match = re.search(r"(\20\d{6})", filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    return None

# =====================================================================
# 模块 2: 重跟踪与分类算子 (保持原样)
# =====================================================================
def classify_waveforms_vectorized(wf_array, ssd_array, P):
    p_max = np.max(wf_array, axis=1)
    noise_floor = np.mean(wf_array[:, 10:20], axis=1)
    masked = np.where(wf_array > noise_floor[:, np.newaxis], wf_array, 0)
    count_valid = (masked > 0).sum(axis=1)
    p_mean = np.divide(masked.sum(axis=1), count_valid, out=np.zeros_like(p_max), where=count_valid > 0)
    pp = np.divide(p_max, p_mean, out=np.zeros_like(p_max), where=p_mean > 0)
    types = np.full(wf_array.shape[0], 'unknown', dtype=object)
    types[(pp > P['pp_lead_threshold']) & (ssd_array < P['ssd_threshold'])] = 'lead'
    types[(pp < P['pp_ice_threshold']) & (ssd_array > P['ssd_threshold'])] = 'ice'
    return types, pp

def retrack_wf(wf, offset, p_type, P):
    # 此处逻辑保留原代码...
    if p_type == 'ice':
        if P['needs_smoothing']: wf = scipy.ndimage.uniform_filter1d(wf, size=3)
        max_v = np.max(wf); peaks, _ = scipy.signal.find_peaks(wf)
        valid = [p for p in peaks if wf[p] >= 0.20 * max_v]
        if not valid: return np.nan
        f_p = valid[0]; target = wf[f_p] * P['ice_retrack_threshold']
        for j in range(f_p, 0, -1):
            if wf[j-1] <= target <= wf[j]:
                t0 = (j-1) + (target - wf[j-1])/(wf[j] - wf[j-1] + 1e-9)
                return (offset + t0 - P['bn']) * P['bin_size'] - 0.1626
    elif p_type == 'lead':
        m_idx = np.argmax(wf); target = wf[m_idx] * P['lead_retrack_threshold']
        for j in range(m_idx, 0, -1):
            if wf[j-1] <= target <= wf[j]:
                t0 = (j-1) + (target - wf[j-1])/(wf[j] - wf[j-1] + 1e-9)
                return (offset + t0 - P['bn']) * P['bin_size']
    return np.nan

# =====================================================================
# 模块 3: SIC 空间采样核心
# =====================================================================
def get_sic_for_points(lons, lats, target_datetime, sic_base_dir):
    """
    根据日期动态构建路径并检索当天的 SIC 文件
    路径格式: sic_base_dir / yyyy / n6250 / *yyyymmdd*.tif
    """
    if not sic_base_dir or not os.path.exists(sic_base_dir):
        return np.full(len(lons), np.nan)
    
    dt = target_datetime
    # 1. 构建目标文件夹路径: D:\S1_CS2_data\SIC\yyyy\n6250
    target_folder = os.path.join(sic_base_dir, str(dt.year), "n6250")
    
    if not os.path.exists(target_folder):
        return np.full(len(lons), np.nan)
    
    # 2. 搜索包含当天日期（如 20240101）的 tif 文件
    date_str = dt.strftime("%Y%m%d")
    search_pattern = os.path.join(target_folder, f"*{date_str}*.tif*")
    files = glob.glob(search_pattern)
    
    if not files:
        # 如果 yyyymmdd 没找到，尝试匹配 yyyy_mm_dd 等通用格式
        date_str_alt = dt.strftime("%Y_%m_%d")
        files = glob.glob(os.path.join(target_folder, f"*{date_str_alt}*.tif*"))
        if not files: return np.full(len(lons), np.nan)

    sic_path = files[0] # 取匹配到的第一个文件
    
    # 3. 采样数据
    try:
        with rasterio.open(sic_path) as src:
            # 建立坐标转换器 (WGS84 -> Tiff CRS)
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xx, yy = transformer.transform(lons, lats)
            
            # 使用采样器提取点位数值
            coords = zip(xx, yy)
            # sample 返回一个 generator，每个点对应一个包含波段值的 array
            samples = [val[0] for val in src.sample(coords)]
            
            sic_values = np.array(samples).astype(float)
            # 处理可能的填充值 (如 255 或 -9999)
            # sic_values[sic_values > 100] = np.nan 
            return sic_values
    except Exception as e:
        print(f"SIC Sampling Error: {e}")
        return np.full(len(lons), np.nan)

# =====================================================================
# 模块 4: MSS/Snow/SLA 链条 (保持原样)
# =====================================================================
def load_mss_fast(lons, lats, mss_path):
    if not mss_path or not os.path.exists(mss_path): return np.full(len(lons), np.nan)
    if mss_path not in _INTERP_CACHE:
        ds = xr.open_dataset(mss_path)
        if ds.lon.max() > 180: ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
        _INTERP_CACHE[mss_path] = RegularGridInterpolator((ds.lat.values, ds.lon.values), 
                                                        ds['mean_sea_surf_sol2'].values, 
                                                        bounds_error=False, fill_value=np.nan)
    return _INTERP_CACHE[mss_path](np.column_stack((lats, lons)))

def load_snow_and_fix(df, snow_dir):
    # 原有的积雪加载逻辑...
    if not snow_dir or df.empty: return df
    month = int(df['utc_time'].iloc[0].month)
    if f"snow_{month}" not in _SNOW_CACHE:
        pattern = os.path.join(snow_dir, f"*clim-{month:02d}-*.nc")
        files = glob.glob(pattern)
        if not files: return df
        ds = xr.open_dataset(files[0])
        lon, lat = ds.lon.values, ds.lat.values
        lon_grid, lat_grid = np.meshgrid(lon, lat); lon_180 = ((lon_grid + 180) % 360) - 180
        valid = ~np.isnan(ds['merged_snow_depth'].values)
        _SNOW_CACHE[f"snow_{month}"] = {
            'pts': (lon_180[valid], lat_grid[valid]),
            'sd': ds['merged_snow_depth'].values[valid],
            'w': ds['w99_weight'].values[valid] if 'w99_weight' in ds.data_vars else np.ones_like(ds['merged_snow_depth'].values[valid])
        }
    c = _SNOW_CACHE[f"snow_{month}"]
    q_lon, q_lat = ((df['lon'].values + 180) % 360) - 180, df['lat'].values
    df['merged_sd_raw'] = griddata(c['pts'], c['sd'], (q_lon, q_lat), method='linear')
    df['w99_weight'] = griddata(c['pts'], c['w'], (q_lon, q_lat), method='linear')
    df['f_myi'] = 0.0 
    df['hs'] = df['merged_sd_raw'] - (1 - df['f_myi']) * 0.5 * df['w99_weight'] * df['merged_sd_raw']
    return df

def compute_sla_chain(df):
    lead_mask = (df['type'] == 'lead') & (df['h_surface'].notna())
    df['SLA_raw'] = np.nan
    df.loc[lead_mask, 'SLA_raw'] = df.loc[lead_mask, 'h_surface'] - df.loc[lead_mask, 'mss_interp']
    valid_leads = lead_mask & df['SLA_raw'].between(-3.0, 3.0)
    if valid_leads.sum() < 2: 
        df['SLA'], df['dist_to_lead'] = np.nan, np.inf
        return df
    window = int(100000 / 380)
    sla_smooth = df.loc[valid_leads, 'SLA_raw'].rolling(window=window, center=True, min_periods=1).mean()
    df['SLA'] = interp1d(sla_smooth.index, sla_smooth.values, kind='linear', bounds_error=False)(df.index)
    l_coords = np.radians(df.loc[valid_leads, ['lat', 'lon']].values)
    tree = cKDTree(l_coords)
    dist, _ = tree.query(np.radians(df[['lat', 'lon']].values), k=1)
    df['dist_to_lead'] = dist * 6371000.0
    df.loc[df['dist_to_lead'] > 200000, 'SLA'] = np.nan
    return df

# =====================================================================
# 模块 5: 单文件处理主函数 (更新 SIC 集成)
# =====================================================================
def process_one_file(file_path, aux_paths, output_dir_base, input_dir_base):
    track_id = os.path.basename(file_path)
    # 通过显式传入的 input_dir_base 计算相对路径
    rel_path = os.path.relpath(os.path.dirname(file_path), start=input_dir_base)
    current_output_dir = os.path.join(output_dir_base, rel_path)
    os.makedirs(current_output_dir, exist_ok=True)

    try:
        P = get_processing_params(file_path)
        ds = xr.open_dataset(file_path, decode_timedelta=True)
        
        # 1. 基础物理量提取
        lat, lon = ds['lat_20_ku'].values, ds['lon_20_ku'].values
        wf = ds['pwr_waveform_20_ku'].values
        noise = np.where(ds['noise_power_20_ku'].values == -9999.99, 0, ds['noise_power_20_ku'].values)
        wf_clean = wf - noise[:, np.newaxis]
        
        # 2. 构建 DataFrame
        df = pd.DataFrame({'track_id': track_id, 'lat': lat, 'lon': lon,
                           'alt': ds['alt_20_ku'].values, 'window_del': ds['window_del_20_ku'].dt.total_seconds().values,
                           'std': ds['stack_std_20_ku'].values})
        df['utc_time'] = datetime(2000, 1, 1) + pd.to_timedelta((ds['time_20_ku'].values - np.datetime64('2000-01-01T00:00:00')) / np.timedelta64(1, 's'), unit='s')

        # 3. SIC 采样
        df['sic'] = get_sic_for_points(df['lon'].values, df['lat'].values, df['utc_time'].iloc[0], aux_paths.get('sic'))

        # 4. 分类与重跟踪
        df['type'], df['pp'] = classify_waveforms_vectorized(wf_clean, df['std'].values, P)
        b_max = np.argmax(wf_clean, axis=1)
        df['abs_offset'] = b_max - P['crop_left']
        df['range_correction'] = [retrack_wf(wf_clean[i, max(0, int(b_max[i]-P['crop_left'])):min(wf_clean.shape[1], int(b_max[i]+P['crop_right']+1))], 
                                           df.loc[i, 'abs_offset'], df.loc[i, 'type'], P) for i in df.index]

        # 5. 海面高度与海冰反演
        df['h_surface'] = df['alt'] - (df['window_del'] * P['c'] / 2 + df['range_correction']) 
        df['mss_interp'] = load_mss_fast(df['lon'], df['lat'], aux_paths.get('mss'))
        df = compute_sla_chain(df)
        
        df['radar_freeboard'] = df['h_surface'] - (df['mss_interp'] + df['SLA'])
        df = load_snow_and_fix(df, aux_paths.get('snow'))
        
        # 最终计算海冰厚度
        df['corrected_freeboard'] = df['radar_freeboard'] + 0.25 * df['hs']
        rho_w, rho_s, rho_i = 1023.9, 324.0, 916.7
        df['sea_ice_thickness'] = (df['corrected_freeboard'] * rho_w + df['hs'] * rho_s) / (rho_w - rho_i)

        # 6. 保存
        out_path = os.path.join(current_output_dir, track_id.replace('.nc', '.parquet'))
        df.to_parquet(out_path, index=False)
        return True, file_path
    except Exception as e:
        return False, f"{track_id}: {str(e)}"

# =====================================================================
# 模块 6: 批量调度 (支持递归遍历)
# =====================================================================
def batch_process(input_dir, output_dir, aux_paths, n_workers=4):
    # 使用 **/*.nc 匹配所有子文件夹下的 nc 文件
    search_pattern = os.path.join(input_dir, "**", "*.nc")
    nc_files = glob.glob(search_pattern, recursive=True)
    
    if not nc_files:
        print(f"未在 {input_dir} 下找到任何 .nc 文件。")
        return
        
    print(f"发现 {len(nc_files)} 个文件。正在并行处理...")
    tasks = [(f, aux_paths, output_dir, input_dir) for f in nc_files]
    
    with mp.Pool(n_workers) as pool:
        results = pool.starmap(process_one_file, tasks)
        for status, msg in results:
            if not status: print(f"FAILED: {msg}")


if __name__ == "__main__":
    # ==== 配置区 ====
    INPUT_DIR = r"E:\Arctic\SAR L1\2024\01" # 根目录，其下应有 2024\01 等结构
    OUTPUT_DIR = r"E:\Arctic\SAR L1\L1_processed" # 输出根目录，处理结果将保存在对应的年月子目录下
    
    AUX_PATHS = {
        'mss': r"E:\Project_2024\CryoSat-2 L1\DTU21MSS_1min_WGS84.nc",
        'snow': r"D:\S1_CS2_data\awi_snow_merged",
        'sic': r"D:\S1_CS2_data\SIC" # 存放每日 SIC Tiff 的文件夹
    }
    
    batch_process(INPUT_DIR, OUTPUT_DIR, AUX_PATHS, n_workers=8)