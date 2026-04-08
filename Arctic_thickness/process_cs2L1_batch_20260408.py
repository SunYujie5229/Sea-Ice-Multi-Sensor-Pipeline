import os
import glob
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal
import scipy.ndimage
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.interpolate import griddata, interp1d
from geopy.distance import geodesic
import multiprocessing as mp

# 忽略曲线拟合和全NaN切片警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=scipy.optimize.OptimizeWarning)

# =====================================================================
# 模块 1: 参数工厂 (SAR / SARIn 自动切换)
# =====================================================================
def get_processing_params(file_path):
    filename = os.path.basename(file_path)
    if "SIR_SIN" in filename:
        mode = "SARIn"
    elif "SIR_SAR" in filename:
        mode = "SAR"
    else:
        raise ValueError(f"无法从文件名识别模式: {filename}")

    params = {
        "mode": mode,
        "c": 299792458.0,
        "bin_size": 0.2342,
        "cropped_bins": 128,
        "crop_left": 50,
        "crop_right": 77,
        "retrack_threshold": 0.70,
        "first_peak_min_ratio": 0.20,
        "pp_lead_threshold": 18,
        "pp_ice_threshold": 9,
    }

    if mode == "SARIn":
        params.update({"raw_bins": 1024, "bn": 512, "ssd_threshold": 4.62, "peak_threshold": 0.45, "needs_smoothing": True})
    elif mode == "SAR":
        params.update({"raw_bins": 256, "bn": 128, "ssd_threshold": 6.29, "peak_threshold": 0.15, "needs_smoothing": False})
    
    return params

# =====================================================================
# 模块 2 & 3: 波形裁剪、PP 计算与 分类
# =====================================================================
def crop_waveform(waveform, crop_left=50, crop_right=77):
    b_max = np.argmax(waveform)
    start_index = b_max - crop_left
    end_index = b_max + crop_right + 1
    
    pad_left = max(0, -start_index)
    pad_right = max(0, end_index - len(waveform))
    start_safe = max(0, start_index)
    end_safe = min(len(waveform), end_index)
    
    cropped = waveform[start_safe:end_safe]
    if pad_left > 0 or pad_right > 0:
        cropped = np.pad(cropped, (pad_left, pad_right), 'constant', constant_values=0)
        
    abs_offset = b_max - crop_left
    return cropped, abs_offset

def classify_waveforms(waveform_array, ssd_array, params):
    n_samples = len(waveform_array)
    types = np.full(n_samples, 'unknown', dtype=object)
    absolute_offsets = np.full(n_samples, np.nan)
    custom_pps = np.full(n_samples, np.nan)
    cropped_waveforms = []

    for i in range(n_samples):
        wf = waveform_array[i]
        ssd = ssd_array[i]
        
        cropped_wf, abs_offset = crop_waveform(wf, params['crop_left'], params['crop_right'])
        cropped_waveforms.append(cropped_wf)
        absolute_offsets[i] = abs_offset

        # 计算 PP
        noise_floor = np.mean(cropped_wf[10:20])
        valid_bins = cropped_wf[cropped_wf > noise_floor]
        if len(valid_bins) > 0:
            p_mean = np.mean(valid_bins)
            p_max = np.max(cropped_wf)
            pp = p_max / p_mean if p_mean > 0 else 0
        else:
            pp = 0
        custom_pps[i] = pp

        # 铅/冰 分类
        if pp > params['pp_lead_threshold'] and ssd < params['ssd_threshold']:
            types[i] = 'lead'
        elif pp < params['pp_ice_threshold'] and ssd > params['ssd_threshold']:
            types[i] = 'ice'

    return types, custom_pps, absolute_offsets, np.array(cropped_waveforms)

# =====================================================================
# 模块 4: 重跟踪 (Retracking)
# =====================================================================
def retrack_diffuse(cropped_wfs, offsets, params):
    """漫反射 (Ice) - TFMRA 70%"""
    n = len(cropped_wfs)
    correction = np.full(n, np.nan)
    for i in range(n):
        wf = cropped_wfs[i]
        if params['needs_smoothing']:
            wf = scipy.ndimage.uniform_filter1d(wf, size=3)
            
        max_val = np.max(wf)
        if max_val <= 0: continue
        peaks, _ = scipy.signal.find_peaks(wf)
        valid_peaks = [p for p in peaks if wf[p] >= params['first_peak_min_ratio'] * max_val]
        if not valid_peaks: continue
        
        first_peak_idx = valid_peaks[0]
        first_peak_val = wf[first_peak_idx]
        threshold_70 = params['retrack_threshold'] * first_peak_val
        
        b_0_local = np.nan
        for j in range(first_peak_idx - 1, -1, -1):
            if wf[j] <= threshold_70 <= wf[j + 1]:
                frac = (threshold_70 - wf[j]) / (wf[j + 1] - wf[j])
                b_0_local = j + frac
                break
                
        if not np.isnan(b_0_local):
            correction[i] = (offsets[i] + b_0_local - params['bn']) * params['bin_size'] - 0.1626 # Tilling 固定偏差
    return correction

def giles_echo_function(t, a, t0, k, sigma):
    sigma, k = max(sigma, 1e-6), max(k, 1e-6)
    tb = k * (sigma ** 2)
    sqrt_k_tb = np.sqrt(k * tb)
    a2 = (5 * k * sigma - 4 * sqrt_k_tb) / (2 * sigma * tb * sqrt_k_tb)
    a3 = (2 * sqrt_k_tb - 3 * k * sigma) / (2 * sigma * (tb ** 2) * sqrt_k_tb)
    
    f = np.zeros_like(t, dtype=float)
    mask1 = t < t0
    f[mask1] = (t[mask1] - t0) / sigma
    mask2 = (t >= t0) & (t < (tb + t0))
    dt2 = t[mask2] - t0
    f[mask2] = a3 * (dt2 ** 3) + a2 * (dt2 ** 2) + (1 / sigma) * dt2
    mask3 = t >= (tb + t0)
    dt3 = np.maximum(t[mask3] - t0, 0)
    f[mask3] = np.sqrt(k * dt3)
    return a * np.exp(-(f ** 2))

def retrack_specular(cropped_wfs, offsets, params):
    """镜面 (Lead) - Giles 拟合"""
    n = len(cropped_wfs)
    correction = np.full(n, np.nan)
    t_128 = np.arange(128, dtype=float)
    
    for i in range(n):
        wf = cropped_wfs[i]
        try:
            popt, _ = curve_fit(giles_echo_function, t_128, wf, 
                                p0=[np.max(wf), float(np.argmax(wf)), 0.5, 1.0], 
                                method='lm', maxfev=3000)
            t0_local = popt[1]
            if 0 <= t0_local <= 127:
                correction[i] = (offsets[i] + t0_local - params['bn']) * params['bin_size']
        except:
            pass
    return correction

# =====================================================================
# 模块 5: 辅助数据加载 (带进程内缓存)
# =====================================================================
_MSS_CACHE = {}
_SNOW_CACHE = {}

def load_mss(lon_series, lat_series, mss_path):
    if not mss_path or not os.path.exists(mss_path):
        return np.full(len(lon_series), np.nan)
        
    if 'data' not in _MSS_CACHE:
        ds = xr.open_dataset(mss_path)
        lon_mss = ds['lon'].values
        lon_mss[lon_mss > 180] -= 360
        _MSS_CACHE['ds'] = ds
        _MSS_CACHE['lon'] = lon_mss
        _MSS_CACHE['lat'] = ds['lat'].values
        
    ds = _MSS_CACHE['ds']
    min_lon, max_lon = lon_series.min(), lon_series.max()
    min_lat, max_lat = lat_series.min(), lat_series.max()
    
    mss_clipped = ds.where(
        (ds.lon >= min_lon) & (ds.lon <= max_lon) &
        (ds.lat >= min_lat) & (ds.lat <= max_lat), drop=True
    )
    if mss_clipped.lon.size == 0 or mss_clipped.lat.size == 0:
        return np.full(len(lon_series), np.nan)
        
    lon_grid, lat_grid = np.meshgrid(mss_clipped.lon.values, mss_clipped.lat.values)
    return griddata(
        (lon_grid.flatten(), lat_grid.flatten()),
        mss_clipped.mean_sea_surf_sol2.values.flatten(),
        (lon_series, lat_series), method='linear'
    )

def load_snow(month, lon_series, lat_series, snow_dir):
    """Warren/AMSR2 月度雪深融合数据"""
    if not snow_dir: return np.full(len(lon_series), np.nan), np.full(len(lon_series), np.nan)
    
    cache_key = f"snow_{month}"
    if cache_key not in _SNOW_CACHE:
        pattern = os.path.join(snow_dir, f"awi-siral-l4-snow_on_seaice-monthly_warren_amsr2_clim-{month:02d}-*.nc")
        files = glob.glob(pattern)
        if not files:
            return np.full(len(lon_series), np.nan), np.full(len(lon_series), np.nan)
        
        ds = xr.open_dataset(files[0])
        lon_grid_s, lat_grid_s = np.meshgrid(ds['lon'].values, ds['lat'].values)
        lon_grid_180 = ((lon_grid_s + 180) % 360) - 180
        
        snow_grid = ds['merged_snow_depth'].values
        w_name = 'w99_weight'
        weight_grid = ds[w_name].values if w_name in ds.data_vars else np.ones_like(snow_grid)
        
        valid = ~np.isnan(snow_grid)
        _SNOW_CACHE[cache_key] = {
            'pts': (lon_grid_180[valid], lat_grid_s[valid]),
            'snow': snow_grid[valid],
            'weight': weight_grid[valid]
        }
        
    cache = _SNOW_CACHE[cache_key]
    pts_lon_180 = ((lon_series.values + 180) % 360) - 180
    
    sd = griddata(cache['pts'], cache['snow'], (pts_lon_180, lat_series), method='linear')
    w = griddata(cache['pts'], cache['weight'], (pts_lon_180, lat_series), method='linear')
    
    # NaN 补全 (Nearest)
    nan_mask = np.isnan(sd)
    if nan_mask.any():
        sd[nan_mask] = griddata(cache['pts'], cache['snow'], (pts_lon_180[nan_mask], lat_series[nan_mask]), method='nearest')
        w[nan_mask]  = griddata(cache['pts'], cache['weight'], (pts_lon_180[nan_mask], lat_series[nan_mask]), method='nearest')
        
    return sd, w

def load_sic_and_icetype(date_obj, lon_series, lat_series, sic_dir, icetype_dir):
    """由于缺失路径目前设为 None，返回全 NaN"""
    n = len(lon_series)
    # TODO: 当有具体数据路径时，在此处通过 YYYYMMDD 匹配 nc 文件，并使用 griddata 提取 SIC 和 f_myi
    return np.full(n, np.nan), np.full(n, 0.0) # 默认 f_myi 为 0.0 (FYI)

# =====================================================================
# 模块 6: SLA 计算与 200km 掩膜
# =====================================================================
def compute_sla(df):
    lead_mask = df['type'] == 'lead'
    df['SLA_raw'] = np.nan
    df.loc[lead_mask, 'SLA_raw'] = df.loc[lead_mask, 'h_surface'] - df.loc[lead_mask, 'mss_interp']
    
    valid_lead = lead_mask & (df['SLA_raw'] >= -3.0) & (df['SLA_raw'] <= 3.0)
    valid_lead_indices = df.index[valid_lead].to_numpy()
    valid_SLA_raw_values = df.loc[valid_lead, 'SLA_raw'].to_numpy()
    
    df['SLA'] = np.nan
    df['dist_to_lead'] = np.inf
    
    if len(valid_SLA_raw_values) > 1:
        # 100km 平滑
        window_100km = int(100000 / 380) # 假设沿着轨间距约 380m
        sla_series = pd.Series(valid_SLA_raw_values, index=valid_lead_indices)
        sla_smoothed = sla_series.rolling(window=window_100km, center=True, min_periods=1).mean()
        
        # 线性插值
        interp_func = interp1d(sla_smoothed.index, sla_smoothed.values, kind='linear', bounds_error=False, fill_value=np.nan)
        df['SLA'] = interp_func(df.index)
        
        # 200km 掩膜计算
        lead_coords_rad = np.radians(df.loc[valid_lead, ['lat', 'lon']].to_numpy())
        all_coords_rad = np.radians(df[['lat', 'lon']].to_numpy())
        if len(lead_coords_rad) > 0:
            tree = cKDTree(lead_coords_rad)
            dist_rad, _ = tree.query(all_coords_rad, k=1)
            df['dist_to_lead'] = dist_rad * 6371000.0
            
        df.loc[df['dist_to_lead'] > 200000, 'SLA'] = np.nan
        
    return df

# =====================================================================
# 模块 7: 单文件主流程
# =====================================================================
def process_one_file(file_path, aux_paths, output_dir):
    track_id = os.path.basename(file_path)
    
    try:
        P = get_processing_params(file_path)
        ds = xr.open_dataset(file_path, decode_timedelta=True)
        
        # 获取 20Hz 数据
        time_20 = ds['time_20_ku'].values
        lat_20 = ds['lat_20_ku'].values
        lon_20 = ds['lon_20_ku'].values
        waveform = ds['pwr_waveform_20_ku'].values
        noise = ds['noise_power_20_ku'].values
        noise = np.where(noise == -9999.99, np.nan, noise)
        
        # 扩展 1Hz 数据到 20Hz
        ind_first = ds["ind_first_meas_20hz_01"].values.astype(int)
        N_20 = len(time_20)
        
        correction_vars = ['mod_dry_tropo_cor_01', 'mod_wet_tropo_cor_01', 'inv_bar_cor_01',
                           'iono_cor_01', 'ocean_tide_01', 'ocean_tide_eq_01', 'load_tide_01',
                           'solid_earth_tide_01', 'pole_tide_01']
        
        corrections_20hz = {var: np.full(N_20, np.nan) for var in correction_vars}
        surf_type = np.full(N_20, np.nan)
        
        for i, start_idx in enumerate(ind_first):
            end_idx = min(start_idx + 20, N_20)
            surf_type[start_idx:end_idx] = ds["surf_type_01"].values[i]
            for var in correction_vars:
                corrections_20hz[var][start_idx:end_idx] = ds[var].values[i]
                
        df = pd.DataFrame({
            'track_id': track_id,
            'time_20_ku': time_20,
            'lat': lat_20, 'lon': lon_20,
            'alt': ds['alt_20_ku'].values,
            'window_del': ds['window_del_20_ku'].dt.total_seconds().values,
            'noise_power_real': noise,
            'std': ds['stack_std_20_ku'].values,
            'surf_type': surf_type
        })
        
        tai_epoch = datetime(2000, 1, 1)
        tai_sec = (time_20 - np.datetime64('2000-01-01T00:00:00')) / np.timedelta64(1, 's')
        df['utc_time'] = [tai_epoch + timedelta(seconds=t) for t in tai_sec]
        
        for var in correction_vars: df[var] = corrections_20hz[var]
        df['corrections_sum'] = sum(df[var] for var in correction_vars)
        
        # 去噪与分类
        waveform_clean = waveform - np.nan_to_num(noise, nan=0.0)[:, np.newaxis]
        types, pps, offsets, cropped_wfs = classify_waveforms(waveform_clean, df['std'].values, P)
        df['type'] = types
        df['pp'] = pps
        df['abs_offset'] = offsets
        
        # 仅处理海洋区域 (0 或 1)
        ocean_mask = df['surf_type'].isin([0, 1])
        lead_mask = (df['type'] == 'lead') & ocean_mask
        ice_mask = (df['type'] == 'ice') & ocean_mask
        
        # 重跟踪
        df['range_correction'] = np.nan
        if lead_mask.any():
            df.loc[lead_mask, 'range_correction'] = retrack_specular(cropped_wfs[lead_mask], offsets[lead_mask], P)
        if ice_mask.any():
            df.loc[ice_mask, 'range_correction'] = retrack_diffuse(cropped_wfs[ice_mask], offsets[ice_mask], P)
            
        # 高程计算
        df['range_final'] = df['window_del'] * P['c'] / 2 + df['range_correction']
        df['h_surface'] = df['alt'] - df['range_final'] - df['corrections_sum']
        
        # 辅助数据 & 物理量计算
        df['mss_interp'] = load_mss(df['lon'], df['lat'], aux_paths.get('mss'))
        df = compute_sla(df)
        
        df['radar_freeboard'] = df['h_surface'] - (df['mss_interp'] + df['SLA'])
        df.loc[(df['type'] != 'ice') | (df['radar_freeboard'] < -0.25) | (df['radar_freeboard'] > 2.25), 'radar_freeboard'] = np.nan
        
        if not df.empty:
            file_date = df['utc_time'].iloc[len(df)//2]
            sd, w = load_snow(file_date.month, df['lon'], df['lat'], aux_paths.get('snow'))
            sic, f_myi = load_sic_and_icetype(file_date, df['lon'], df['lat'], aux_paths.get('sic'), aux_paths.get('ice_type'))
        else:
            sd, w, sic, f_myi = np.nan, np.nan, np.nan, 0.0
            
        df['snow_depth_raw'] = sd
        df['sic'] = sic
        df['f_myi'] = f_myi
        df['ice_type'] = np.where(f_myi > 0.5, 'MYI', 'FYI') # 简化表示
        
        # 雪深修正与干舷
        df['hs'] = sd - (1 - f_myi) * 0.5 * w * sd
        df['corrected_freeboard'] = df['radar_freeboard'] + 0.25 * df['hs']
        df.loc[(df['corrected_freeboard'] < -0.3) | (df['corrected_freeboard'] > 3.0), 'corrected_freeboard'] = np.nan
        
        # 密度标记与厚度
        df['density_water_used'] = 1023.9
        df['density_snow_used'] = 324.0
        df['density_ice_used'] = df['f_myi'] * 882.0 + (1 - df['f_myi']) * 916.7
        
        df['sea_ice_thickness'] = (
            (df['corrected_freeboard'] * df['density_water_used'] + df['hs'] * df['density_snow_used']) / 
            (df['density_water_used'] - df['density_ice_used'])
        )

        # 构建最终列顺序
        columns_out = [
            'track_id', 'time_20_ku', 'utc_time', 'lat', 'lon', 
            'surf_type', 'type', 'pp', 'std',
            'alt', 'window_del', 'noise_power_real',
            'corrections_sum'
        ] + correction_vars + [
            'abs_offset', 'range_correction', 'range_final',
            'h_surface', 'mss_interp', 'SLA', 'dist_to_lead',
            'radar_freeboard', 'corrected_freeboard',
            'snow_depth_raw', 'hs', 'sic', 'ice_type', 'f_myi',
            'density_ice_used', 'density_snow_used', 'density_water_used',
            'sea_ice_thickness'
        ]
        
        out_path = os.path.join(output_dir, track_id.replace('.nc', '.parquet'))
        df[columns_out].to_parquet(out_path, index=False)
        return True, file_path
    
    except Exception as e:
        return False, f"{file_path}: {str(e)}"

# =====================================================================
# 模块 8 & 9: 批量调度与入口
# =====================================================================
def worker_wrapper(args):
    return process_one_file(*args)

def batch_process(input_dir, output_dir, aux_paths, n_workers=6):
    os.makedirs(output_dir, exist_ok=True)
    nc_files = glob.glob(os.path.join(input_dir, "*.nc"))
    
    if not nc_files:
        print(f"在 {input_dir} 中没有找到 .nc 文件。")
        return
        
    print(f"发现 {len(nc_files)} 个文件。启动 {n_workers} 个 Worker 进行处理...")
    
    tasks = [(f, aux_paths, output_dir) for f in nc_files]
    
    if n_workers == 1:
        # 单线程调试
        for task in tasks:
            status, msg = worker_wrapper(task)
            print(f"{'SUCCESS' if status else 'FAILED'}: {msg}")
    else:
        # 多进程处理
        with mp.Pool(n_workers) as pool:
            results = pool.imap_unordered(worker_wrapper, tasks)
            for status, msg in results:
                print(f"{'SUCCESS' if status else 'FAILED'}: {msg}")

if __name__ == "__main__":
    # ==== 配置区 ====
    INPUT_DIR = r"E:\Arctic\2024\01"
    OUTPUT_DIR = r"E:\Arctic\2024\01\Processed_Parquet_gauss"
    
    AUX_PATHS = {
        'mss': r"E:\Project_2024\CryoSat-2 L1\DTU21MSS_1min_WGS84.nc",
        'snow': r"D:\S1_CS2_data\awi_snow_merged",
        'sic': None,       # 设置为 None 会自动填充 NaN
        'ice_type': None   # 设置为 None 会自动填充默认的一年冰 (f_myi=0)
    }
    
    # 调试时请保持为 1，确认无误后可根据 CPU 核心数调大（如 4 或 8）
    N_WORKERS = 6
    # ================
    
    batch_process(INPUT_DIR, OUTPUT_DIR, AUX_PATHS, N_WORKERS)