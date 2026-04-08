import os
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal
import scipy.ndimage
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator, interp1d, griddata
import multiprocessing as mp
#############################
#相较1增加了glob读取多个文件夹层级
############################
# 忽略数值计算产生的警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=scipy.optimize.OptimizeWarning)

# =====================================================================
# 模块 1: 参数配置与全局缓存
# =====================================================================
_INTERP_CACHE = {} 
_SNOW_CACHE = {}   

def get_processing_params(file_path):
    filename = os.path.basename(file_path)
    mode = "SARIn" if "SIR_SIN" in filename else "SAR"
    return {
        "mode": mode,
        "c": 299792458.0,
        "bin_size": 0.2342,
        "bn": 512 if mode == "SARIn" else 128, #bn 是波形中心点在窗口中的位置，SARIn 模式下为 512，SAR 模式下为 128
        "crop_left": 50,
        "crop_right": 77,
        "lead_retracker_mode": "threshold", # 可选: "threshold" (50% TFMRA) 或 "giles" (拟合)
        "ice_retrack_threshold": 0.70,     # Ice: TFMRA 70%
        "lead_retrack_threshold": 0.50,    # Lead: TFMRA 50%
        "pp_lead_threshold": 18,
        "pp_ice_threshold": 9,
        "ssd_threshold": 4.62 if mode == "SARIn" else 6.29,
        "needs_smoothing": True if mode == "SARIn" else False
    }

# =====================================================================
# 模块 2: 重跟踪与分类算子
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

def giles_echo_function(t, a, t0, k, sigma):
    sigma, k = max(sigma, 1e-6), max(k, 1e-6)
    tb = k * (sigma**2); sqrt_k_tb = np.sqrt(k * tb)
    a2 = (5*k*sigma - 4*sqrt_k_tb)/(2*sigma*tb*sqrt_k_tb)
    a3 = (2*sqrt_k_tb - 3*k*sigma)/(2*sigma*(tb**2)*sqrt_k_tb)
    f = np.zeros_like(t)
    m1, m2, m3 = (t < t0), ((t >= t0) & (t < tb+t0)), (t >= tb+t0)
    f[m1] = (t[m1]-t0)/sigma
    dt2 = t[m2]-t0; f[m2] = a3*(dt2**3) + a2*(dt2**2) + dt2/sigma
    dt3 = np.maximum(t[m3]-t0, 0); f[m3] = np.sqrt(k*dt3)
    return a * np.exp(-(f**2))

def retrack_wf(wf, offset, p_type, P):
    if p_type == 'ice':
        if P['needs_smoothing']: wf = scipy.ndimage.uniform_filter1d(wf, size=3)
        max_v = np.max(wf)
        peaks, _ = scipy.signal.find_peaks(wf)
        valid = [p for p in peaks if wf[p] >= 0.20 * max_v]
        if not valid: return np.nan
        f_p = valid[0]
        target = wf[f_p] * P['ice_retrack_threshold']
        for j in range(f_p, 0, -1):
            if wf[j-1] <= target <= wf[j]:
                t0 = (j-1) + (target - wf[j-1])/(wf[j] - wf[j-1] + 1e-9)
                return (offset + t0 - P['bn']) * P['bin_size'] - 0.1626
    elif p_type == 'lead':
        if P['lead_retracker_mode'] == "threshold":
            m_idx = np.argmax(wf); target = wf[m_idx] * P['lead_retrack_threshold']
            for j in range(m_idx, 0, -1):
                if wf[j-1] <= target <= wf[j]:
                    t0 = (j-1) + (target - wf[j-1])/(wf[j] - wf[j-1] + 1e-9)
                    return (offset + t0 - P['bn']) * P['bin_size']
        else:
            try:
                popt, _ = curve_fit(giles_echo_function, np.arange(len(wf)), wf, 
                                    p0=[np.max(wf), float(np.argmax(wf)), 0.5, 1.0], maxfev=1000)
                return (offset + popt[1] - P['bn']) * P['bin_size']
            except: pass
    return np.nan

# =====================================================================
# 模块 3: 辅助数据加速加载 (MSS & Snow)
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
    
    nan_m = df['merged_sd_raw'].isna()
    if nan_m.any():
        df.loc[nan_m, 'merged_sd_raw'] = griddata(c['pts'], c['sd'], (q_lon[nan_m], q_lat[nan_m]), method='nearest')
        df.loc[nan_m, 'w99_weight'] = griddata(c['pts'], c['w'], (q_lon[nan_m], q_lat[nan_m]), method='nearest')
    
    df['f_myi'] = 0.0 # 默认 FYI
    df['hs'] = df['merged_sd_raw'] - (1 - df['f_myi']) * 0.5 * df['w99_weight'] * df['merged_sd_raw']
    return df

# =====================================================================
# 模块 4: SLA 计算 (100km & 200km Mask)
# =====================================================================
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
# 模块 5: 单文件处理主函数
# =====================================================================
def process_one_file(file_path, aux_paths, output_dir):
    track_id = os.path.basename(file_path)
    try:
        P = get_processing_params(file_path)
        ds = xr.open_dataset(file_path, decode_timedelta=True)
        
        lat, lon = ds['lat_20_ku'].values, ds['lon_20_ku'].values
        wf = ds['pwr_waveform_20_ku'].values
        noise = np.where(ds['noise_power_20_ku'].values == -9999.99, 0, ds['noise_power_20_ku'].values)
        wf_clean = wf - noise[:, np.newaxis]
        
        # 扩展 1Hz 到 20Hz
        ind_first, N_20 = ds["ind_first_meas_20hz_01"].values.astype(int), len(lat)
        c_vars = ['mod_dry_tropo_cor_01', 'mod_wet_tropo_cor_01', 'inv_bar_cor_01', 'iono_cor_01', 
                  'ocean_tide_01', 'ocean_tide_eq_01', 'load_tide_01', 'solid_earth_tide_01', 'pole_tide_01']
        corr_20 = {v: np.full(N_20, np.nan) for v in c_vars}; surf_20 = np.full(N_20, np.nan)
        for i, start in enumerate(ind_first):
            end = min(start + 20, N_20)
            surf_20[start:end] = ds["surf_type_01"].values[i]
            for v in c_vars: corr_20[v][start:end] = ds[v].values[i]
            
        df = pd.DataFrame({'track_id': track_id, 'lat': lat, 'lon': lon, 'surf_type': surf_20,
                           'alt': ds['alt_20_ku'].values, 'window_del': ds['window_del_20_ku'].dt.total_seconds().values,
                           'std': ds['stack_std_20_ku'].values})
        for v in c_vars: df[v] = corr_20[v]
        df['corrections_sum'] = df[c_vars].sum(axis=1)
        df['utc_time'] = datetime(2000, 1, 1) + pd.to_timedelta((ds['time_20_ku'].values - np.datetime64('2000-01-01T00:00:00')) / np.timedelta64(1, 's'), unit='s')

        # 分类与重跟踪
        df['type'], df['pp'] = classify_waveforms_vectorized(wf_clean, df['std'].values, P)
        b_max = np.argmax(wf_clean, axis=1); df['abs_offset'] = b_max - P['crop_left']
        
        df['range_correction'] = np.nan
        ocean_mask = df['surf_type'].isin([0, 1])
        for i in df.index[ocean_mask]:
            start, end = int(b_max[i]-P['crop_left']), int(b_max[i]+P['crop_right']+1)
            wf_seg = wf_clean[i, max(0, start):min(wf_clean.shape[1], end)]
            df.loc[i, 'range_correction'] = retrack_wf(wf_seg, df.loc[i, 'abs_offset'], df.loc[i, 'type'], P)

        # 物理量计算链
        df['range_final'] = df['window_del'] * P['c'] / 2 + df['range_correction']
        df['h_surface'] = df['alt'] - df['range_final'] - df['corrections_sum']
        df['mss_interp'] = load_mss_fast(df['lon'], df['lat'], aux_paths.get('mss'))
        df = compute_sla_chain(df)
        
        df['radar_freeboard'] = df['h_surface'] - (df['mss_interp'] + df['SLA'])
        df.loc[(df['type'] != 'ice') | (~df['radar_freeboard'].between(-0.25, 2.25)), 'radar_freeboard'] = np.nan
        
        df = load_snow_and_fix(df, aux_paths.get('snow'))
        df['corrected_freeboard'] = df['radar_freeboard'] + 0.25 * df['hs']
        df.loc[~df['corrected_freeboard'].between(-0.3, 3.0), 'corrected_freeboard'] = np.nan
        
        rho_w, rho_s = 1023.9, 324.0
        df['rho_i'] = df['f_myi'] * 882.0 + (1 - df['f_myi']) * 916.7
        df['sea_ice_thickness'] = (df['corrected_freeboard'] * rho_w + df['hs'] * rho_s) / (rho_w - df['rho_i'])

        # 导出 Parquet
        out_path = os.path.join(output_dir, track_id.replace('.nc', '.parquet'))
        df.to_parquet(out_path, index=False)
        return True, file_path
    except Exception as e:
        return False, f"{track_id}: {str(e)}"

# =====================================================================
# 模块 8 & 9: 批量调度与入口 (已找回并集成)
# =====================================================================
def worker_wrapper(args):
    return process_one_file(*args)

def batch_process(root_input_dir, root_output_dir, aux_paths, n_workers=4):
    """
    遍历 root_input_dir 下的所有子文件夹（yyyy/mm/），处理找到的 .nc 文件，
    并在 root_output_dir 中保留相同的目录结构。
    """
    # 使用 **/*.nc 配合 recursive=True 递归查找所有子目录下的文件
    search_pattern = os.path.join(root_input_dir, "**", "*.nc")
    nc_files = glob.glob(search_pattern, recursive=True)
    
    if not nc_files:
        print(f"在 {root_input_dir} 及其子目录中没有找到 .nc 文件。")
        return
        
    print(f"发现 {len(nc_files)} 个文件。启动 {n_workers} 个 Worker 进行处理...")
    
    tasks = []
    for f in nc_files:
        # 计算相对路径，以便在输出目录重建相同的 yyyy/mm 结构
        rel_dir = os.path.relpath(os.path.dirname(f), root_input_dir)
        current_output_dir = os.path.join(root_output_dir, rel_dir)
        
        # 预先创建该层级的输出文件夹
        os.makedirs(current_output_dir, exist_ok=True)
        
        tasks.append((f, aux_paths, current_output_dir))
    
    # 进程池处理
    if n_workers == 1:
        for task in tasks:
            status, msg = worker_wrapper(task)
            print(f"{'SUCCESS' if status else 'FAILED'}: {msg}")
    else:
        with mp.Pool(n_workers) as pool:
            # 使用 imap_unordered 提高响应速度
            results = pool.imap_unordered(worker_wrapper, tasks)
            for status, msg in results:
                print(f"{'SUCCESS' if status else 'FAILED'}: {msg}")

if __name__ == "__main__":
    # ==== 配置区 ====
    # 现在指向根目录，例如 E:\Arctic
    # 程序会自动处理 E:\Arctic\2024\01\*.nc, E:\Arctic\2024\02\*.nc 等
    INPUT_DIR = r"E:\Arctic\SAR L1" 
    OUTPUT_DIR = r"E:\Arctic_Processed_L1_Results"
    
    AUX_PATHS = {
        'mss': r"E:\Project_2024\CryoSat-2 L1\DTU21MSS_1min_WGS84.nc",
        'snow': r"D:\S1_CS2_data\awi_snow_merged",
        'sic': None,
        'ice_type': None
    }
    
    N_WORKERS = 8 
    # ================
    
    batch_process(INPUT_DIR, OUTPUT_DIR, AUX_PATHS, N_WORKERS)