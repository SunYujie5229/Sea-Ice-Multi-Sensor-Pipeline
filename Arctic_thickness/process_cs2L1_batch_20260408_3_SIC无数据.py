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
from datetime import datetime
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator, interp1d, griddata

# 忽略计算警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =====================================================================
# 1. 配置与辅助函数
# =====================================================================
_CACHE = {"mss": {}, "snow": {}}

def get_params(file_path):
    mode = "SARIn" if "SIR_SIN" in os.path.basename(file_path) else "SAR"
    return {
        "mode": mode, "c": 299792458.0, "bin_size": 0.2342,
        "bn": 512 if mode == "SARIn" else 128,
        "pp_lead_threshold": 18, "pp_ice_threshold": 9,
        "ssd_threshold": 4.62 if mode == "SARIn" else 6.29,
        "ice_retrack": 0.70, "lead_retrack": 0.50,
        "crop_left": 50, "crop_right": 77,
        "needs_smoothing": (mode == "SARIn")
    }

# =====================================================================
# 2. 核心算法算子
# =====================================================================
def classify_waveforms(wf, ssd, P):
    p_max = np.max(wf, axis=1)
    # 计算 pp (Pulse Peakiness)
    noise_floor = np.mean(wf[:, 10:20], axis=1)
    masked = np.where(wf > noise_floor[:, np.newaxis], wf, 0)
    pp = np.divide(p_max, (masked.sum(axis=1) / (masked > 0).sum(axis=1)), 
                   out=np.zeros_like(p_max), where=(masked > 0).sum(axis=1) > 0)
    
    types = np.full(wf.shape[0], 'unknown', dtype=object)
    types[(pp > P['pp_lead_threshold']) & (ssd < P['ssd_threshold'])] = 'lead'
    types[(pp < P['pp_ice_threshold']) & (ssd > P['ssd_threshold'])] = 'ice'
    return types, pp

def retrack_wf(wf, p_type, P):
    if p_type not in ['ice', 'lead'] or np.isnan(wf).all(): return np.nan
    
    try:
        if p_type == 'ice':
            if P['needs_smoothing']: wf = scipy.ndimage.uniform_filter1d(wf, size=3)
            peaks, _ = scipy.signal.find_peaks(wf, height=np.max(wf)*0.2)
            if not peaks.any(): return np.nan
            f_p = peaks[0]
            target = wf[f_p] * P['ice_retrack']
        else: # lead
            f_p = np.argmax(wf)
            target = wf[f_p] * P['lead_retrack']

        for j in range(f_p, 0, -1):
            if wf[j-1] <= target <= wf[j]:
                return (j-1) + (target - wf[j-1])/(wf[j] - wf[j-1] + 1e-9)
    except: pass
    return np.nan

# =====================================================================
# 3. 数据采样模块 (SIC / MSS / Snow)
# =====================================================================
def sample_sic(df, sic_dir):
    dt = df['utc_time'].iloc[0]
    date_str = dt.strftime("%Y%m%d")
    path = os.path.join(sic_dir, str(dt.year), "n6250", f"*{date_str}*.tif*")
    files = glob.glob(path)
    
    if not files: return np.full(len(df), np.nan), "MISSING"
    
    try:
        with rasterio.open(files[0]) as src:
            trans = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xx, yy = trans.transform(df['lon'].values, df['lat'].values)
            samples = np.array([v[0] for v in src.sample(zip(xx, yy))]).astype(float)
            samples[samples > 100] = np.nan # 过滤 255 等填充值
            return samples, "OK"
    except Exception as e:
        return np.full(len(df), np.nan), f"ERROR:{str(e)[:20]}"

def load_mss(lons, lats, mss_path):
    if not mss_path or not os.path.exists(mss_path): return np.full(len(lons), np.nan)
    if mss_path not in _CACHE["mss"]:
        ds = xr.open_dataset(mss_path)
        if ds.lon.max() > 180: ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
        _CACHE["mss"][mss_path] = RegularGridInterpolator((ds.lat.values, ds.lon.values), 
                                                           ds['mean_sea_surf_sol2'].values, 
                                                           bounds_error=False, fill_value=np.nan)
    return _CACHE["mss"][mss_path](np.column_stack((lats, lons)))

# =====================================================================
# 4. 主处理单元
# =====================================================================
def process_one_file(file_path, aux_paths, output_root, input_root):
    fname = os.path.basename(file_path)
    status_log = []
    
    try:
        P = get_params(file_path)
        ds = xr.open_dataset(file_path, decode_timedelta=True)
        
        # 1. 基础数据准备
        df = pd.DataFrame({
            'lat': ds['lat_20_ku'].values, 'lon': ds['lon_20_ku'].values,
            'alt': ds['alt_20_ku'].values, 'std': ds['stack_std_20_ku'].values,
            'window_del': ds['window_del_20_ku'].dt.total_seconds().values
        })
        df['utc_time'] = datetime(2000, 1, 1) + pd.to_timedelta((ds['time_20_ku'].values - np.datetime64('2000-01-01T00:00:00')) / np.timedelta64(1, 's'), unit='s')
        
        # 2. SIC 采样进度
        df['sic'], sic_status = sample_sic(df, aux_paths.get('sic'))
        status_log.append(f"SIC:{sic_status}")

        # 3. 波形处理
        wf = ds['pwr_waveform_20_ku'].values - np.maximum(ds['noise_power_20_ku'].values, 0)[:, np.newaxis]
        df['type'], df['pp'] = classify_waveforms(wf, df['std'].values, P)
        
        b_max = np.argmax(wf, axis=1)
        retrack_results = []
        for i in range(len(df)):
            wf_crop = wf[i, max(0, b_max[i]-50) : min(wf.shape[1], b_max[i]+78)]
            t0 = retrack_wf(wf_crop, df.iloc[i]['type'], P)
            offset = (b_max[i] - 50) + t0 - P['bn']
            # 应用改正量 (重定轨中心 + 物理常数)
            corr = offset * P['bin_size'] - (0.1626 if df.iloc[i]['type'] == 'ice' else 0)
            retrack_results.append(corr)
        df['range_corr'] = retrack_results

        # 4. 高程链条计算
        df['h_surface'] = df['alt'] - (df['window_del'] * P['c'] / 2 + df['range_corr'])
        df['mss'] = load_mss(df['lon'].values, df['lat'].values, aux_paths.get('mss'))
        
        # SLA 逻辑
        df['SLA_raw'] = np.where(df['type']=='lead', df['h_surface'] - df['mss'], np.nan)
        valid_sla = df['SLA_raw'].dropna().loc[df['SLA_raw'].between(-3, 3)]
        if len(valid_sla) > 2:
            sla_interp = interp1d(valid_sla.index, valid_sla.rolling(250, center=True, min_periods=1).mean(), 
                                  bounds_error=False)(df.index)
            df['SLA'] = sla_interp
            status_log.append("SLA:OK")
        else:
            df['SLA'] = np.nan
            status_log.append("SLA:FAIL")

        # 5. 海冰厚度反演 (此处简化演示，具体物理公式保持你原有的)
        df['radar_fb'] = df['h_surface'] - (df['mss'] + df['SLA'])
        # (雪深加载 load_snow_and_fix 逻辑建议放在此处...)
        
        # 保存结果
        rel_path = os.path.relpath(os.path.dirname(file_path), input_root)
        out_dir = os.path.join(output_root, rel_path)
        os.makedirs(out_dir, exist_ok=True)
        df.to_parquet(os.path.join(out_dir, fname.replace('.nc', '.parquet')), index=False)
        
        print(f"  [SUCCESS] {fname} | {' | '.join(status_log)}")
        return True
    except Exception as e:
        print(f"  [FAILED]  {fname} | Error: {str(e)}")
        return False

# =====================================================================
# 5. 执行入口
# =====================================================================
def main():
    INPUT_DIR = r"E:\Arctic\SAR L1\2024\01"
    OUTPUT_DIR = r"E:\Arctic\SAR L1\L1_processed"
    AUX = {
        'mss': r"E:\Project_2024\CryoSat-2 L1\DTU21MSS_1min_WGS84.nc",
        'snow': r"D:\S1_CS2_data\awi_snow_merged",
        'sic': r"D:\S1_CS2_data\SIC"
    }

    files = glob.glob(os.path.join(INPUT_DIR, "**", "*.nc"), recursive=True)
    print(f"Starting pipeline. Total files: {len(files)}")
    print("-" * 60)

    # 建议先测试单进程，确认输出信息无误
    # process_one_file(files[0], AUX, OUTPUT_DIR, INPUT_DIR)

    with mp.Pool(processes=8) as pool:
        tasks = [(f, AUX, OUTPUT_DIR, INPUT_DIR) for f in files]
        pool.starmap(process_one_file, tasks)

if __name__ == "__main__":
    main()