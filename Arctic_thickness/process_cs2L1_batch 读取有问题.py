"""
process_cs2_batch.py
====================
CryoSat-2 L1b 批处理脚本（SAR / SARIn 通用）
基于 Tilling et al. (2018) 算法

功能：
  - 自动扫描指定目录下所有 .nc 文件
  - 识别 SAR / SARIn 模式，自动切换参数
  - 全北极范围处理（不做区域裁剪）
  - 每个 nc 文件输出一个 .parquet 文件
  - 字段覆盖：位置/时间/分类/波形参数/重跟踪/高程/
              地球物理校正/MSS/SLA/雪深/SIC/ice_type/freeboard/SIT

用法：
  python process_cs2_batch.py

修改 CONFIG 区块中的路径即可，无需改动其他代码。
"""

# ============================================================
# ██████████  CONFIG  ████████████████████████████████████████
# ============================================================
CONFIG = {
    # ── 输入目录（扫描所有 .nc）──────────────────────────────
    "L1_DIR"     : r"E:\Arctic\2024\09",

    # ── 辅助数据路径 ─────────────────────────────────────────
    "MSS_FILE"   : r"E:\Project_2024\CryoSat-2 L1\DTU21MSS_1min_WGS84.nc",
    "SNOW_DIR"   : r"D:\S1_CS2_data\awi_snow_merged",
    # SIC 数据目录（NSIDC Bootstrap / CDR，每日一个 nc，文件名含 YYYYMMDD）
    # 若无 SIC 数据，设为 None，该字段填 NaN
    "SIC_DIR"    : None,   # 例：r"D:\SIC\2024"
    # OSI-SAF 冰型目录（每日一个 nc，文件名含 YYYYMMDD）
    # 若无，设为 None
    "ICE_TYPE_DIR": None,  # 例：r"D:\IceType\2024"

    # ── 输出目录 ─────────────────────────────────────────────
    "OUTPUT_DIR" : r"E:\Arctic\2024\09\parquet_out",

    # ── 并行进程数（CPU 核心数，1 = 串行）────────────────────
    "N_WORKERS"  : 4,

    # ── 全局纬度下限（不做经纬度裁剪，只过滤南界）───────────
    "MIN_LAT"    : 60.0,

    # ── 物理常量 ─────────────────────────────────────────────
    "RHO_W"      : 1023.9,   # 海水密度 kg/m³
    "RHO_S"      : 324.0,    # 雪密度 kg/m³  (春季参考值)
    "RHO_I_FYI"  : 916.7,    # 一年冰密度 kg/m³
    "RHO_I_MYI"  : 882.0,    # 多年冰密度 kg/m³
}
# ============================================================


import os
import gc
import glob
import logging
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal
import xarray as xr
from scipy.interpolate import griddata, interp1d
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ── 日志 ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================
# 1. 参数工厂
# ============================================================
def get_processing_params(file_path: str) -> dict:
    """
    根据文件名自动识别 SAR / SARIn 模式，
    返回 Tilling et al. (2018) 所需全部参数字典。
    """
    filename = os.path.basename(file_path)

    if "SIR_SIN" in filename:
        mode = "SARIn"
    elif "SIR_SAR" in filename:
        mode = "SAR"
    else:
        raise ValueError(f"无法从文件名识别模式: {filename}")

    params = dict(
        mode                = mode,
        c                   = 299792458.0,
        bin_size            = 0.2342,        # m / bin
        cropped_bins        = 128,
        crop_left           = 50,
        crop_right          = 77,
        retrack_threshold   = 0.70,
        first_peak_min_ratio= 0.20,
        pp_lead_threshold   = 18,
        pp_ice_threshold    = 9,
    )

    if mode == "SARIn":
        params.update(raw_bins=1024, bn=512,  ssd_threshold=4.62,
                      peak_threshold=0.45, needs_smoothing=True)
    else:
        params.update(raw_bins=256,  bn=128,  ssd_threshold=6.29,
                      peak_threshold=0.15, needs_smoothing=False)

    return params


# ============================================================
# 2. 波形工具函数
# ============================================================
def crop_waveform(waveform: np.ndarray,
                  crop_left: int = 50,
                  crop_right: int = 77):
    """
    将单条波形裁剪为 128 bins（crop_left + crop_right + 1）。
    返回 (cropped_wf, abs_offset)。
    abs_offset = b_max - crop_left（裁剪窗口左边界在原始数组中的绝对位置）。
    适用于任意长度原始波形（SAR=256，SARIn=1024）。
    """
    b_max        = int(np.argmax(waveform))
    start_index  = b_max - crop_left
    end_index    = b_max + crop_right + 1

    pad_left  = max(0, -start_index)
    pad_right = max(0, end_index - len(waveform))
    start_safe = max(0, start_index)
    end_safe   = min(len(waveform), end_index)

    cropped = waveform[start_safe:end_safe]
    if pad_left > 0 or pad_right > 0:
        cropped = np.pad(cropped, (pad_left, pad_right),
                         mode='constant', constant_values=0)

    abs_offset = b_max - crop_left
    return cropped, abs_offset


def compute_pp(cropped_wf: np.ndarray) -> float:
    """Pulse Peakiness（Tilling 2018 定义）。"""
    noise_floor = np.mean(cropped_wf[10:20])
    valid_bins  = cropped_wf[cropped_wf > noise_floor]
    if len(valid_bins) == 0:
        return 0.0
    p_mean = np.mean(valid_bins)
    p_max  = np.max(cropped_wf)
    return float(p_max / p_mean) if p_mean > 0 else 0.0


# ============================================================
# 3. 波形分类（PP + SSD）
# ============================================================
def classify_waveforms(waveform_array: np.ndarray,
                       ssd_array: np.ndarray,
                       params: dict):
    """
    返回 (types, pps, abs_offsets, cropped_waveforms)。
    types 元素为 'lead' | 'ice' | 'unknown'。
    """
    n          = len(waveform_array)
    crop_left  = params["crop_left"]
    crop_right = params["crop_right"]
    ssd_thr    = params["ssd_threshold"]
    pp_lead    = params["pp_lead_threshold"]
    pp_ice     = params["pp_ice_threshold"]

    types    = np.full(n, "unknown", dtype=object)
    pps      = np.full(n, np.nan)
    offsets  = np.full(n, np.nan)
    cropped  = []

    for i in range(n):
        wf  = waveform_array[i]
        ssd = ssd_array[i]

        cwf, off = crop_waveform(wf, crop_left, crop_right)
        cropped.append(cwf)
        offsets[i] = off

        pp = compute_pp(cwf)
        pps[i] = pp

        if pp > pp_lead and ssd < ssd_thr:
            types[i] = "lead"
        elif pp < pp_ice and ssd > ssd_thr:
            types[i] = "ice"

    return types, pps, offsets, np.array(cropped)


# ============================================================
# 4. 重跟踪
# ============================================================
def retrack_diffuse(waveform_array: np.ndarray, params: dict) -> np.ndarray:
    """
    漫反射回波（Ice）TFMRA 70% 前缘追踪。
    返回距离校正量 C_R (m)，NaN 表示该点无效。
    """
    n              = len(waveform_array)
    range_corr     = np.full(n, np.nan)
    b_n            = params["bn"]
    bin_size       = params["bin_size"]
    needs_smooth   = params["needs_smoothing"]
    crop_left      = params["crop_left"]
    crop_right     = params["crop_right"]
    peak_min_ratio = params["first_peak_min_ratio"]
    retrack_thr    = params["retrack_threshold"]

    for i in range(n):
        cwf, abs_offset = crop_waveform(waveform_array[i], crop_left, crop_right)

        wf_proc = scipy.ndimage.uniform_filter1d(cwf, size=3) if needs_smooth else cwf.copy()

        max_val = np.max(wf_proc)
        if max_val <= 0:
            continue

        peaks, _ = scipy.signal.find_peaks(wf_proc)
        valid_pk = [p for p in peaks if wf_proc[p] >= peak_min_ratio * max_val]
        if not valid_pk:
            continue

        fp_idx = valid_pk[0]
        fp_val = wf_proc[fp_idx]
        thr70  = retrack_thr * fp_val

        b0_local = np.nan
        for j in range(fp_idx - 1, -1, -1):
            if wf_proc[j] <= thr70 <= wf_proc[j + 1]:
                frac     = (thr70 - wf_proc[j]) / (wf_proc[j + 1] - wf_proc[j])
                b0_local = j + frac
                break

        if np.isnan(b0_local):
            continue

        b0_global      = abs_offset + b0_local
        range_corr[i]  = (b0_global - b_n) * bin_size

    return range_corr


def _giles_echo(t, a, t0, k, sigma):
    """Giles 分段镜面回波模型。"""
    sigma = max(sigma, 1e-6)
    k     = max(k, 1e-6)
    tb    = k * sigma ** 2
    sqkt  = np.sqrt(k * tb)
    a2    = (5 * k * sigma - 4 * sqkt) / (2 * sigma * tb * sqkt)
    a3    = (2 * sqkt - 3 * k * sigma) / (2 * sigma * tb ** 2 * sqkt)

    f = np.zeros_like(t, dtype=float)
    m1 = t < t0
    f[m1] = (t[m1] - t0) / sigma

    m2    = (t >= t0) & (t < t0 + tb)
    dt2   = t[m2] - t0
    f[m2] = a3 * dt2 ** 3 + a2 * dt2 ** 2 + dt2 / sigma

    m3    = t >= t0 + tb
    dt3   = np.maximum(t[m3] - t0, 0)
    f[m3] = np.sqrt(k * dt3)

    return a * np.exp(-(f ** 2))


def retrack_specular(waveform_array: np.ndarray, params: dict) -> np.ndarray:
    """
    镜面回波（Lead）Giles 曲线拟合重跟踪。
    返回距离校正量 C_R (m)。
    """
    n          = len(waveform_array)
    range_corr = np.full(n, np.nan)
    b_n        = params["bn"]
    bin_size   = params["bin_size"]
    crop_left  = params["crop_left"]
    crop_right = params["crop_right"]
    t128       = np.arange(128, dtype=float)

    for i in range(n):
        cwf, abs_offset = crop_waveform(waveform_array[i], crop_left, crop_right)
        p0 = [np.max(cwf), float(np.argmax(cwf)), 0.5, 1.0]
        try:
            popt, _ = curve_fit(_giles_echo, t128, cwf, p0=p0,
                                method="lm", maxfev=3000)
            t0_local = popt[1]
            if not (0 <= t0_local <= 127):
                continue
        except RuntimeError:
            continue

        b0_global     = abs_offset + t0_local
        range_corr[i] = (b0_global - b_n) * bin_size

    return range_corr


# ============================================================
# 5. 辅助数据加载（MSS、SIC、IceType、Snow）
# ============================================================
_MSS_CACHE: dict = {}   # 进程内全局缓存，避免重复读大文件

def load_mss(mss_file: str) -> tuple:
    """
    加载 DTU MSS 数据，返回 (lon_flat, lat_flat, mss_flat)。
    使用进程内缓存，每个进程只读一次。
    """
    global _MSS_CACHE
    if "data" in _MSS_CACHE:
        return _MSS_CACHE["data"]

    ds  = xr.open_dataset(mss_file)
    lon = ds["lon"].values.copy()
    lon[lon > 180] -= 360
    lat = ds["lat"].values

    lon_g, lat_g = np.meshgrid(lon, lat)
    mss_v        = ds["mean_sea_surf_sol2"].values
    valid        = ~np.isnan(mss_v)
    result = (lon_g[valid].ravel(), lat_g[valid].ravel(), mss_v[valid].ravel())
    _MSS_CACHE["data"] = result
    return result


def interpolate_mss(lon_pts: np.ndarray,
                    lat_pts: np.ndarray,
                    mss_file: str) -> np.ndarray:
    """将 MSS 双线性插值到轨迹点。"""
    lons, lats, vals = load_mss(mss_file)
    return griddata((lons, lats), vals, (lon_pts, lat_pts), method="linear")


def load_snow_for_month(snow_dir: str, month: int) -> dict | None:
    """
    读取指定月份的雪深文件，返回插值所需的字典；
    文件不存在时返回 None。
    """
    fpath = os.path.join(
        snow_dir,
        f"awi-siral-l4-snow_on_seaice-monthly_warren_amsr2_clim-"
        f"{int(month):02d}-fv1p0.nc"
    )
    if not os.path.exists(fpath):
        return None

    ds         = xr.open_dataset(fpath)
    lon_s      = ds["lon"].values
    lat_s      = ds["lat"].values
    lon_g, lat_g = np.meshgrid(lon_s, lat_s)
    lon_180    = ((lon_g + 180) % 360) - 180
    snow       = ds["merged_snow_depth"].values
    w_name     = "w99_weight"
    weight     = ds[w_name].values if w_name in ds.data_vars else np.ones_like(snow)
    valid      = ~np.isnan(snow)

    return dict(
        lon   = lon_180[valid].ravel(),
        lat   = lat_g[valid].ravel(),
        snow  = snow[valid].ravel(),
        weight= weight[valid].ravel(),
    )


def interpolate_snow(lon_pts, lat_pts, snow_dict: dict):
    """雪深 + 权重插值（linear + nearest 补 NaN 边界）。"""
    pts   = (snow_dict["lon"], snow_dict["lat"])
    lon_p = ((lon_pts + 180) % 360) - 180

    sd = griddata(pts, snow_dict["snow"],   (lon_p, lat_pts), method="linear")
    sw = griddata(pts, snow_dict["weight"], (lon_p, lat_pts), method="linear")

    nan_m = np.isnan(sd)
    if nan_m.any():
        sd[nan_m] = griddata(pts, snow_dict["snow"],   (lon_p[nan_m], lat_pts[nan_m]), method="nearest")
        sw[nan_m] = griddata(pts, snow_dict["weight"], (lon_p[nan_m], lat_pts[nan_m]), method="nearest")

    return sd, sw


def load_sic_for_date(sic_dir: str | None, date: datetime) -> tuple | None:
    """
    尝试在 sic_dir 中找到对应日期的 SIC 文件（文件名含 YYYYMMDD）。
    返回 (lon_flat, lat_flat, sic_flat) 或 None。
    支持常见 NSIDC Bootstrap / CDR 命名格式。
    """
    if sic_dir is None:
        return None
    pattern = os.path.join(sic_dir, f"*{date.strftime('%Y%m%d')}*.nc")
    files   = glob.glob(pattern)
    if not files:
        return None
    try:
        ds  = xr.open_dataset(files[0])
        # 优先查找常见 SIC 变量名
        for vname in ("seaice_conc", "sic", "sea_ice_concentration",
                      "ice_conc", "goddard_bt_seaice_conc"):
            if vname in ds.data_vars:
                sic = ds[vname].values.squeeze().astype(float)
                # NSIDC 常用 0~1 或 0~100，统一到 0~1
                if np.nanmax(sic) > 1.5:
                    sic = sic / 100.0
                lon = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
                lat = ds["latitude"].values  if "latitude"  in ds.coords else ds["lat"].values
                if lon.ndim == 1:
                    lon, lat = np.meshgrid(lon, lat)
                lon[lon > 180] -= 360
                valid = ~np.isnan(sic) & (sic >= 0) & (sic <= 1)
                return lon[valid].ravel(), lat[valid].ravel(), sic[valid].ravel()
    except Exception as e:
        log.warning(f"读取 SIC 文件失败: {files[0]} — {e}")
    return None


def interpolate_sic(lon_pts, lat_pts, sic_data: tuple | None) -> np.ndarray:
    """将 SIC 插值到轨迹点；无数据时全填 NaN。"""
    out = np.full(len(lon_pts), np.nan)
    if sic_data is None:
        return out
    lons, lats, vals = sic_data
    try:
        out = griddata((lons, lats), vals, (lon_pts, lat_pts), method="nearest")
    except Exception:
        pass
    return out


def load_ice_type_for_date(ice_type_dir: str | None, date: datetime) -> tuple | None:
    """
    读取 OSI-SAF 冰型数据（或类似格式）。
    返回 (lon_flat, lat_flat, ice_type_flat: 1=FYI, 2=MYI) 或 None。
    """
    if ice_type_dir is None:
        return None
    pattern = os.path.join(ice_type_dir, f"*{date.strftime('%Y%m%d')}*.nc")
    files   = glob.glob(pattern)
    if not files:
        return None
    try:
        ds = xr.open_dataset(files[0])
        for vname in ("ice_type", "icetype", "type"):
            if vname in ds.data_vars:
                it  = ds[vname].values.squeeze().astype(float)
                lon = ds["lon"].values if "lon" in ds.coords else ds["longitude"].values
                lat = ds["lat"].values if "lat" in ds.coords else ds["latitude"].values
                if lon.ndim == 1:
                    lon, lat = np.meshgrid(lon, lat)
                lon[lon > 180] -= 360
                valid = ~np.isnan(it)
                return lon[valid].ravel(), lat[valid].ravel(), it[valid].ravel()
    except Exception as e:
        log.warning(f"读取 ice_type 文件失败: {files[0]} — {e}")
    return None


def interpolate_ice_type(lon_pts, lat_pts, it_data: tuple | None) -> np.ndarray:
    """将 ice_type 插值（nearest）到轨迹点；无数据时全填 NaN。"""
    out = np.full(len(lon_pts), np.nan)
    if it_data is None:
        return out
    lons, lats, vals = it_data
    try:
        out = griddata((lons, lats), vals, (lon_pts, lat_pts), method="nearest")
    except Exception:
        pass
    return out


# ============================================================
# 6. SLA 计算
# ============================================================
def compute_sla(df_in: pd.DataFrame,
                h_col: str = "h_surface",
                mss_col: str = "mss_interp",
                type_col: str = "surface_type") -> np.ndarray:
    """
    Tilling 2018 SLA 流程：
      1. SLA_raw = h_surface - MSS（仅 lead 点）
      2. ±3m 异常剔除
      3. 100 km 滑动平均（沿 index 近似；采样 ~380m）
      4. 线性插值到全轨迹（不外推）
      5. 超过 200km 的点设为 NaN

    返回与 df_in 等长的 SLA 数组。
    """
    sla = np.full(len(df_in), np.nan)

    lead_mask = df_in[type_col] == "lead"
    if lead_mask.sum() < 2:
        return sla

    sla_raw = df_in[h_col] - df_in[mss_col]
    valid   = lead_mask & (sla_raw >= -3.0) & (sla_raw <= 3.0)
    if valid.sum() < 2:
        return sla

    idx_valid = df_in.index[valid].to_numpy()
    val_valid = sla_raw[valid].to_numpy()

    win100 = max(int(100000 / 380), 3)
    series = pd.Series(val_valid, index=idx_valid)
    smooth = series.rolling(window=win100, center=True, min_periods=1).mean()

    fn = interp1d(smooth.index, smooth.values,
                  kind="linear", bounds_error=False, fill_value=np.nan)
    sla_interp = fn(df_in.index)

    # 200 km 距离掩膜（使用 cKDTree + Haversine 近似）
    R = 6371000.0
    all_rad  = np.radians(df_in[["lat", "lon"]].to_numpy())
    lead_rad = np.radians(df_in.loc[lead_mask, ["lat", "lon"]].to_numpy())
    if len(lead_rad) > 0:
        tree = cKDTree(lead_rad)
        dist_rad, _ = tree.query(all_rad, k=1)
        too_far = (dist_rad * R) > 200_000
        sla_interp[too_far] = np.nan

    return sla_interp


# ============================================================
# 7. 主处理函数（单文件）
# ============================================================
def process_one_file(L1_path: str, cfg: dict) -> pd.DataFrame | None:
    """
    处理单个 CryoSat-2 L1b NC 文件，返回结果 DataFrame。
    失败时返回 None。
    """
    fname = os.path.basename(L1_path)
    log.info(f"  开始处理: {fname}")

    # ── 参数初始化 ────────────────────────────────────────────
    try:
        P = get_processing_params(L1_path)
    except ValueError as e:
        log.warning(f"  跳过（无法识别模式）: {e}")
        return None

    mode     = P["mode"]
    c        = P["c"]
    bin_size = P["bin_size"]
    bn       = P["bn"]

    # ── 读取 L1b ──────────────────────────────────────────────
    try:
        ds = xr.open_dataset(L1_path)
    except Exception as e:
        log.error(f"  无法读取文件: {e}")
        return None

    # 基础变量
    time_20     = ds["time_20_ku"].values
    lat_20      = ds["lat_20_ku"].values
    lon_20      = ds["lon_20_ku"].values
    wf_raw      = ds["pwr_waveform_20_ku"].values
    noise_pw    = ds["noise_power_20_ku"].values
    noise_pw    = np.where(noise_pw == -9999.99, np.nan, noise_pw)
    ssd_20      = ds["stack_std_20_ku"].values
    pp_native   = ds["stack_peakiness_20_ku"].values       # 原始 PP（保留字段）
    kurtosis    = ds["stack_kurtosis_20_ku"].values
    window_del  = ds["window_del_20_ku"].values
    alt_20      = ds["alt_20_ku"].values

    # sigma0（后向散射）—— 部分文件可能没有，安全读取
    sigma0 = np.full(len(lat_20), np.nan)
    for vname in ("sig0_20_ku", "sigma0_20_ku", "backscatter_20_ku"):
        if vname in ds.data_vars:
            sigma0 = ds[vname].values.astype(float)
            break

    # 验证 bins
    if wf_raw.shape[1] != P["raw_bins"]:
        log.warning(f"  bins 不匹配: file={wf_raw.shape[1]}, expected={P['raw_bins']}，跳过")
        ds.close()
        return None

    # ── 时间转换 ──────────────────────────────────────────────
    TAI_EPOCH   = datetime(2000, 1, 1, 0, 0, 0)
    tai_sec     = (time_20 - np.datetime64("2000-01-01T00:00:00")) / np.timedelta64(1, "s")
    utc_time    = np.array([TAI_EPOCH + timedelta(seconds=float(t)) for t in tai_sec])

    # 文件代表的日期（取轨道中间时刻）
    mid_idx  = len(utc_time) // 2
    file_date = utc_time[mid_idx]

    # ── surf_type 扩展到 20Hz ─────────────────────────────────
    surf_type_1hz  = ds["surf_type_01"].values
    ind_first      = ds["ind_first_meas_20hz_01"].values.astype(int)
    N20            = len(lat_20)
    surf_type_20hz = np.full(N20, np.nan)
    for i in range(len(ind_first)):
        s = ind_first[i]
        e = min(s + 20, N20)
        surf_type_20hz[s:e] = surf_type_1hz[i]

    # ── 地球物理校正（扩展到 20Hz）───────────────────────────
    CORR_VARS = [
        "mod_dry_tropo_cor_01", "mod_wet_tropo_cor_01", "inv_bar_cor_01",
        "iono_cor_01",
        "ocean_tide_01", "ocean_tide_eq_01", "load_tide_01",
        "solid_earth_tide_01", "pole_tide_01",
    ]
    corr_20hz = {v: np.full(N20, np.nan) for v in CORR_VARS}
    for v in CORR_VARS:
        if v not in ds.data_vars:
            continue
        arr = ds[v].values
        for i in range(len(ind_first)):
            s = ind_first[i]
            e = min(s + 20, N20)
            corr_20hz[v][s:e] = arr[i]
    corrections_sum = np.nansum(
        np.stack([corr_20hz[v] for v in CORR_VARS], axis=1), axis=1
    )
    # 将全 NaN 的行（所有校正项都缺失）设回 NaN
    all_nan_rows = np.all(np.isnan(
        np.stack([corr_20hz[v] for v in CORR_VARS], axis=1)), axis=1)
    corrections_sum[all_nan_rows] = np.nan

    ds.close()

    # ── 纬度过滤（全北极，MIN_LAT 以上）────────────────────────
    lat_mask = lat_20 >= cfg["MIN_LAT"]
    if lat_mask.sum() == 0:
        log.info(f"  无北极测量点，跳过")
        return None

    def mask(arr, m=lat_mask):
        return arr[m]

    lat      = mask(lat_20)
    lon      = mask(lon_20)
    utc      = mask(utc_time)
    wf       = mask(wf_raw)
    noise    = mask(noise_pw)
    ssd      = mask(ssd_20)
    pp_nat   = mask(pp_native)
    kurt     = mask(kurtosis)
    win_del  = mask(window_del)
    alt      = mask(alt_20)
    surf     = mask(surf_type_20hz)
    sig0     = mask(sigma0)
    corr_sum = mask(corrections_sum)
    for v in CORR_VARS:
        corr_20hz[v] = mask(corr_20hz[v])

    # window_del 转秒
    if np.issubdtype(win_del.dtype, np.timedelta64):
        win_del_sec = win_del / np.timedelta64(1, "s")
    else:
        win_del_sec = win_del.astype(float)  # 已经是秒

    # ── 噪声去除 ──────────────────────────────────────────────
    noise_nan = np.nan_to_num(noise, nan=0.0)
    wf_clean  = wf - noise_nan[:, np.newaxis]

    # ── 波形分类 ──────────────────────────────────────────────
    types, pps, offsets, cropped = classify_waveforms(wf_clean, ssd, P)

    # ── 重跟踪 ────────────────────────────────────────────────
    ocean_mask   = (surf == 0) | (surf == 1)
    lead_mask    = (types == "lead") & ocean_mask
    ice_mask_arr = (types == "ice")  & ocean_mask

    range_window_del = win_del_sec * c / 2.0   # 标称距离
    range_corr       = np.full(len(lat), np.nan)

    wf_lead = wf_clean[lead_mask]
    if wf_lead.shape[0] > 0:
        rc_lead = retrack_specular(wf_lead, P)
        range_corr[lead_mask] = rc_lead

    wf_ice = wf_clean[ice_mask_arr]
    if wf_ice.shape[0] > 0:
        rc_ice = retrack_diffuse(wf_ice, P)
        # 漫反射 vs 镜面固定偏差校正 -16.26 cm
        range_corr[ice_mask_arr] = rc_ice - 0.1626

    range_final = np.full(len(lat), np.nan)
    valid_corr  = ~np.isnan(range_corr)
    range_final[valid_corr] = range_window_del[valid_corr] + range_corr[valid_corr]

    # ── 地表高程 ──────────────────────────────────────────────
    h_surface = alt - range_final - corr_sum

    # ── 构建中间 DataFrame ────────────────────────────────────
    df = pd.DataFrame({
        "lat"              : lat,
        "lon"              : lon,
        "time"             : utc,
        "alt"              : alt,
        "surf_type"        : surf,
        "surface_type"     : types,            # lead / ice / unknown
        "tilling_pp"       : pps,              # 计算的 PP
        "pp_native"        : pp_nat,           # L1b 原始 pp
        "stack_std"        : ssd,              # SSD（分类用）
        "stack_kurtosis"   : kurt,
        "sigma0"           : sig0,
        "noise_power"      : noise,
        "window_del_sec"   : win_del_sec,
        "range_window_del" : range_window_del,
        "range_correction" : range_corr,
        "range_final"      : range_final,
        "h_surface"        : h_surface,
        "corrections_sum"  : corr_sum,
    })
    # 保留各校正分量（供后续建模使用）
    for v in CORR_VARS:
        df[v] = corr_20hz[v]

    # ── MSS 插值 ──────────────────────────────────────────────
    try:
        mss = interpolate_mss(lon, lat, cfg["MSS_FILE"])
    except Exception as e:
        log.warning(f"  MSS 插值失败: {e}")
        mss = np.full(len(lat), np.nan)
    df["mss_interp"] = mss

    # ── SLA ───────────────────────────────────────────────────
    df = df.reset_index(drop=True)
    sla = compute_sla(df, h_col="h_surface", mss_col="mss_interp",
                      type_col="surface_type")
    df["sla"] = sla

    # ── 雷达干舷 ──────────────────────────────────────────────
    df["radar_freeboard"] = np.nan
    ice_loc = df["surface_type"] == "ice"
    df.loc[ice_loc, "radar_freeboard"] = (
        df.loc[ice_loc, "h_surface"] -
        (df.loc[ice_loc, "mss_interp"] + df.loc[ice_loc, "sla"])
    )
    # 过滤范围 [-0.25, 2.25] m
    fb_valid = (df["radar_freeboard"] >= -0.25) & (df["radar_freeboard"] <= 2.25)
    df.loc[~fb_valid, "radar_freeboard"] = np.nan

    # ── 雪深 ──────────────────────────────────────────────────
    month      = file_date.month
    snow_dict  = load_snow_for_month(cfg["SNOW_DIR"], month) if cfg["SNOW_DIR"] else None
    if snow_dict:
        try:
            sd_raw, sd_wt = interpolate_snow(lon, lat, snow_dict)
        except Exception as e:
            log.warning(f"  雪深插值失败: {e}")
            sd_raw = np.full(len(lat), np.nan)
            sd_wt  = np.full(len(lat), np.nan)
    else:
        sd_raw = np.full(len(lat), np.nan)
        sd_wt  = np.full(len(lat), np.nan)

    df["snow_depth_raw"] = sd_raw
    df["w99_weight"]     = sd_wt

    # ── SIC ───────────────────────────────────────────────────
    sic_data = load_sic_for_date(cfg["SIC_DIR"], file_date)
    df["sic"] = interpolate_sic(lon, lat, sic_data)

    # ── 冰型 (FYI=1 / MYI=2) ─────────────────────────────────
    it_data = load_ice_type_for_date(cfg["ICE_TYPE_DIR"], file_date)
    df["ice_type_raw"] = interpolate_ice_type(lon, lat, it_data)
    # 转换为字符串标签，方便后续使用
    df["ice_type"] = np.where(
        df["ice_type_raw"] == 1, "FYI",
        np.where(df["ice_type_raw"] == 2, "MYI", np.nan)
    )

    # ── 混合密度 & 雪深修正 ───────────────────────────────────
    # f_myi: 若有 ice_type 则用，否则假设全 FYI
    f_myi = np.where(df["ice_type_raw"] == 2, 1.0, 0.0)
    df["f_myi"] = f_myi

    rho_w = cfg["RHO_W"]
    rho_s = cfg["RHO_S"]
    rho_i = f_myi * cfg["RHO_I_MYI"] + (1 - f_myi) * cfg["RHO_I_FYI"]
    df["density_ice_used"] = rho_i
    df["density_snow_used"] = rho_s
    df["density_water_used"] = rho_w

    # 雪深修正（一年冰 Warren×0.5）
    c_fyi       = 0.5
    c_factor    = (1 - f_myi) * c_fyi * np.nan_to_num(sd_wt, nan=1.0)
    hs          = sd_raw * (1 - c_factor)
    df["snow_depth"] = hs

    # ── 校正干舷 ──────────────────────────────────────────────
    df["freeboard_corrected"] = df["radar_freeboard"] + 0.25 * df["snow_depth"]
    fc_valid = (df["freeboard_corrected"] >= -0.3) & (df["freeboard_corrected"] <= 3.0)
    df.loc[~fc_valid, "freeboard_corrected"] = np.nan

    # ── 海冰厚度（粗版）─────────────────────────────────────
    df["sit_raw"] = (
        (df["freeboard_corrected"] * rho_w + df["snow_depth"] * rho_s) /
        (rho_w - rho_i)
    )
    # 合理范围过滤
    sit_valid = (df["sit_raw"] >= 0.0) & (df["sit_raw"] <= 10.0)
    df.loc[~sit_valid, "sit_raw"] = np.nan

    # ── 元数据字段 ────────────────────────────────────────────
    df["track_id"]   = Path(L1_path).stem   # 文件名（不含扩展名）作为轨道 ID
    df["mode"]       = mode
    df["file_date"]  = file_date.strftime("%Y-%m-%d")

    # ── 字段排序 ──────────────────────────────────────────────
    ORDERED_COLS = [
        # 位置 & 时间
        "lat", "lon", "time", "file_date", "track_id", "mode",
        # 地表分类
        "surf_type", "surface_type", "ice_type", "f_myi",
        # 原始观测
        "alt", "sigma0", "noise_power", "window_del_sec",
        "stack_std", "stack_kurtosis", "pp_native", "tilling_pp",
        # 重跟踪
        "range_window_del", "range_correction", "range_final",
        # 高程
        "h_surface", "mss_interp", "sla", "corrections_sum",
        *CORR_VARS,
        # Freeboard
        "radar_freeboard", "freeboard_corrected",
        # 环境变量
        "sic", "snow_depth_raw", "w99_weight", "snow_depth",
        "ice_type_raw",
        # 密度假设
        "density_ice_used", "density_snow_used", "density_water_used",
        # 厚度
        "sit_raw",
    ]
    # 保留所有列，缺失的列保持原样
    existing = [c for c in ORDERED_COLS if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    df = df[existing + remaining]

    log.info(f"  完成: {fname} → {len(df)} 行, "
             f"lead={int((df['surface_type']=='lead').sum())}, "
             f"ice={int((df['surface_type']=='ice').sum())}")
    return df


# ============================================================
# 8. 批处理入口
# ============================================================
def batch_process(cfg: dict):
    """
    扫描 L1_DIR 中所有 .nc 文件，逐文件处理并保存 parquet。
    """
    out_dir = Path(cfg["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(glob.glob(os.path.join(cfg["L1_DIR"], "*.nc")))
    if not nc_files:
        log.error(f"在 {cfg['L1_DIR']} 中未找到任何 .nc 文件！")
        return

    log.info(f"共找到 {len(nc_files)} 个 NC 文件，开始批处理 ...")
    log.info(f"输出目录: {out_dir}")
    log.info(f"并行进程数: {cfg['N_WORKERS']}")

    n_ok   = 0
    n_fail = 0

    def _process_and_save(fpath):
        """工作函数：处理 + 保存，返回 (成功, 文件名)。"""
        try:
            df = process_one_file(fpath, cfg)
            if df is None or len(df) == 0:
                return False, fpath
            stem     = Path(fpath).stem
            out_path = out_dir / f"{stem}.parquet"
            df.to_parquet(out_path, index=False, engine="pyarrow")
            del df
            gc.collect()
            return True, fpath
        except Exception:
            log.error(f"处理失败: {fpath}\n{traceback.format_exc()}")
            return False, fpath

    if cfg["N_WORKERS"] == 1:
        # 串行模式（方便调试）
        for fpath in nc_files:
            ok, _ = _process_and_save(fpath)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
    else:
        # 并行模式
        with ProcessPoolExecutor(max_workers=cfg["N_WORKERS"]) as ex:
            futures = {ex.submit(_process_and_save, f): f for f in nc_files}
            for fut in as_completed(futures):
                ok, fpath = fut.result()
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1

    log.info("=" * 60)
    log.info(f"批处理完成：成功 {n_ok} / 失败 {n_fail} / 总计 {len(nc_files)}")
    log.info(f"Parquet 文件已保存至: {out_dir}")


# ============================================================
# 9. 入口
# ============================================================
if __name__ == "__main__":
    batch_process(CONFIG)
