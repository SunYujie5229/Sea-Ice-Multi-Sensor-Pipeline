"""
process_cs2_batch.py
====================
CryoSat-2 L1b 批处理脚本（SAR / SARIn 通用）
基于 Tilling et al. (2018) 算法

目录结构约定
  L1 : L1_DIR  / YYYY / MM / CS_OFFL_SIR_SIN_1B_*.nc
  L2 : L2_BASE_DIR / YYYY / MM / CS_OFFL_SIR_SIN_2__*.nc
  输出: OUTPUT_DIR / YYYY / MM / <stem>.parquet

用法：
  python process_cs2_batch.py
修改顶部 CONFIG 区块即可，其余代码无需改动。

修复记录（相对于上一版本）
  #05 扫描：改为递归扫描，自动兼容根目录/年/月各级深度
  #06 L2查找：增加跨月回退搜索 end 月目录
  #07 fill value：noise_power 及所有辅助数组统一过滤 >1e10 大值
  #08 broadcast：fill value 替换为 NaN 后再广播
  #09 L2对齐：长度不一致时记录 WARNING，不再静默
  #10 波形负值：wf_clean clip 到 0，避免负功率进入分类
  #11 并行：_worker 提到模块顶层，Windows ProcessPoolExecutor 可 pickle
  #12 类型注解：Optional/Dict/List 改为 typing，兼容 Python 3.8
  #13 MSS缓存：改为以文件路径为 key，多次调用不冲突
  #14 时间列：转为 pd.Timestamp（datetime64[ns]），parquet 写入友好
      parquet 引擎：pyarrow 失败自动 fallback 到 fastparquet
"""

# ===========================================================
CONFIG = {
    # ── L1 根目录（自动递归扫描 YYYY/MM 子目录）─────────────
    # 可以是根目录 E:\Arctic\2024，也可以直接指向月份目录
    # E:\Arctic\2024\09，脚本会自动判断
    "L1_DIR"      : r"E:\Arctic\2024\10",

    # ── L2 根目录（同样按 YYYY/MM 存放）─────────────────────
    # 设为 None 则 L2 字段全部填 NaN
    "L2_BASE_DIR" : r"E:\Arctic_L2",

    # ── 辅助数据路径 ──────────────────────────────────────────
    "MSS_FILE"    : r"E:\Project_2024\CryoSat-2 L1\DTU21MSS_1min_WGS84.nc",
    "SNOW_DIR"    : r"D:\S1_CS2_data\awi_snow_merged",
    # SIC / 冰型 若暂无数据设为 None，对应字段自动填 NaN
    "SIC_DIR"     : None,
    "ICE_TYPE_DIR": None,

    # ── 输出根目录（自动创建 YYYY/MM 子目录）────────────────
    "OUTPUT_DIR"  : r"E:\Arctic\parquet_out",

    # ── 并行进程数（Windows 建议从 1 开始调试）───────────────
    "N_WORKERS"   : 1,

    # ── 纬度南界（低于此纬度的点直接丢弃）───────────────────
    "MIN_LAT"     : 60.0,

    # ── 物理常量 ──────────────────────────────────────────────
    "RHO_W"       : 1023.9,
    "RHO_S"       : 324.0,
    "RHO_I_FYI"   : 916.7,
    "RHO_I_MYI"   : 882.0,
}
# ===========================================================


# ── 标准库 ──────────────────────────────────────────────────
import gc
import glob
import logging
import os
import re as _re
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 第三方库 ─────────────────────────────────────────────────
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal
import xarray as xr
from scipy.interpolate import griddata, interp1d
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ── 日志（同时写文件）────────────────────────────────────────
_log_handlers: List[logging.Handler] = [logging.StreamHandler()]
try:
    _log_handlers.append(logging.FileHandler("process_cs2.log", encoding="utf-8"))
except Exception:
    pass
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=_log_handlers,
)
log = logging.getLogger(__name__)

# CryoSat-2 通用 fill value 阈值（超过此绝对值视为无效）
_FILL_THRESHOLD = 1e10


# ===========================================================
# 1. 参数工厂
# ===========================================================
def get_processing_params(file_path: str) -> dict:
    """根据文件名自动识别 SAR / SARIn 模式，返回全部算法参数。"""
    filename = os.path.basename(file_path)
    if "SIR_SIN" in filename:
        mode = "SARIn"
    elif "SIR_SAR" in filename:
        mode = "SAR"
    else:
        raise ValueError(f"无法从文件名识别模式: {filename}")

    params = dict(
        mode                 = mode,
        c                    = 299792458.0,
        bin_size             = 0.2342,
        cropped_bins         = 128,
        crop_left            = 50,
        crop_right           = 77,
        retrack_threshold    = 0.70,
        first_peak_min_ratio = 0.20,
        pp_lead_threshold    = 18,
        pp_ice_threshold     = 9,
    )
    if mode == "SARIn":
        params.update(raw_bins=1024, bn=512,  ssd_threshold=4.62,
                      peak_threshold=0.45, needs_smoothing=True)
    else:
        params.update(raw_bins=256,  bn=128,  ssd_threshold=6.29,
                      peak_threshold=0.15, needs_smoothing=False)
    return params


# ===========================================================
# 2. 文件扫描（递归，兼容 root/YYYY/MM 各级深度）FIX #05
# ===========================================================
def scan_l1_files(l1_dir: str) -> List[str]:
    """
    递归扫描 l1_dir 下所有 CryoSat-2 L1b nc 文件。
    兼容三种目录深度：
      · root/                  → 顶层直接含 nc
      · root/YYYY/             → 年份子目录
      · root/YYYY/MM/          → 月份子目录（标准格式）
    只保留文件名同时满足：含 _1B_ 且含 SIR_SIN 或 SIR_SAR。
    """
    root = Path(l1_dir)
    if not root.exists():
        log.error(f"L1_DIR 不存在: {l1_dir}")
        return []
    all_nc = sorted(root.rglob("*.nc"))
    result = [
        str(f) for f in all_nc
        if _re.search(r"_1[Bb]_", f.name)
        and ("SIR_SIN" in f.name or "SIR_SAR" in f.name)
    ]
    log.info(f"扫描到 {len(result)} 个 L1b 文件（根目录: {l1_dir}）")
    return result


# ===========================================================
# 3. 波形工具
# ===========================================================
def crop_waveform(waveform: np.ndarray,
                  crop_left: int = 50,
                  crop_right: int = 77) -> Tuple[np.ndarray, int]:
    """将任意长度波形裁剪为 128 bins，返回 (cropped_wf, abs_offset)。"""
    b_max       = int(np.argmax(waveform))
    start_index = b_max - crop_left
    end_index   = b_max + crop_right + 1
    pad_left    = max(0, -start_index)
    pad_right   = max(0, end_index - len(waveform))
    cropped = waveform[max(0, start_index):min(len(waveform), end_index)]
    if pad_left > 0 or pad_right > 0:
        cropped = np.pad(cropped, (pad_left, pad_right),
                         mode="constant", constant_values=0)
    return cropped, b_max - crop_left


def compute_pp(cwf: np.ndarray) -> float:
    noise_floor = np.mean(cwf[10:20])
    valid = cwf[cwf > noise_floor]
    if len(valid) == 0:
        return 0.0
    p_mean = float(np.mean(valid))
    return float(np.max(cwf)) / p_mean if p_mean > 0 else 0.0


# ===========================================================
# 4. 波形分类
# ===========================================================
def classify_waveforms(wf_arr: np.ndarray,
                       ssd_arr: np.ndarray,
                       params: dict) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
    n = len(wf_arr)
    cl, cr = params["crop_left"], params["crop_right"]
    ssd_thr = params["ssd_threshold"]
    pp_lead = params["pp_lead_threshold"]
    pp_ice  = params["pp_ice_threshold"]

    types   = np.full(n, "unknown", dtype=object)
    pps     = np.full(n, np.nan)
    offsets = np.full(n, np.nan)
    cropped = []

    for i in range(n):
        cwf, off = crop_waveform(wf_arr[i], cl, cr)
        cropped.append(cwf)
        offsets[i] = off
        pp = compute_pp(cwf)
        pps[i] = pp
        ssd = float(ssd_arr[i])
        if pp > pp_lead and ssd < ssd_thr:
            types[i] = "lead"
        elif pp < pp_ice and ssd > ssd_thr:
            types[i] = "ice"

    return types, pps, offsets, np.array(cropped)


# ===========================================================
# 5. 重跟踪
# ===========================================================
def retrack_diffuse(wf_arr: np.ndarray, params: dict) -> np.ndarray:
    n = len(wf_arr)
    rc = np.full(n, np.nan)
    b_n = params["bn"]; bs = params["bin_size"]
    cl = params["crop_left"]; cr = params["crop_right"]
    pmr = params["first_peak_min_ratio"]
    rthr = params["retrack_threshold"]
    smooth = params["needs_smoothing"]

    for i in range(n):
        cwf, off = crop_waveform(wf_arr[i], cl, cr)
        wfp = scipy.ndimage.uniform_filter1d(cwf, 3) if smooth else cwf.copy()
        mx = float(np.max(wfp))
        if mx <= 0:
            continue
        peaks, _ = scipy.signal.find_peaks(wfp)
        vpk = [p for p in peaks if wfp[p] >= pmr * mx]
        if not vpk:
            continue
        fp_idx = vpk[0]
        thr70 = rthr * float(wfp[fp_idx])
        b0 = np.nan
        for j in range(fp_idx - 1, -1, -1):
            if wfp[j] <= thr70 <= wfp[j + 1]:
                b0 = j + (thr70 - wfp[j]) / (wfp[j + 1] - wfp[j])
                break
        if not np.isnan(b0):
            rc[i] = (off + b0 - b_n) * bs
    return rc


def _giles_echo(t, a, t0, k, sigma):
    sigma = max(sigma, 1e-6); k = max(k, 1e-6)
    tb = k * sigma**2; sqkt = np.sqrt(k * tb)
    a2 = (5*k*sigma - 4*sqkt) / (2*sigma*tb*sqkt)
    a3 = (2*sqkt - 3*k*sigma) / (2*sigma*tb**2*sqkt)
    f = np.zeros_like(t, dtype=float)
    m1 = t < t0; f[m1] = (t[m1] - t0) / sigma
    m2 = (t >= t0) & (t < t0 + tb); dt2 = t[m2] - t0
    f[m2] = a3*dt2**3 + a2*dt2**2 + dt2/sigma
    m3 = t >= t0 + tb
    f[m3] = np.sqrt(k * np.maximum(t[m3] - t0, 0))
    return a * np.exp(-(f**2))


def retrack_specular(wf_arr: np.ndarray, params: dict) -> np.ndarray:
    n = len(wf_arr)
    rc = np.full(n, np.nan)
    b_n = params["bn"]; bs = params["bin_size"]
    cl = params["crop_left"]; cr = params["crop_right"]
    t128 = np.arange(128, dtype=float)

    for i in range(n):
        cwf, off = crop_waveform(wf_arr[i], cl, cr)
        p0 = [float(np.max(cwf)), float(np.argmax(cwf)), 0.5, 1.0]
        try:
            popt, _ = curve_fit(_giles_echo, t128, cwf, p0=p0,
                                method="lm", maxfev=3000)
            t0l = float(popt[1])
            if 0 <= t0l <= 127:
                rc[i] = (off + t0l - b_n) * bs
        except RuntimeError:
            pass
    return rc


# ===========================================================
# 6. 辅助数据加载
# ===========================================================
_MSS_CACHE: Dict[str, Tuple] = {}   # FIX #13: 按路径缓存


def load_mss(mss_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    global _MSS_CACHE
    if mss_file in _MSS_CACHE:
        return _MSS_CACHE[mss_file]
    ds = xr.open_dataset(mss_file)
    lon = ds["lon"].values.copy(); lon[lon > 180] -= 360
    lat = ds["lat"].values
    lg, latg = np.meshgrid(lon, lat)
    mss = ds["mean_sea_surf_sol2"].values
    v = ~np.isnan(mss)
    r = (lg[v].ravel(), latg[v].ravel(), mss[v].ravel())
    _MSS_CACHE[mss_file] = r
    return r


def interpolate_mss(lon_pts: np.ndarray, lat_pts: np.ndarray,
                    mss_file: str) -> np.ndarray:
    lons, lats, vals = load_mss(mss_file)
    return griddata((lons, lats), vals, (lon_pts, lat_pts), method="linear")


def load_snow_for_month(snow_dir: str, month: int) -> Optional[dict]:
    fp = os.path.join(
        snow_dir,
        f"awi-siral-l4-snow_on_seaice-monthly_warren_amsr2_clim-"
        f"{int(month):02d}-fv1p0.nc",
    )
    if not os.path.exists(fp):
        return None
    ds = xr.open_dataset(fp)
    lon_s = ds["lon"].values; lat_s = ds["lat"].values
    lg, latg = np.meshgrid(lon_s, lat_s)
    lon180 = ((lg + 180) % 360) - 180
    snow = ds["merged_snow_depth"].values
    wname = "w99_weight"
    wt = ds[wname].values if wname in ds.data_vars else np.ones_like(snow)
    v = ~np.isnan(snow)
    return dict(lon=lon180[v].ravel(), lat=latg[v].ravel(),
                snow=snow[v].ravel(), weight=wt[v].ravel())


def interpolate_snow(lon_pts: np.ndarray, lat_pts: np.ndarray,
                     sd: dict) -> Tuple[np.ndarray, np.ndarray]:
    pts = (sd["lon"], sd["lat"]); lp = ((lon_pts+180)%360)-180
    s = griddata(pts, sd["snow"],   (lp, lat_pts), method="linear")
    w = griddata(pts, sd["weight"], (lp, lat_pts), method="linear")
    nm = np.isnan(s)
    if nm.any():
        s[nm] = griddata(pts, sd["snow"],   (lp[nm], lat_pts[nm]), method="nearest")
        w[nm] = griddata(pts, sd["weight"], (lp[nm], lat_pts[nm]), method="nearest")
    return s, w


def _find_nc_by_date(directory: Optional[str], date: datetime) -> Optional[str]:
    if not directory:
        return None
    files = glob.glob(os.path.join(directory, f"*{date.strftime('%Y%m%d')}*.nc"))
    return files[0] if files else None


def load_sic_for_date(sic_dir: Optional[str],
                      date: datetime) -> Optional[Tuple]:
    fp = _find_nc_by_date(sic_dir, date)
    if not fp:
        return None
    try:
        ds = xr.open_dataset(fp)
        for vn in ("seaice_conc","sic","sea_ice_concentration",
                   "ice_conc","goddard_bt_seaice_conc"):
            if vn in ds.data_vars:
                sic = ds[vn].values.squeeze().astype(float)
                if np.nanmax(sic) > 1.5:
                    sic /= 100.0
                lon = (ds["longitude"].values if "longitude" in ds.coords
                       else ds["lon"].values)
                lat = (ds["latitude"].values  if "latitude"  in ds.coords
                       else ds["lat"].values)
                if lon.ndim == 1:
                    lon, lat = np.meshgrid(lon, lat)
                lon = lon.copy(); lon[lon > 180] -= 360
                v = ~np.isnan(sic) & (sic>=0) & (sic<=1)
                return lon[v].ravel(), lat[v].ravel(), sic[v].ravel()
    except Exception as e:
        log.warning(f"  读取 SIC 失败: {fp} — {e}")
    return None


def interpolate_sic(lon_pts: np.ndarray, lat_pts: np.ndarray,
                    data: Optional[Tuple]) -> np.ndarray:
    out = np.full(len(lon_pts), np.nan)
    if data:
        try:
            out = griddata((data[0],data[1]), data[2],
                           (lon_pts,lat_pts), method="nearest")
        except Exception:
            pass
    return out


def load_ice_type_for_date(itdir: Optional[str],
                           date: datetime) -> Optional[Tuple]:
    fp = _find_nc_by_date(itdir, date)
    if not fp:
        return None
    try:
        ds = xr.open_dataset(fp)
        for vn in ("ice_type","icetype","type"):
            if vn in ds.data_vars:
                it  = ds[vn].values.squeeze().astype(float)
                lon = ds["lon"].values if "lon" in ds.coords else ds["longitude"].values
                lat = ds["lat"].values if "lat" in ds.coords else ds["latitude"].values
                if lon.ndim == 1:
                    lon, lat = np.meshgrid(lon, lat)
                lon = lon.copy(); lon[lon > 180] -= 360
                v = ~np.isnan(it)
                return lon[v].ravel(), lat[v].ravel(), it[v].ravel()
    except Exception as e:
        log.warning(f"  读取 ice_type 失败: {fp} — {e}")
    return None


def interpolate_ice_type(lon_pts: np.ndarray, lat_pts: np.ndarray,
                         data: Optional[Tuple]) -> np.ndarray:
    out = np.full(len(lon_pts), np.nan)
    if data:
        try:
            out = griddata((data[0],data[1]), data[2],
                           (lon_pts,lat_pts), method="nearest")
        except Exception:
            pass
    return out


# ===========================================================
# 7. L2 对比变量读取
# ===========================================================
_L2_TARGET_VARS: List[str] = [
    "sea_ice_concentration_01",
    "snow_depth_cor_20_ku",
    "radar_freeboard_20_ku",
    "surf_type_20_ku",
]
_L2_BROADCAST_1HZ = {"sea_ice_concentration_01"}


def _parse_l1_stem(stem: str) -> Optional[dict]:
    m = _re.match(
        r"^(CS)_(OFFL|NRT|LTA|REP)_(SIR)_(SAR|SIN|LRM)_"
        r"(1B|1b)_(\d{8}T\d{6})_(\d{8}T\d{6})_(\w+)$",
        stem,
    )
    if not m:
        return None
    return dict(mission=m.group(1), timeliness=m.group(2),
                instrument=m.group(3), product=m.group(4),
                start=m.group(6), end=m.group(7), version=m.group(8))


def _build_l2_stem(info: dict) -> str:
    return (f"{info['mission']}_{info['timeliness']}_{info['instrument']}_"
            f"{info['product']}_2__{info['start']}_{info['end']}_{info['version']}")


def find_l2_file(l1_path: str,
                 l2_base_dir: Optional[str]) -> Optional[str]:
    """
    在 L2_BASE_DIR/YYYY/MM/ 下查找对应 L2 文件。FIX #06：增加跨月回退。
    优先级: 精确匹配 → start月模糊 → end月精确 → end月模糊
    """
    if not l2_base_dir:
        return None
    info = _parse_l1_stem(Path(l1_path).stem)
    if not info:
        return None
    try:
        start_dt = datetime.strptime(info["start"], "%Y%m%dT%H%M%S")
        end_dt   = datetime.strptime(info["end"],   "%Y%m%dT%H%M%S")
    except ValueError:
        return None

    base = Path(l2_base_dir)
    l2_stem = _build_l2_stem(info)

    def _search(l2_dir: Path) -> Optional[str]:
        if not l2_dir.exists():
            return None
        exact = l2_dir / f"{l2_stem}.nc"
        if exact.exists():
            return str(exact)
        for f in l2_dir.glob("*.nc"):
            if info["start"] in f.stem and info["product"] in f.stem:
                return str(f)
        return None

    # start 月
    r = _search(base / start_dt.strftime("%Y") / start_dt.strftime("%m"))
    if r:
        return r
    # end 月（跨月时 FIX #06）
    if (end_dt.year, end_dt.month) != (start_dt.year, start_dt.month):
        r = _search(base / end_dt.strftime("%Y") / end_dt.strftime("%m"))
    return r


def _safe_fill(arr: np.ndarray) -> np.ndarray:
    """FIX #07 #08：将超过 fill threshold 的值替换为 NaN。"""
    arr = arr.astype(float)
    arr[np.abs(arr) > _FILL_THRESHOLD] = np.nan
    return arr


def broadcast_1hz_to_20hz(arr_1hz: np.ndarray,
                           ind_first: np.ndarray,
                           n20: int) -> np.ndarray:
    """FIX #08：广播前已由调用方 _safe_fill 处理 fill value。"""
    out = np.full(n20, np.nan, dtype=float)
    for i, s in enumerate(ind_first):
        s = int(s); e = min(s + 20, n20)
        val = float(arr_1hz[i])
        if not np.isnan(val):
            out[s:e] = val
    return out


def read_l2_vars(l2_path: str,
                 lat_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """FIX #09：长度不一致时记录 WARNING，而非静默。"""
    n_l1  = len(lat_mask)
    result: Dict[str, np.ndarray] = {
        v: np.full(lat_mask.sum(), np.nan) for v in _L2_TARGET_VARS
    }
    try:
        ds = xr.open_dataset(l2_path)
    except Exception as e:
        log.warning(f"  L2 打开失败: {l2_path} — {e}")
        return result

    # L2 的 20Hz 长度
    n20_l2 = None
    for dv in ("lat_poca_20_ku","lat_20_ku","time_20_ku"):
        if dv in ds.data_vars or dv in ds.coords:
            n20_l2 = len(ds[dv]); break
    if n20_l2 is None:
        log.warning(f"  L2 无法确定 20Hz 长度: {l2_path}")
        ds.close(); return result

    if n20_l2 != n_l1:
        log.warning(f"  L2/L1 长度不一致 L2={n20_l2} L1={n_l1}: "
                    f"{os.path.basename(l2_path)}，超出部分填 NaN")

    # ind_first（broadcast 1Hz 用）
    ind_first_l2 = None
    for iname in ("ind_first_meas_20hz_01","ind_first_meas_20_ku_01"):
        if iname in ds.data_vars or iname in ds.coords:
            ind_first_l2 = ds[iname].values.astype(int); break

    n_common = min(n20_l2, n_l1)

    for vname in _L2_TARGET_VARS:
        if vname not in ds.data_vars:
            log.debug(f"  L2 缺少 {vname}（填 NaN）"); continue

        raw = _safe_fill(ds[vname].values.squeeze())

        if vname in _L2_BROADCAST_1HZ:
            if ind_first_l2 is None:
                log.warning(f"  L2 缺少 ind_first，无法 broadcast {vname}"); continue
            arr_20 = broadcast_1hz_to_20hz(raw, ind_first_l2, n20_l2)
        else:
            arr_20 = raw

        aligned = np.full(n_l1, np.nan)
        aligned[:n_common] = arr_20[:n_common]
        result[vname] = aligned[lat_mask]

    ds.close()
    return result


# ===========================================================
# 8. SLA 计算
# ===========================================================
def compute_sla(df_in: pd.DataFrame,
                h_col: str = "h_surface",
                mss_col: str = "mss_interp",
                type_col: str = "surface_type") -> np.ndarray:
    sla     = np.full(len(df_in), np.nan)
    lm      = df_in[type_col] == "lead"
    if lm.sum() < 2:
        return sla
    sla_raw = df_in[h_col] - df_in[mss_col]
    valid   = lm & (sla_raw >= -3.0) & (sla_raw <= 3.0)
    if valid.sum() < 2:
        return sla
    idx_v = df_in.index[valid].to_numpy()
    val_v = sla_raw[valid].to_numpy()
    win   = max(int(100_000/380), 3)
    smooth = pd.Series(val_v, index=idx_v).rolling(win, center=True, min_periods=1).mean()
    fn = interp1d(smooth.index, smooth.values,
                  kind="linear", bounds_error=False, fill_value=np.nan)
    sla = fn(df_in.index)
    R = 6_371_000.0
    all_r  = np.radians(df_in[["lat","lon"]].to_numpy())
    lead_r = np.radians(df_in.loc[lm, ["lat","lon"]].to_numpy())
    if len(lead_r) > 0:
        tree = cKDTree(lead_r)
        dr, _ = tree.query(all_r, k=1)
        sla[dr * R > 200_000] = np.nan
    return sla


# ===========================================================
# 9. 安全读取 L1b 字段
# ===========================================================
def _safe_read(ds: xr.Dataset, var: str,
               fill_val: float = -9999.99) -> Optional[np.ndarray]:
    """读取变量，替换 fill value 和大值为 NaN；变量不存在返回 None。"""
    if var not in ds.data_vars and var not in ds.coords:
        return None
    arr = ds[var].values.astype(float)
    arr[np.abs(arr - fill_val) < 1e-3] = np.nan
    arr[np.abs(arr) > _FILL_THRESHOLD] = np.nan     # FIX #07
    return arr


def _bcast(ds: xr.Dataset, var: str,
           ind_first: np.ndarray, n20: int) -> np.ndarray:
    arr = _safe_read(ds, var)
    if arr is None:
        return np.full(n20, np.nan)
    return broadcast_1hz_to_20hz(arr, ind_first, n20)


# ===========================================================
# 10. 主处理函数（单文件）
# ===========================================================
def process_one_file(l1_path: str, cfg: dict) -> Optional[pd.DataFrame]:
    fname = os.path.basename(l1_path)
    log.info(f"  处理: {fname}")

    try:
        P = get_processing_params(l1_path)
    except ValueError as e:
        log.warning(f"  跳过（模式未识别）: {e}"); return None

    mode = P["mode"]; c = P["c"]

    # ── 打开文件 ────────────────────────────────────────────
    try:
        ds = xr.open_dataset(l1_path)
    except Exception as e:
        log.error(f"  打开失败（跳过）: {fname} — {e}"); return None

    # ── 必须字段 ────────────────────────────────────────────
    try:
        time_20        = ds["time_20_ku"].values
        lat_20         = ds["lat_20_ku"].values
        lon_20         = ds["lon_20_ku"].values
        wf_raw         = ds["pwr_waveform_20_ku"].values
        alt_20         = ds["alt_20_ku"].values
        window_del_raw = ds["window_del_20_ku"].values
        N20 = len(lat_20)
    except KeyError as e:
        log.error(f"  缺少必须字段 {e}（跳过）: {fname}")
        ds.close(); return None

    # ── 可选字段 ────────────────────────────────────────────
    noise_pw  = _safe_read(ds, "noise_power_20_ku")  or np.zeros(N20)
    ssd_20    = _safe_read(ds, "stack_std_20_ku")    or np.full(N20, np.nan)
    pp_native = _safe_read(ds, "stack_peakiness_20_ku") or np.full(N20, np.nan)
    kurtosis  = _safe_read(ds, "stack_kurtosis_20_ku")  or np.full(N20, np.nan)
    sigma0    = np.full(N20, np.nan)
    for vn in ("sig0_20_ku","sigma0_20_ku","backscatter_20_ku"):
        tmp = _safe_read(ds, vn)
        if tmp is not None:
            sigma0 = tmp; break

    # ── bins 验证 ───────────────────────────────────────────
    if wf_raw.ndim != 2 or wf_raw.shape[1] != P["raw_bins"]:
        log.warning(f"  bins 不匹配 {wf_raw.shape}（跳过）: {fname}")
        ds.close(); return None

    # ── 时间转换（FIX #14：转为 datetime64[ns]）────────────
    TAI_EPOCH = datetime(2000, 1, 1)
    try:
        tai_sec = ((time_20 - np.datetime64("2000-01-01T00:00:00"))
                   / np.timedelta64(1, "s"))
        utc_arr = np.array(
            [TAI_EPOCH + timedelta(seconds=float(t)) for t in tai_sec],
            dtype="datetime64[ns]",
        )
    except Exception as e:
        log.error(f"  时间转换失败（跳过）: {fname} — {e}")
        ds.close(); return None

    file_date = TAI_EPOCH + timedelta(seconds=float(tai_sec[N20 // 2]))

    # ── ind_first_meas ──────────────────────────────────────
    try:
        ind_first = ds["ind_first_meas_20hz_01"].values.astype(int)
    except KeyError:
        log.warning(f"  缺少 ind_first_meas_20hz_01，surf_type/校正填 NaN")
        ind_first = None

    # ── surf_type broadcast ─────────────────────────────────
    surf_type_20hz = np.full(N20, np.nan)
    if ind_first is not None:
        arr = _safe_read(ds, "surf_type_01")
        if arr is not None:
            surf_type_20hz = broadcast_1hz_to_20hz(arr, ind_first, N20)

    # ── 地球物理校正 broadcast ──────────────────────────────
    CORR_VARS = [
        "mod_dry_tropo_cor_01","mod_wet_tropo_cor_01","inv_bar_cor_01",
        "iono_cor_01",
        "ocean_tide_01","ocean_tide_eq_01","load_tide_01",
        "solid_earth_tide_01","pole_tide_01",
    ]
    corr_20hz: Dict[str, np.ndarray] = {}
    for v in CORR_VARS:
        corr_20hz[v] = (_bcast(ds, v, ind_first, N20)
                        if ind_first is not None
                        else np.full(N20, np.nan))

    corr_stack   = np.stack([corr_20hz[v] for v in CORR_VARS], axis=1)
    all_nan      = np.all(np.isnan(corr_stack), axis=1)
    corr_sum_all = np.nansum(corr_stack, axis=1)
    corr_sum_all[all_nan] = np.nan

    ds.close()

    # ── 纬度过滤 ────────────────────────────────────────────
    lat_mask = lat_20 >= cfg["MIN_LAT"]
    if lat_mask.sum() == 0:
        log.info(f"  无北极点（MIN_LAT={cfg['MIN_LAT']}°），跳过")
        return None

    def M(a: np.ndarray) -> np.ndarray:
        return a[lat_mask]

    lat      = M(lat_20);         lon      = M(lon_20)
    utc      = M(utc_arr);        wf       = M(wf_raw)
    noise    = M(noise_pw);       ssd      = M(ssd_20)
    pp_nat   = M(pp_native);      kurt     = M(kurtosis)
    win_del  = M(window_del_raw); alt      = M(alt_20)
    surf     = M(surf_type_20hz); sig0     = M(sigma0)
    corr_sum = M(corr_sum_all)
    for v in CORR_VARS:
        corr_20hz[v] = M(corr_20hz[v])

    # ── window_del → 秒（FIX #04 单位注释）─────────────────
    # L1b window_del_20_ku 通常为 timedelta64（皮秒精度）
    # 少数版本为 float，单位为秒；中位数合理范围 ~0.003 s
    if np.issubdtype(win_del.dtype, np.timedelta64):
        win_del_sec = (win_del / np.timedelta64(1, "s")).astype(float)
    else:
        win_del_sec = win_del.astype(float)
        med = float(np.nanmedian(win_del_sec))
        if 0 < med < 1e-6:   # 疑似皮秒
            log.warning(f"  window_del 中位数 {med:.3e}，疑似 ps 单位，自动×1e-12")
            win_del_sec *= 1e-12

    # ── 噪声去除（FIX #10：clip 避免负功率）────────────────
    noise_nn = np.nan_to_num(noise, nan=0.0)
    wf_clean = np.clip(wf.astype(float) - noise_nn[:, np.newaxis], 0, None)

    # ── 分类 & 重跟踪 ───────────────────────────────────────
    types, pps, _, _ = classify_waveforms(wf_clean, ssd, P)
    ocean_msk = (surf == 0) | (surf == 1)
    lead_msk  = (types == "lead") & ocean_msk
    ice_msk   = (types == "ice")  & ocean_msk

    range_wdel = win_del_sec * c / 2.0
    range_corr = np.full(len(lat), np.nan)
    if lead_msk.sum() > 0:
        range_corr[lead_msk] = retrack_specular(wf_clean[lead_msk], P)
    if ice_msk.sum() > 0:
        range_corr[ice_msk] = retrack_diffuse(wf_clean[ice_msk], P) - 0.1626

    range_final = np.full(len(lat), np.nan)
    ok = ~np.isnan(range_corr)
    range_final[ok] = range_wdel[ok] + range_corr[ok]
    h_surface = alt - range_final - corr_sum

    # ── 构建 DataFrame ──────────────────────────────────────
    df = pd.DataFrame({
        "lat"              : lat,
        "lon"              : lon,
        "time"             : pd.to_datetime(utc),   # FIX #14
        "alt"              : alt,
        "surf_type"        : surf,
        "surface_type"     : types,
        "tilling_pp"       : pps,
        "pp_native"        : pp_nat,
        "stack_std"        : ssd,
        "stack_kurtosis"   : kurt,
        "sigma0"           : sig0,
        "noise_power"      : noise,
        "window_del_sec"   : win_del_sec,
        "range_window_del" : range_wdel,
        "range_correction" : range_corr,
        "range_final"      : range_final,
        "h_surface"        : h_surface,
        "corrections_sum"  : corr_sum,
    })
    for v in CORR_VARS:
        df[v] = corr_20hz[v]
    df = df.reset_index(drop=True)

    # ── MSS ─────────────────────────────────────────────────
    try:
        mss = interpolate_mss(lon, lat, cfg["MSS_FILE"])
    except Exception as e:
        log.warning(f"  MSS 失败: {e}"); mss = np.full(len(lat), np.nan)
    df["mss_interp"] = mss

    # ── SLA ─────────────────────────────────────────────────
    df["sla"] = compute_sla(df)

    # ── 雷达干舷 ────────────────────────────────────────────
    df["radar_freeboard"] = np.nan
    iloc = df["surface_type"] == "ice"
    df.loc[iloc, "radar_freeboard"] = (
        df.loc[iloc, "h_surface"]
        - df.loc[iloc, "mss_interp"]
        - df.loc[iloc, "sla"]
    )
    fb_ok = (df["radar_freeboard"] >= -0.25) & (df["radar_freeboard"] <= 2.25)
    df.loc[~fb_ok, "radar_freeboard"] = np.nan

    # ── 雪深 ────────────────────────────────────────────────
    snow_dict = None
    if cfg.get("SNOW_DIR"):
        try:
            snow_dict = load_snow_for_month(cfg["SNOW_DIR"], file_date.month)
        except Exception as e:
            log.warning(f"  雪深读取失败: {e}")
    sd_raw = sd_wt = np.full(len(lat), np.nan)
    if snow_dict:
        try:
            sd_raw, sd_wt = interpolate_snow(lon, lat, snow_dict)
        except Exception as e:
            log.warning(f"  雪深插值失败: {e}")
    df["snow_depth_raw"] = sd_raw
    df["w99_weight"]     = sd_wt

    # ── SIC / 冰型 ──────────────────────────────────────────
    df["sic"] = interpolate_sic(lon, lat,
                                load_sic_for_date(cfg.get("SIC_DIR"), file_date))
    it_raw = interpolate_ice_type(lon, lat,
                                  load_ice_type_for_date(cfg.get("ICE_TYPE_DIR"), file_date))
    df["ice_type_raw"] = it_raw
    df["ice_type"] = np.where(it_raw==1, "FYI", np.where(it_raw==2, "MYI", pd.NA))

    # ── 密度 & 雪深修正 ─────────────────────────────────────
    f_myi = np.where(it_raw == 2, 1.0, 0.0)
    df["f_myi"] = f_myi
    rho_w = cfg["RHO_W"]; rho_s = cfg["RHO_S"]
    rho_i = f_myi * cfg["RHO_I_MYI"] + (1 - f_myi) * cfg["RHO_I_FYI"]
    df["density_ice_used"]   = rho_i
    df["density_snow_used"]  = rho_s
    df["density_water_used"] = rho_w
    c_fac = (1-f_myi) * 0.5 * np.nan_to_num(sd_wt, nan=1.0)
    df["snow_depth"] = sd_raw * (1 - c_fac)

    # ── 校正干舷 & 厚度 ─────────────────────────────────────
    df["freeboard_corrected"] = df["radar_freeboard"] + 0.25 * df["snow_depth"]
    fc_ok = (df["freeboard_corrected"]>=-0.3) & (df["freeboard_corrected"]<=3.0)
    df.loc[~fc_ok, "freeboard_corrected"] = np.nan
    df["sit_raw"] = ((df["freeboard_corrected"]*rho_w + df["snow_depth"]*rho_s)
                     / (rho_w - rho_i))
    sit_ok = (df["sit_raw"]>=0.0) & (df["sit_raw"]<=10.0)
    df.loc[~sit_ok, "sit_raw"] = np.nan

    # ── L2 对比变量 ─────────────────────────────────────────
    l2_path = find_l2_file(l1_path, cfg.get("L2_BASE_DIR"))
    if l2_path:
        log.info(f"  匹配 L2: {os.path.basename(l2_path)}")
        for vn, arr in read_l2_vars(l2_path, lat_mask).items():
            df[f"l2_{vn}"] = arr
    else:
        log.debug("  未找到 L2 文件，L2 字段填 NaN")
        for vn in _L2_TARGET_VARS:
            df[f"l2_{vn}"] = np.nan

    # ── 元数据 & 字段排序 ───────────────────────────────────
    df["track_id"]  = Path(l1_path).stem
    df["mode"]      = mode
    df["file_date"] = file_date.strftime("%Y-%m-%d")

    ORDERED = [
        "lat","lon","time","file_date","track_id","mode",
        "surf_type","surface_type","ice_type","f_myi",
        "alt","sigma0","noise_power","window_del_sec",
        "stack_std","stack_kurtosis","pp_native","tilling_pp",
        "range_window_del","range_correction","range_final",
        "h_surface","mss_interp","sla","corrections_sum",
        *CORR_VARS,
        "radar_freeboard","freeboard_corrected",
        "sic","snow_depth_raw","w99_weight","snow_depth","ice_type_raw",
        "density_ice_used","density_snow_used","density_water_used",
        "sit_raw",
        "l2_sea_ice_concentration_01","l2_snow_depth_cor_20_ku",
        "l2_radar_freeboard_20_ku","l2_surf_type_20_ku",
    ]
    existing  = [c for c in ORDERED if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    df = df[existing + remaining]

    log.info(f"  完成: {fname} → {len(df)}行  "
             f"lead={(df['surface_type']=='lead').sum()}  "
             f"ice={(df['surface_type']=='ice').sum()}")
    return df


# ===========================================================
# 11. 顶层 worker（FIX #11：模块级函数，Windows 可 pickle）
# ===========================================================
def _worker(args: Tuple) -> Tuple[bool, str]:
    """处理单文件 + 保存 parquet，定义在顶层供 ProcessPoolExecutor 使用。"""
    l1_path, cfg = args
    try:
        df = process_one_file(l1_path, cfg)
        if df is None or len(df) == 0:
            return False, l1_path

        stem = Path(l1_path).stem
        info = _parse_l1_stem(stem)
        if info:
            sd = datetime.strptime(info["start"], "%Y%m%dT%H%M%S")
            out_dir = (Path(cfg["OUTPUT_DIR"])
                       / sd.strftime("%Y") / sd.strftime("%m"))
        else:
            out_dir = Path(cfg["OUTPUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.parquet"

        # FIX #14：pyarrow 失败自动 fallback
        try:
            df.to_parquet(out_path, index=False, engine="pyarrow")
        except Exception:
            df.to_parquet(out_path, index=False, engine="fastparquet")

        del df; gc.collect()
        return True, l1_path
    except Exception:
        log.error(f"处理失败: {l1_path}\n{traceback.format_exc()}")
        return False, l1_path


# ===========================================================
# 12. 批处理入口
# ===========================================================
def batch_process(cfg: dict) -> None:
    # 启动前路径检查
    for key in ("L1_DIR", "MSS_FILE", "OUTPUT_DIR"):
        if not cfg.get(key):
            log.error(f"CONFIG['{key}'] 未设置，退出"); return
    if not os.path.exists(cfg["MSS_FILE"]):
        log.error(f"MSS 文件不存在: {cfg['MSS_FILE']}，退出"); return

    l1_files = scan_l1_files(cfg["L1_DIR"])
    if not l1_files:
        log.error("未找到任何 L1b 文件，退出"); return

    Path(cfg["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    log.info(f"共 {len(l1_files)} 个文件  N_WORKERS={cfg['N_WORKERS']}")

    tasks = [(f, cfg) for f in l1_files]
    n_ok = n_fail = 0

    if cfg["N_WORKERS"] == 1:
        for args in tasks:
            ok, _ = _worker(args)
            n_ok += ok; n_fail += (not ok)
    else:
        # FIX #11：_worker 为模块级函数，Windows 下可正常 pickle
        with ProcessPoolExecutor(max_workers=cfg["N_WORKERS"]) as ex:
            futures = {ex.submit(_worker, a): a[0] for a in tasks}
            for fut in as_completed(futures):
                ok, _ = fut.result()
                n_ok += ok; n_fail += (not ok)

    log.info("=" * 60)
    log.info(f"完成：成功 {n_ok}  失败 {n_fail}  总计 {len(l1_files)}")
    log.info(f"输出目录: {cfg['OUTPUT_DIR']}")


# ===========================================================
# 入口
# ===========================================================
if __name__ == "__main__":
    batch_process(CONFIG)
