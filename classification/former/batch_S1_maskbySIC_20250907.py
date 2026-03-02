# -*- coding: utf-8 -*-

# Batch mask Sentinel-1 with Bremen SIC (0–100) by date matching
# - S1 folder : F:\NWP\sentinel1 gee\2023
# - SIC folder: E:\NWP\CS2_S1_matched\SIC\n6250
# - Output    : F:\NWP\sentinel1 maskbySIC\2023

# 功能要点：
# 1) 读取 GEE 导出的 S1 堆栈（HH/HV/ANGLE，通常为 int16×100），先基于“原始 int16”识别外部零值。
# 2) 将 Bremen SIC (0–100) 重投影到 S1 网格，并按 [0,100] 有效域、缓冲带与上界阈值构建掩膜。
# 3) 应用掩膜输出 float32（用于后续波段计算）；可选同时输出 int16×100 存档版本。
# 4) 支持批处理；按文件名中的日期（优先匹配 8 位 YYYYMMDD）关联 SIC 影像。


import os
import re
from pathlib import Path
from datetime import datetime, timedelta
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import matplotlib.pyplot as plt

# ===================== 路径 & 基本设置 =====================
S1_FOLDER   = r"F:\NWP\sentinel1 gee\2023"
SIC_FOLDER  = r"E:\NWP\CS2_S1_matched\SIC\n6250"
OUT_DIR     = r"F:\NWP\sentinel1 maskbySIC\2023"
os.makedirs(OUT_DIR, exist_ok=True)

# 匹配 S1 文件的模式（可按需调整）
S1_GLOB_PATTERNS = [
    "*.tif", "*.tiff"
]

# 是否生成 quicklook PNG
GENERATE_QUICKLOOK = True

# ========== GEE 压缩还原 & 保存选项 ==========
# 每个波段的还原因子（把 int16×100 还原到 float）
PER_BAND_SCALE = {"HH": 0.01, "HV": 0.01, "ANGLE": 1.0}
SAVE_INT16_X100 = False       # 是否同时保存一个重新×100压缩的 int16 版本
INT16_NODATA    = -32768

# ========== S1 外部零值掩膜策略 ==========
# 任一极化（HH 或 HV）==0 即视为无效区域（True=有效）
ANY_ZERO_IS_INVALID = True

# ========== SIC 阈值与缓冲带 ==========
# Bremen SIC 范围 0–100；有效域外（<0 或 >100）一律无效。
SIC_WATER_MAX = 10.0          # ≤此值判作水（MASK_MODE='water' 时保留）
SIC_ICE_MIN   = 15.0          # ≥此值且 ≤SIC_ICE_MAX 判作冰
SIC_ICE_MAX   = 100.0         # 海冰上界（也可设 95 更保守）
MASK_MODE     = "ice"         # "ice" 或 "water"
SIC_RESAMPLING = Resampling.bilinear

# ========== 日期匹配设置 ==========
# 从文件名中提取日期：优先 8 位数字 YYYYMMDD；也支持 2023-01-05 / 2023_01_05 / 202301 等常见模式
# 若找不到完全相同日期的 SIC，是否允许在 ±N 天内找最近的（0 = 只允许同日）
SEARCH_NEAREST_DAYS = 0

# ===================== 辅助函数 =====================
def list_s1_files(folder, patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(folder, pat)))
    files = [f for f in files if Path(f).is_file()]
    files.sort()
    return files

def extract_date_from_name(name: str):
    """
    从文件名中提取日期，返回 datetime.date；提取失败返回 None
    策略：
      1) 先找连续8位数字形如 YYYYMMDD
      2) 再尝试 YYYY[-/_]?MM[-/_]?DD
      3) 再尝试 YYYYMM（默认日=15作为中位；不建议，但保底）
    """
    base = Path(name).stem

    # 1) 8位 YYYYMMDD
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", base)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d).date()
        except ValueError:
            pass

    # 2) YYYY[-/_]?MM[-/_]?DD
    m = re.search(r"(20\d{2})[-_/]?(\d{2})[-_/]?(\d{2})", base)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d).date()
        except ValueError:
            pass

    # 3) YYYYMM（fallback，日=15）
    m = re.search(r"(20\d{2})(\d{2})", base)
    if m:
        y, mo = map(int, m.groups())
        try:
            return datetime(y, mo, 15).date()
        except ValueError:
            pass

    return None

def index_sic_by_date(sic_folder):
    """
    建立 {date: [paths]} 索引；一个日期可能有多条（早/晚轨、不同版本）
    """
    idx = {}
    for tif in glob.glob(os.path.join(sic_folder, "*.tif*")):
        d = extract_date_from_name(tif)
        if d is None:
            continue
        idx.setdefault(d, []).append(tif)
    # 每日多文件时，默认按文件名排序（你也可以自定义优先规则）
    for k in idx:
        idx[k].sort()
    return idx

def pick_sic_for_date(sic_index, target_date):
    """
    选择与 target_date 最匹配的 SIC：
      - 优先同日；
      - 若 SEARCH_NEAREST_DAYS > 0，允许 ±N 天内就近匹配（返回最近的那天）。
    返回：path 或 None
    """
    if target_date in sic_index:
        return sic_index[target_date][0]
    if SEARCH_NEAREST_DAYS > 0:
        best = None
        best_dt = None
        for dt, paths in sic_index.items():
            diff = abs((dt - target_date).days)
            if diff <= SEARCH_NEAREST_DAYS:
                if best is None or diff < best:
                    best = diff
                    best_dt = dt
        if best_dt is not None:
            return sic_index[best_dt][0]
    return None

def read_s1_stack(path):
    """
    返回 (data_raw_int16, profile, band_names)
    band_names 尽力从 descriptions 读取；否则猜测为 ['HH','HV','ANGLE'] 前3个
    """
    with rasterio.open(path) as src:
        data = src.read()  # (B,H,W)
        prof = src.profile.copy()
        descs = []
        try:
            descs = [src.descriptions[i] for i in range(src.count)]
        except Exception:
            pass
        if descs and all(d is not None for d in descs):
            names = [d.upper() for d in descs]
        else:
            # 猜测前三为 HH, HV, ANGLE
            names = []
            if src.count >= 1: names.append("HH")
            if src.count >= 2: names.append("HV")
            if src.count >= 3: names.append("ANGLE")
    return data, prof, names

def build_s1_valid_mask_from_zero(data_raw, band_names, any_zero_invalid=True):
    """
    用原始 int16 的 HH/HV 识别外部零值区域。
    any_zero_invalid=True: 只要 HH==0 或 HV==0 就判无效
    """
    up = [n.upper() for n in band_names]
    hh_idx = up.index("HH") if "HH" in up else None
    hv_idx = up.index("HV") if "HV" in up else None
    H, W = data_raw.shape[1], data_raw.shape[2]
    valid = np.ones((H, W), dtype=bool)

    if hh_idx is not None and hv_idx is not None:
        if any_zero_invalid:
            valid &= (data_raw[hh_idx] != 0)
            valid &= (data_raw[hv_idx] != 0)
        else:
            valid &= ~((data_raw[hh_idx] == 0) & (data_raw[hv_idx] == 0))
    elif hh_idx is not None:
        valid &= (data_raw[hh_idx] != 0)
    elif hv_idx is not None:
        valid &= (data_raw[hv_idx] != 0)

    return valid  # True=有效

def restore_to_float(data_raw, band_names, per_band_scale):
    out = data_raw.astype(np.float32)
    up = [n.upper() for n in band_names]
    for i, n in enumerate(up):
        scale = per_band_scale.get(n, 1.0)
        if scale != 1.0:
            out[i] *= scale
    return out

def align_sic_to_s1(sic_path, s1_profile, resampling=SIC_RESAMPLING):
    with rasterio.open(sic_path) as src:
        dst = np.full((s1_profile["height"], s1_profile["width"]), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=s1_profile["transform"],
            dst_crs=s1_profile["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst

def sic_keep_mask_with_bounds(sic_arr, mode, water_max, ice_min, ice_max):
    """
    仅在 sic ∈ [0,100] 的像元上考虑分类，其它一律无效。
    缓冲带 (water_max, ice_min) 统一剔除。
    mode="ice"  : keep = (ice_min ≤ sic ≤ ice_max)
    mode="water": keep = (sic ≤ water_max)
    """
    sic_valid = np.isfinite(sic_arr) & (sic_arr >= 0.0) & (sic_arr <= 100.0)
    is_water = sic_arr <= water_max
    is_ice   = (sic_arr >= ice_min) & (sic_arr <= ice_max)
    if mode == "ice":
        keep = is_ice
    elif mode == "water":
        keep = is_water
    else:
        raise ValueError("MASK_MODE must be 'ice' or 'water'")
    in_buffer = (sic_arr > water_max) & (sic_arr < ice_min)
    keep = keep & sic_valid & (~in_buffer)
    keep = np.where(np.isfinite(sic_arr), keep, False)
    return keep

def save_float32(out_path, data_stack, base_profile, band_names=None):
    prof = base_profile.copy()
    prof.update(count=data_stack.shape[0], dtype="float32", nodata=np.nan,
                compress="DEFLATE", tiled=True, BIGTIFF="IF_SAFER")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(data_stack)
        # 写入波段描述（可在大多数软件中看到）
        if band_names:
            for i, nm in enumerate(band_names, start=1):
                dst.set_band_description(i, str(nm))

def save_int16_x100(out_path, data_stack_float, band_names, base_profile, nodata_val=INT16_NODATA):
    """
    - HH/HV: float ×100 → int16（与 GEE 压缩一致）
    - ANGLE: 不缩放，四舍五入到 int16
    - SIC:   保持 0–100 原尺度（不 ×100），四舍五入到 int16
    """
    out = []
    up = [n.upper() for n in band_names]
    for i, n in enumerate(up):
        arr = data_stack_float[i].copy()

        if n in ("HH", "HV"):
            arr = np.where(np.isfinite(arr), arr * 100.0, np.nan)  # ×100
        elif n == "SIC":
            # 保持 0–100 原尺度（可按需 clip）
            arr = np.where(np.isfinite(arr), np.clip(arr, 0, 100), np.nan)
        # ANGLE 或其他：不缩放

        # 量化到 int16
        arr = np.where(np.isfinite(arr), np.clip(np.round(arr), -32767, 32767), nodata_val).astype(np.int16)
        out.append(arr)

    out = np.stack(out, axis=0)
    prof = base_profile.copy()
    prof.update(count=out.shape[0], dtype="int16", nodata=nodata_val,
                compress="DEFLATE", tiled=True, BIGTIFF="IF_SAFER")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out)
        for i, nm in enumerate(band_names, start=1):
            dst.set_band_description(i, str(nm))

def quicklook_png(out_png, s1_float_masked, sic_aligned, band_names, keep_mask):
    try:
        # 选 HH（或第一个）
        hh_idx = 0
        for i, n in enumerate([n.upper() for n in band_names]):
            if n in ("HH", "SIGMA0_HH"):
                hh_idx = i; break
        hh = s1_float_masked[hh_idx]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axes[0].imshow(hh, vmin=np.nanpercentile(hh, 2), vmax=np.nanpercentile(hh, 98))
        axes[0].set_title(f"S1 {band_names[hh_idx]} (masked)")
        axes[0].axis("off"); fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(sic_aligned)
        axes[1].contour(keep_mask.astype(np.uint8), levels=[0.5], linewidths=0.8)
        axes[1].set_title("Bremen SIC (0–100), contour=kept")
        axes[1].axis("off"); fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close(fig)
    except Exception as e:
        print(f"[Quicklook] Failed: {e}")

# ===================== 主流程（批处理） =====================
def main():
    sic_index = index_sic_by_date(SIC_FOLDER)
    if not sic_index:
        print(f"[WARN] No SIC files indexed under: {SIC_FOLDER}")

    s1_files = list_s1_files(S1_FOLDER, S1_GLOB_PATTERNS)
    if not s1_files:
        print(f"[ERROR] No Sentinel-1 files found under: {S1_FOLDER}")
        return

    print(f"Found {len(s1_files)} S1 files.")
    done = 0
    for s1_path in s1_files:
        s1_name = Path(s1_path).name
        s1_date = extract_date_from_name(s1_name)

        if s1_date is None:
            print(f"[SKIP] {s1_name}: no date found in filename.")
            continue

        sic_path = pick_sic_for_date(sic_index, s1_date)
        if sic_path is None:
            print(f"[MISS] {s1_name}: no SIC found for date {s1_date} (±{SEARCH_NEAREST_DAYS} days).")
            continue

        try:
            # 1) 读取 S1（原始 int16×100）
            s1_raw, s1_profile, band_names = read_s1_stack(s1_path)

            # 2) S1 外部零值掩膜（True=有效）
            s1_valid = build_s1_valid_mask_from_zero(s1_raw, band_names, any_zero_invalid=ANY_ZERO_IS_INVALID)

            # 3) 还原到 float
            s1_float = restore_to_float(s1_raw, band_names, PER_BAND_SCALE)

            # 4) 对齐 SIC
            sic_aligned = align_sic_to_s1(sic_path, s1_profile, resampling=SIC_RESAMPLING)

            # 5) SIC 掩膜（有效域 + 缓冲带 + 上界）
            keep_sic = sic_keep_mask_with_bounds(sic_aligned, MASK_MODE, SIC_WATER_MAX, SIC_ICE_MIN, SIC_ICE_MAX)

 # 6) 组合掩膜
            keep_final = s1_valid & keep_sic

            # 7) 应用掩膜（float 输出用 NaN）
            s1_masked_float = s1_float.copy()
            for b in range(s1_masked_float.shape[0]):
                band = s1_masked_float[b]
                band[~keep_final] = np.nan
                s1_masked_float[b] = band

            # 7.1) 增加一个 SIC 波段（与 keep_final 一致做掩膜）
            sic_masked = sic_aligned.copy()
            sic_masked[~keep_final] = np.nan

            # 拼接为 [HH, HV, ANGLE, SIC]
            stack_with_sic = np.concatenate([s1_masked_float, sic_masked[None, ...]], axis=0)
            band_names_ext = band_names + ["SIC"]

            # 8) 保存（float32 + 可选 int16×100）
            base = Path(s1_path).stem
            out_float = os.path.join(
                OUT_DIR,
                f"{base}_bySIC_{MASK_MODE}_w{int(SIC_WATER_MAX)}_i{int(SIC_ICE_MIN)}_imax{int(SIC_ICE_MAX)}_float32.tif"
            )
            save_float32(out_float, stack_with_sic, s1_profile, band_names_ext)

            if SAVE_INT16_X100:
                out_i16 = os.path.join(
                    OUT_DIR,
                    f"{base}_bySIC_{MASK_MODE}_w{int(SIC_WATER_MAX)}_i{int(SIC_ICE_MIN)}_imax{int(SIC_ICE_MAX)}_int16x100.tif"
                )
                save_int16_x100(out_i16, stack_with_sic, band_names_ext, s1_profile)

            if GENERATE_QUICKLOOK:
                out_png = os.path.join(OUT_DIR, f"{base}_bySIC_quicklook.png")
                quicklook_png(out_png, stack_with_sic, sic_aligned, band_names_ext, keep_final)

            done += 1
            print(f"[OK] {s1_name}  →  {Path(out_float).name}  (+SIC band)")
        except Exception as e:
            print(f"[FAIL] {s1_name}: {e}")

    print(f"Completed: {done}/{len(s1_files)} files processed.")

if __name__ == "__main__":
    main()
