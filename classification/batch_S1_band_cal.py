# # -*- coding: utf-8 -*-
# """
# Append SAR feature bands to each GeoTIFF in a folder
# Input  : masked float32 stacks with bands [HH, HV, ANGLE, SIC] (NaN masked)
# Output : same bands + 5 new bands: [HH_norm, HV_norm, HH_div_HV, HH_minus_HV, sum_div_diff]
# """

import os
import glob
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling

# ========== 路径设置 ==========
INPUT_DIR  = r"F:\NWP\sentinel1 maskbySIC\2023"           # 你的输入文件夹
OUTPUT_DIR = r"F:\NWP\sentinel1 maskbySIC\2023_features"  # 新的输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 匹配哪些文件
GLOB_PATTERNS = ["*.tif", "*.tiff"]

# ========== 归一化与数值安全设置 ==========
# 使用稳健百分位数做 0-1 归一化（避免极端值影响）
NORM_LO_PCT = 2.0
NORM_HI_PCT = 98.0
# 除法时避免除以 0（仅在非 NaN 像元上起作用；NaN 会被保留）
EPS = 1e-6

# ========== 工具函数 ==========
def list_tifs(folder, patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    files = [f for f in files if Path(f).is_file()]
    files.sort()
    return files

def find_band_indices(src):
    """
    尝试通过 band description 定位 HH/HV/ANGLE/SIC
    若无描述，则按常见顺序推断：HH(0), HV(1), ANGLE(2), SIC(3)
    返回: dict，如 {"HH":0, "HV":1, "ANGLE":2, "SIC":3}（存在的才返回）
    """
    mapping = {}
    try:
        descs = [src.descriptions[i] for i in range(src.count)]
    except Exception:
        descs = [None] * src.count

    # 先按描述匹配
    for i, d in enumerate(descs):
        if not d:
            continue
        u = d.upper()
        if "HH" == u or "SIGMA0_HH" == u:
            mapping["HH"] = i
        elif "HV" == u or "SIGMA0_HV" == u:
            mapping["HV"] = i
        elif "ANGLE" == u or "INCIDENCE_ANGLE" in u:
            mapping["ANGLE"] = i
        elif "SIC" == u:
            mapping["SIC"] = i

    # 没找到就按默认顺序猜测
    if "HH" not in mapping and src.count >= 1:
        mapping["HH"] = 0
    if "HV" not in mapping and src.count >= 2:
        mapping["HV"] = 1
    if "ANGLE" not in mapping and src.count >= 3:
        mapping["ANGLE"] = 2
    if "SIC" not in mapping and src.count >= 4:
        mapping["SIC"] = 3

    return mapping

def robust_minmax_norm(arr, lo_pct=2.0, hi_pct=98.0):
    """
    基于百分位的 0-1 归一化，保留 NaN。
    """
    out = arr.astype(np.float32).copy()
    valid = np.isfinite(out)
    if not np.any(valid):
        return out  # 全 NaN 直接返回

    lo = np.nanpercentile(out, lo_pct)
    hi = np.nanpercentile(out, hi_pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # 退化情况：直接返回原值（或全 0）
        return out

    out[valid] = (out[valid] - lo) / (hi - lo)
    out[valid] = np.clip(out[valid], 0.0, 1.0)
    return out

def compute_features(hh, hv, eps=EPS):
    """
    输入：HH, HV（float32，带 NaN 掩膜）
    输出：5 个特征波段：
      - hh_norm, hv_norm
      - hh_div_hv = HH/HV
      - hh_minus_hv = HH - HV
      - sum_div_diff = (HH+HV)/(HH-HV)
    """
    # 归一化（稳健百分位）
    hh_norm = robust_minmax_norm(hh)
    hv_norm = robust_minmax_norm(hv)

    # 比值 HH/HV
    with np.errstate(divide='ignore', invalid='ignore'):
        hh_div_hv = np.where(np.isfinite(hh) & np.isfinite(hv),
                             hh / np.where(np.abs(hv) < eps, np.nan, hv),
                             np.nan).astype(np.float32)

    # 差值 HH - HV
    hh_minus_hv = (hh - hv).astype(np.float32)

    # (HH+HV)/(HH-HV)
    denom = hh_minus_hv
    with np.errstate(divide='ignore', invalid='ignore'):
        sum_div_diff = np.where(
            np.isfinite(hh) & np.isfinite(hv),
            (hh + hv) / np.where(np.abs(denom) < eps, np.nan, denom),
            np.nan
        ).astype(np.float32)

    return hh_norm, hv_norm, hh_div_hv, hh_minus_hv, sum_div_diff

def write_with_appended_bands(out_path, base_profile, orig_stack, orig_band_names, feats_stack, feats_names):
    """
    将原始波段与新特征波段拼接写出；float32 + NaN nodata；写 band descriptions
    """
    stack = np.concatenate([orig_stack, feats_stack], axis=0)
    names = list(orig_band_names) + list(feats_names)

    prof = base_profile.copy()
    prof.update(
        count=stack.shape[0],
        dtype="float32",
        nodata=np.nan,
        compress="DEFLATE",
        tiled=True,
        BIGTIFF="IF_SAFER"
    )

    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(stack)
        for i, nm in enumerate(names, start=1):
            try:
                dst.set_band_description(i, str(nm))
            except Exception:
                pass

# ========== 主流程 ==========
def main():
    files = list_tifs(INPUT_DIR, GLOB_PATTERNS)
    if not files:
        print(f"[ERROR] No tif found in: {INPUT_DIR}")
        return
    print(f"Found {len(files)} files.")

    done = 0
    for fp in files:
        try:
            with rasterio.open(fp) as src:
                profile = src.profile.copy()
                data = src.read()  # (B,H,W) float32 + NaN
                band_map = find_band_indices(src)

                if "HH" not in band_map or "HV" not in band_map:
                    print(f"[SKIP] {Path(fp).name}: cannot locate HH/HV bands.")
                    continue

                hh = data[band_map["HH"]].astype(np.float32)
                hv = data[band_map["HV"]].astype(np.float32)

                # 计算特征
                hh_norm, hv_norm, hh_div_hv, hh_minus_hv, sum_div_diff = compute_features(hh, hv, eps=EPS)

                feats = np.stack([hh_norm, hv_norm, hh_div_hv, hh_minus_hv, sum_div_diff], axis=0)
                feat_names = ["HH_norm", "HV_norm", "HH_div_HV", "HH_minus_HV", "sum_div_diff"]

                # 原始波段名（如果缺失就给默认）
                try:
                    orig_names = [src.descriptions[i] if src.descriptions[i] else f"Band_{i+1}" for i in range(src.count)]
                except Exception:
                    orig_names = [f"Band_{i+1}" for i in range(src.count)]

                # 输出路径
                base = Path(fp).stem
                out_path = os.path.join(OUTPUT_DIR, f"{base}_with_feats.tif")

                write_with_appended_bands(out_path, profile, data, orig_names, feats, feat_names)
                print(f"[OK] {Path(fp).name} → {Path(out_path).name}")
                done += 1

        except Exception as e:
            print(f"[FAIL] {Path(fp).name}: {e}")

    print(f"Completed: {done}/{len(files)} files.")

if __name__ == "__main__":
    main()
