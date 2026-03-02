import rasterio
import numpy as np
from pathlib import Path

# === 修改为要检查的输出影像路径 ===
tif_path = r"F:\NWP\sentinel1 gee\2023\S1A_EW_GRDM_1SDH_20230412T121730_20230412T121835_048063_05C70C_40A9_EW_HH_HV_angle_int16x100_87caa6ee.tif"

print(f"🔍 检查文件: {Path(tif_path).name}\n")

with rasterio.open(tif_path) as src:
    print(f"CRS: {src.crs}")
    print(f"Shape: {src.height} × {src.width}")
    print(f"Bands: {src.count}")
    print(f"nodata (metadata): {src.nodata}\n")

    # nodata 值（根据元数据或默认 -9999）
    nodata_val = src.nodata if src.nodata is not None else -9999

    # 依次检查每个波段
    for i in range(1, src.count + 1):
        band_name = src.descriptions[i - 1] or f"Band_{i}"
        arr = src.read(i).astype(np.float32)

        total_pixels = arr.size
        nodata_pixels = np.sum(arr == nodata_val)
        valid_pixels = total_pixels - nodata_pixels
        valid_ratio = valid_pixels / total_pixels * 100

        # 基本统计（仅对有效区）
        valid_data = arr[arr != nodata_val]
        mean_val = np.nanmean(valid_data)
        min_val = np.nanmin(valid_data)
        max_val = np.nanmax(valid_data)

        print(f"📘 {band_name}")
        print(f"   有效区比例: {valid_ratio:.2f}%")
        print(f"   平均值: {mean_val:.3f}, 最小: {min_val:.3f}, 最大: {max_val:.3f}\n")

print("✅ 检查完成。")
