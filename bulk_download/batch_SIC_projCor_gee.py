import os
from osgeo import gdal, osr

# ========= 用户配置部分 =========
input_folder = r"D:\S1_CS2_data\SIC\2019\n6250"        # 原始AMS​​R2文件夹
output_folder = r"D:\S1_CS2_data\SIC\2019\n6250_fix"     # 修复后输出文件夹
os.makedirs(output_folder, exist_ok=True)

# ========= 投影定义 =========
EPSG_CODES = {
    "n6250": 3411,   # 北极
    "s6250": 3976,   # 南极
}

def fix_projection(input_path, output_path, epsg_code):
    """重写影像投影定义并保存为新文件"""
    try:
        # 打开输入文件
        ds = gdal.Open(input_path)
        if ds is None:
            print(f"❌ 无法打开: {input_path}")
            return False
        
        # 创建投影定义
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg_code)
        
        # 使用 gdal.Warp 重写CRS并复制到新文件
        gdal.Warp(
            destNameOrDestDS=output_path,
            srcDSOrSrcDSTab=ds,
            dstSRS=srs.ExportToWkt(),
            format="GTiff",
            outputType=gdal.GDT_Float32,  # 保持浮点数据
            dstNodata=-9999,               # 可选 nodata
        )
        print(f"✅ 已修复投影: {os.path.basename(input_path)} → EPSG:{epsg_code}")
        return True

    except Exception as e:
        print(f"⚠️ 处理出错 {input_path}: {e}")
        return False


# ========= 主程序 =========
def batch_fix_amsr2(input_folder, output_folder):
    tifs = [f for f in os.listdir(input_folder) if f.lower().endswith(".tif")]
    print(f"📂 找到 {len(tifs)} 个 .tif 文件")

    for tif in tifs:
        input_path = os.path.join(input_folder, tif)
        output_path = os.path.join(output_folder, tif)

        # 自动识别北/南极版本
        epsg = None
        for key, code in EPSG_CODES.items():
            if key in tif.lower():
                epsg = code
                break
        
        if epsg is None:
            print(f"⚠️ 未识别投影类型 (跳过): {tif}")
            continue
        
        fix_projection(input_path, output_path, epsg)

    print("🎯 全部文件处理完成！")


if __name__ == "__main__":
    batch_fix_amsr2(input_folder, output_folder)
