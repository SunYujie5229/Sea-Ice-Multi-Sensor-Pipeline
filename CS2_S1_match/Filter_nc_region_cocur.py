import os
from netCDF4 import Dataset
from openpyxl import Workbook
import shutil
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 创建一个线程锁用于Excel写入和进度更新
excel_lock = threading.Lock()

def process_nc_file(args):
    nc_file_path, year_folder, region, required_vars = args
    try:
        file = os.path.basename(nc_file_path)
        # 从文件名中提取年份
        date_parts = [part for part in file.split('_') if len(part) >= 8 and part[0:8].isdigit()]
        if not date_parts:
            return ["未知", nc_file_path, "无法提取年份", 0, 0, "Unknown"]
            
        year = int(date_parts[0][0:4])  # 提取年份
        
        # 检查年份是否在指定范围内
        if year < start_year or year > end_year:
            return None

        nc_data = Dataset(nc_file_path, "r")
        
        # 检查必要的变量是否存在
        if not all(var in nc_data.variables for var in required_vars):
            nc_data.close()
            return [year, nc_file_path, "缺少必要变量", 0, 0, "Unknown"]

        # 获取经纬度数据
        lons = nc_data.variables[required_vars[0]][:]
        lats = nc_data.variables[required_vars[1]][:]
        
        # 获取文件的时间信息
        sensing_start = nc_data.sensing_start if hasattr(nc_data, 'sensing_start') else 'Unknown'

        # 检查是否有点落在研究区域内
        total_points = len(lons)
        points_in_region = 0
        data_within_range = False
        
        for lon, lat in zip(lons, lats):
            point = Point(lon, lat)
            if any(region.contains(point)):
                points_in_region += 1
                data_within_range = True

        if data_within_range:
            # 确保年份文件夹存在
            os.makedirs(year_folder, exist_ok=True)
            target_path = os.path.join(year_folder, file)
            if not os.path.exists(target_path):
                shutil.copy2(nc_file_path, target_path)
            return [year, nc_file_path, "是", total_points, points_in_region, sensing_start]
        else:
            return [year, nc_file_path, "否", total_points, 0, sensing_start]

    except Exception as e:
        return ["未知", nc_file_path, f"处理出错: {str(e)}", 0, 0, "Unknown"]
    finally:
        if 'nc_data' in locals():
            nc_data.close()

# 主程序部分
if __name__ == "__main__":
    # 读取研究区域的shapefile
    shp_path = r'C:\Users\TJ002\Desktop\code\Cal_code_data\NWP_orbit_processing\Arctic_Canada_North.shp'
    region = gpd.read_file(shp_path)

    # 根文件夹路径
    root_folder = r'Z:\Cryosat\Cryosat-2_bsaeE\SIR_SIN_L2'
    base_output_folder = r'E:\NWP\CS2'

    # 设置年份范围
    start_year = 2024
    end_year = 2024

    # 检查根文件夹是否存在
    if not os.path.exists(root_folder):
        print(f"根文件夹路径不存在: {root_folder}")
        exit(1)

    # 创建基础输出文件夹
    if not os.path.exists(base_output_folder):
        os.makedirs(base_output_folder)

    # 创建一个Excel工作簿用于记录处理的文件
    wb = Workbook()
    ws = wb.active
    ws.append(["年份", "NC文件路径", "是否在研究区域内", "数据点总数", "区域内点数", "文件时间"])

    # 首先统计需要处理的文件总数
    print("正在统计文件数量...")
    total_files = 0
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".nc"):
                total_files += 1

    print(f"共找到 {total_files} 个NC文件")
    
    # 构建NC文件列表和参数
    nc_files = []
    required_vars = ['lon_01', 'lat_01']
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".nc"):
                nc_file_path = os.path.join(root, file)
                # 从文件名中提取年份
                date_parts = [part for part in file.split('_') if len(part) >= 8 and part[0:8].isdigit()]
                if date_parts:
                    year = int(date_parts[0][0:4])  # 提取年份
                    if start_year <= year <= end_year:
                        year_folder = os.path.join(base_output_folder, str(year))
                        args = (nc_file_path, year_folder, region, required_vars)
                        nc_files.append(args)
    
    # 使用线程池处理文件
    with ThreadPoolExecutor(max_workers=8) as executor:  # 可以根据需要调整线程数
        futures = [executor.submit(process_nc_file, args) for args in nc_files]
        
        # 使用tqdm显示进度
        with tqdm(total=len(nc_files), desc="处理进度") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    with excel_lock:
                        ws.append(result)
                pbar.update(1)

    # 保存Excel文件
    excel_file_path = os.path.join(base_output_folder, "CS2_processing_record.xlsx")
    try:
        wb.save(excel_file_path)
        print(f"\n已保存处理记录到: {excel_file_path}")
    except Exception as e:
        print(f"\n保存Excel文件失败: {e}")