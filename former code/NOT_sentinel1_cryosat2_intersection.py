import os
import pandas as pd
import netCDF4 as nc
import geopandas as gpd
from shapely.geometry import Point, Polygon
from datetime import datetime, timedelta
from dateutil.parser import parse
import numpy as np
from tqdm import tqdm
import concurrent.futures
import threading
import re

# 创建一个线程锁用于Excel写入和进度更新
excel_lock = threading.Lock()

# 创建输出目录
output_dir = r"E:\NWP\sentinel1_cryosat2_intersection"
os.makedirs(output_dir, exist_ok=True)

def create_polygon_from_coordinates(row):
    """
    从Sentinel-1数据行中提取坐标并创建Polygon对象
    
    参数:
    row -- 包含坐标信息的DataFrame行
    
    返回:
    shapely.geometry.Polygon对象或None
    """
    try:
        # 检查是否有所有必要的坐标字段
        required_fields = [
            'Near Start Lon', 'Near Start Lat',
            'Far Start Lon', 'Far Start Lat',
            'Far End Lon', 'Far End Lat',
            'Near End Lon', 'Near End Lat'
        ]
        
        if not all(field in row for field in required_fields):
            return None
        
        # 提取坐标点
        coords = [
            (float(row['Near Start Lon']), float(row['Near Start Lat'])),
            (float(row['Far Start Lon']), float(row['Far Start Lat'])),
            (float(row['Far End Lon']), float(row['Far End Lat'])),
            (float(row['Near End Lon']), float(row['Near End Lat'])),
            (float(row['Near Start Lon']), float(row['Near Start Lat']))  # 闭合多边形
        ]
        
        return Polygon(coords)
    except Exception as e:
        print(f"创建多边形时出错: {e}")
        return None

def filter_by_time_and_space(sentinel1_df, cryosat2_nc_path, time_window_hours=2, year_filter=None):
    """
    根据时间和空间条件过滤Sentinel-1数据与CryoSat-2数据的交集
    
    参数:
    sentinel1_df -- 包含Sentinel-1数据的DataFrame
    cryosat2_nc_path -- CryoSat-2 NC文件路径
    time_window_hours -- 时间窗口大小（小时）
    
    返回:
    过滤后的DataFrame
    """
    if sentinel1_df is None or len(sentinel1_df) == 0:
        print("没有Sentinel-1数据可供过滤")
        return None
    
    try:
        # 读取CryoSat-2 NC文件
        ds = nc.Dataset(cryosat2_nc_path, "r")
        
        # 检查必要的变量是否存在
        required_vars = ['lon_01', 'lat_01', 'time_cor_01']
        if not all(var in ds.variables for var in required_vars):
            print(f"文件 {cryosat2_nc_path} 缺少必要的变量")
            ds.close()
            return None
            
        # 如果设置了年份过滤，检查文件名中是否包含该年份
        if year_filter is not None:
            # 从文件名中提取年份
            file_name = os.path.basename(cryosat2_nc_path)
            # 尝试从文件名中提取年份
            year_match = re.search(r'\d{4}', file_name)
            if year_match:
                file_year = year_match.group(0)
                if file_year != str(year_filter):
                    print(f"文件 {file_name} 年份不匹配，跳过")
                    ds.close()
                    return None
            else:
                # 如果文件名中没有年份，尝试从其他属性中获取
                if hasattr(ds, 'sensing_start'):
                    sensing_year = parse(ds.sensing_start).replace(tzinfo=None).year
                    if sensing_year != year_filter:
                        print(f"文件 {file_name} 年份不匹配，跳过")
                        ds.close()
                        return None
        
        # 获取CryoSat-2数据的经纬度和时间
        cs2_lons = ds.variables['lon_01'][:]
        cs2_lats = ds.variables['lat_01'][:]
        cs2_times = ds.variables['time_cor_01'][:]
        
        # 获取文件的时间信息
        sensing_start = ds.sensing_start if hasattr(ds, 'sensing_start') else None
        
        # 转换CryoSat-2时间格式
        cs2_datetimes = []
        if sensing_start:
            try:
                # 如果有sensing_start属性，使用它作为参考时间
                base_time = parse(sensing_start).replace(tzinfo=None)  # 移除时区信息
                for t in cs2_times:
                    # 假设时间是以秒为单位的偏移量
                    cs2_datetimes.append(base_time + timedelta(seconds=float(t)))
            except Exception as e:
                print(f"转换时间出错: {e}")
                ds.close()
                return None
        else:
            # 尝试其他可能的时间格式
            try:
                # 假设时间是以秒为单位的时间戳，起始日期为1970-01-01
                for t in cs2_times:
                    cs2_datetimes.append(datetime.utcfromtimestamp(t))  # utcfromtimestamp返回的是无时区信息的datetime
            except:
                print("无法转换CryoSat-2时间格式")
                ds.close()
                return None
        
        # 创建CryoSat-2点的列表
        cs2_points = [Point(lon, lat) for lon, lat in zip(cs2_lons, cs2_lats)]
        
        # 过滤Sentinel-1数据
        filtered_rows = []
        
        # 为每行Sentinel-1数据创建多边形
        for idx, row in sentinel1_df.iterrows():
            try:
                # 解析Sentinel-1的开始和结束时间
                start_time_field = 'Start Time' if 'Start Time' in row else 'startTime'
                end_time_field = 'End Time' if 'End Time' in row else 'stopTime'
                
                if start_time_field not in row or end_time_field not in row:
                    print(f"行 {idx} 缺少时间字段")
                    continue
                    
                # 确保解析的时间没有时区信息
                s1_start = parse(row[start_time_field]).replace(tzinfo=None)
                s1_end = parse(row[end_time_field]).replace(tzinfo=None)
                
                # 创建Sentinel-1数据的多边形
                s1_polygon = create_polygon_from_coordinates(row)
                if s1_polygon is None:
                    print(f"行 {idx} 无法创建多边形")
                    continue
                
                # 检查时间和空间交集
                has_intersection = False
                for cs2_point, cs2_time in zip(cs2_points, cs2_datetimes):
                    # 检查时间交集
                    time_diff = min(
                        abs((s1_start - cs2_time).total_seconds()),
                        abs((s1_end - cs2_time).total_seconds())
                    )
                    
                    # 如果时间差小于指定的时间窗口，检查空间交集
                    if time_diff <= time_window_hours * 3600:
                        # 检查点是否在多边形内
                        if s1_polygon.contains(cs2_point):
                            has_intersection = True
                            break
                
                if has_intersection:
                    filtered_rows.append(row)
            except Exception as e:
                print(f"处理行 {idx} 时出错: {e}")
                continue
        
        ds.close()
        
        # 创建过滤后的DataFrame
        filtered_df = pd.DataFrame(filtered_rows)
        
        if len(filtered_df) == 0:
            print("过滤后没有符合条件的Sentinel-1数据")
            return None
        
        return filtered_df
    
    except Exception as e:
        print(f"过滤过程中出错: {e}")
        return None

def process_cryosat2_file(args):
    """
    处理单个CryoSat-2文件与Sentinel-1数据的交集
    
    参数:
    args -- 包含处理参数的元组 (sentinel1_df, nc_file_path, time_window_hours, year_filter)
    
    返回:
    (nc_file_path, filtered_df) 元组
    """
    sentinel1_df, nc_file_path, time_window_hours, year_filter = args
    try:
        filtered_df = filter_by_time_and_space(sentinel1_df, nc_file_path, time_window_hours, year_filter)
        return (nc_file_path, filtered_df)
    except Exception as e:
        print(f"处理文件 {nc_file_path} 时出错: {e}")
        return (nc_file_path, None)

def batch_process_cryosat2_files(sentinel1_df, cryosat2_folder, time_window_hours=2, max_workers=8, year_filter=None):
    """
    批量处理CryoSat-2文件与Sentinel-1数据的交集
    
    参数:
    sentinel1_df -- 包含Sentinel-1数据的DataFrame
    cryosat2_folder -- 包含CryoSat-2 NC文件的文件夹路径
    time_window_hours -- 时间窗口大小（小时）
    max_workers -- 并行处理的最大线程数
    
    返回:
    包含所有交集结果的DataFrame
    """
    if sentinel1_df is None or len(sentinel1_df) == 0:
        print("没有Sentinel-1数据可供处理")
        return None
    
    if not os.path.exists(cryosat2_folder):
        print(f"CryoSat-2文件夹不存在: {cryosat2_folder}")
        return None
    
    # 查找所有NC文件
    nc_files = []
    for root, dirs, files in os.walk(cryosat2_folder):
        for file in files:
            if file.lower().endswith(".nc"):
                nc_files.append(os.path.join(root, file))
    
    if not nc_files:
        print(f"在 {cryosat2_folder} 中未找到NC文件")
        return None
    
    print(f"找到 {len(nc_files)} 个CryoSat-2 NC文件")
    
    # 准备并行处理参数
    process_args = [(sentinel1_df, nc_file, time_window_hours, year_filter) for nc_file in nc_files]
    
    # 使用线程池并行处理
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_cryosat2_file, args) for args in process_args]
        
        # 使用tqdm显示进度
        with tqdm(total=len(nc_files), desc="处理CryoSat-2文件") as pbar:
            for future in concurrent.futures.as_completed(futures):
                nc_file, filtered_df = future.result()
                if filtered_df is not None and len(filtered_df) > 0:
                    # 添加文件信息列
                    filtered_df['CryoSat2_File'] = os.path.basename(nc_file)
                    
                    # 确保包含S1的URL字段
                    if 'URL' not in filtered_df.columns and 'url' in filtered_df.columns:
                        filtered_df['URL'] = filtered_df['url']
                    elif 'URL' not in filtered_df.columns and 'downloadLink' in filtered_df.columns:
                        filtered_df['URL'] = filtered_df['downloadLink']
                    elif 'URL' not in filtered_df.columns:
                        # 如果没有URL字段，尝试从其他字段构建
                        if 'Filename' in filtered_df.columns:
                            filtered_df['URL'] = "需要手动构建URL: " + filtered_df['Filename']
                        else:
                            filtered_df['URL'] = "未找到URL信息"
                    
                    all_results.append(filtered_df)
                pbar.update(1)
    
    # 合并所有结果
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # 保存合并结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"sentinel1_cryosat2_intersection_{timestamp}.csv")
        combined_df.to_csv(output_file, index=False)
        print(f"✅ 已保存交集结果到: {output_file}")
        
        return combined_df
    else:
        print("未找到任何交集")
        return None

# 配置参数（在此处手动设置，无需交互输入）
# 修改以下变量值以适应您的需求
SENTINEL1_CSV_PATH = r"E:\NWP\Sentinel1\Metadata\2023\sentinel1_2023_all.csv"  # Sentinel-1 CSV文件路径
CRYOSAT2_FOLDER_PATH = r"E:\NWP\CS2\test2023"    # CryoSat-2 NC文件夹路径
TIME_WINDOW_HOURS = 2.0                                           # 时间窗口大小（小时）
YEAR_FILTER = 2023                                              # 年份筛选（设为None则不筛选）

# 主函数
def main():
    print("Sentinel-1与CryoSat-2时空交集过滤工具")
    print("="*33)
    
    # 检查文件和文件夹是否存在
    if not os.path.exists(SENTINEL1_CSV_PATH):
        print(f"文件不存在: {SENTINEL1_CSV_PATH}")
        return
    
    if not os.path.exists(CRYOSAT2_FOLDER_PATH):
        print(f"文件夹不存在: {CRYOSAT2_FOLDER_PATH}")
        return
    
    # 读取Sentinel-1数据
    print(f"正在读取Sentinel-1数据: {SENTINEL1_CSV_PATH}")
    sentinel1_df = pd.read_csv(SENTINEL1_CSV_PATH)
    print(f"已读取Sentinel-1数据，包含 {len(sentinel1_df)} 条记录")
    
    # 显示处理参数
    print(f"\n处理参数:")
    print(f"- CryoSat-2文件夹: {CRYOSAT2_FOLDER_PATH}")
    print(f"- 时间窗口: {TIME_WINDOW_HOURS} 小时")
    print(f"- 年份筛选: {YEAR_FILTER if YEAR_FILTER is not None else '无'}")
    
    # 批量处理
    result_df = batch_process_cryosat2_files(sentinel1_df, CRYOSAT2_FOLDER_PATH, TIME_WINDOW_HOURS, year_filter=YEAR_FILTER)
    
    if result_df is not None:
        print(f"找到 {len(result_df)} 条交集记录")
        
        # 显示结果中包含的字段
        print("\n结果包含以下字段:")
        for col in result_df.columns:
            print(f"- {col}")
    else:
        print("未找到交集记录")

if __name__ == "__main__":
    main()