import asf_search as asf
import pandas as pd
import os
import datetime
from dateutil.parser import parse
from datetime import datetime, timedelta
import numpy as np
import traceback
import time
import sys

# 定义研究区域的多边形 (Arctic Canada North)
STUDY_AREA = "POLYGON ((-125.0 74.0, -75.0 74.0, -75.0 77.0, -74.73 77.51, -74.28 78.06, -73.80 78.49, -73.11 78.81, -72.25 79.11, -71.01 79.39, -70.08 79.64, -69.35 79.94, -68.64 80.3, -67.8 80.56, -67.02 80.78, -66.06 80.95, -65.13 81.13, -64.33 81.27, -63.37 81.5, -62.55 81.72, -61.86 81.93, -60.86 82.1, -60.3 82.3, -60.0 82.5, -60.0 85.0, -125.0 85.0, -125.0 74.0))"

# 创建基础输出目录
BASE_OUTPUT_DIR = r"E:\NWP\Sentinel1\Metadata"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# 定义季度
QUARTERS = {
    1: (1, 3),  # Q1: 1月-3月
    2: (4, 6),  # Q2: 4月-6月
    3: (7, 9),  # Q3: 7月-9月
    4: (10, 12)  # Q4: 10月-12月
}

# 定义按季度下载的函数
def download_sentinel1_metadata_by_quarter(year, quarter, polygon=STUDY_AREA, max_results=10000, max_retries=3):
    # 获取季度的开始和结束月份
    start_month, end_month = QUARTERS[quarter]
    
    # 计算季度的开始和结束日期
    start_date = f"{year}-{start_month:02d}-01T00:00:00Z"
    
    # 计算结束日期（下一个季度的第一天）
    if end_month == 12:
        next_year = year + 1
        next_month = 1
    else:
        next_year = year
        next_month = end_month + 1
        
    end_date = f"{next_year}-{next_month:02d}-01T00:00:00Z"
    
    # 创建年份目录
    year_dir = os.path.join(BASE_OUTPUT_DIR, f"{year}")
    os.makedirs(year_dir, exist_ok=True)
    
    # 设置搜索参数
    opts = asf.ASFSearchOptions(**{
        "maxResults": max_results,
        "intersectsWith": polygon,
        "start": start_date,
        "end": end_date,
        "dataset": ["SENTINEL-1"]
    })
    
    # 执行查询，带有重试机制
    for retry in range(max_retries):
        try:
            print(f"正在查询 {year}年第{quarter}季度 (Q{quarter}: {start_month}-{end_month}月) 的Sentinel-1数据...")
            results = asf.search(opts=opts)
            print(f"找到 {len(results)} 条记录")
            break
        except Exception as e:
            if retry < max_retries - 1:
                wait_time = (retry + 1) * 5  # 递增等待时间
                print(f"查询出错: {e}\n等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"查询失败，已达到最大重试次数: {e}")
                print(traceback.format_exc())
                return None
    
    if len(results) == 0:
        print(f"⚠️ {year}年第{quarter}季度没有找到数据")
        return None
    
    # 提取属性并转换为DataFrame
    data = []
    for r in results:
        try:
            # 提取几何信息
            geom = r.geometry['coordinates'][0]
            near_end = geom[0]
            near_start = geom[1]
            far_start = geom[2]
            far_end = geom[3]
            
            # 提取属性
            record = {
                "Granule Name": r.properties.get('sceneName', ''),
                'Platform': r.properties.get('platform', ''),
                'Sensor': r.properties.get('sensor', 'C-SAR'),
                'Beam Mode': r.properties.get('beamModeType', ''),
                'Start Time': r.properties.get('startTime', ''),
                'End Time': r.properties.get('stopTime', ''),
                'Orbit': r.properties.get('orbit', ''),
                'Path Number': r.properties.get('pathNumber', ''),
                'Processing Date': r.properties.get('processingDate', ''),
                'Processing Level': r.properties.get('processingLevel', ''),
                'Center Lat': r.properties.get('centerLat', ''),
                'Center Lon': r.properties.get('centerLon', ''),
                'Near Start Lat': near_start[1], 'Near Start Lon': near_start[0],
                'Far Start Lat': far_start[1], 'Far Start Lon': far_start[0],
                'Far End Lat': far_end[1], 'Far End Lon': far_end[0],
                'Near End Lat': near_end[1], 'Near End Lon': near_end[0],
                'Ascending or Descending': r.properties.get('flightDirection', ''),
                'URL': r.properties.get('url', '')
            }
            
            data.append(record)
        except Exception as e:
            print(f"⚠️ 处理记录时出错: {e}")
            continue
    
    if not data:
        print(f"⚠️ {year}年第{quarter}季度没有有效数据可处理")
        return None
        
    df = pd.DataFrame(data)
    
    # 保存为CSV文件
    filename = f"sentinel1_{year}_Q{quarter}.csv"
    filepath = os.path.join(year_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"✅ CSV已保存: {filepath}")
    
    return df

# 批量处理多个年份和季度的数据
def batch_process_quarterly(start_year, end_year):
    for year in range(start_year, end_year + 1):
        print(f"\n===== 处理 {year} 年数据 =====")
        year_dfs = []
        
        for quarter in range(1, 5):  # 1-4季度
            try:
                df = download_sentinel1_metadata_by_quarter(year, quarter)
                if df is not None:
                    year_dfs.append(df)
            except Exception as e:
                print(f"❌ 处理 {year}年第{quarter}季度时发生错误: {e}")
                print(traceback.format_exc())
                continue
        
        # 合并该年的所有季度数据
        if year_dfs:
            try:
                combined_df = pd.concat(year_dfs)
                combined_filepath = os.path.join(BASE_OUTPUT_DIR, f"{year}", f"sentinel1_{year}_all.csv")
                combined_df.to_csv(combined_filepath, index=False)
                print(f"✅ {year}年合并CSV已保存: {combined_filepath}")
            except Exception as e:
                print(f"❌ 合并 {year}年数据时发生错误: {e}")
                print(traceback.format_exc())
        else:
            print(f"⚠️ {year}年没有数据可合并")

# 主函数
def main():
    print("Sentinel-1元数据季度批量下载工具 (2014-2024)")
    print("==========================================")
    
    try:
        # 设置开始和结束年份
        start_year = 2014  # Sentinel-1发射于2014年
        end_year = 2024
        
        # 批量处理
        batch_process_quarterly(start_year, end_year)
        
        print("\n✅ 所有数据处理完成!")
    except Exception as e:
        print(f"\n❌ 程序执行过程中发生错误: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()