import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
import os
from shapely.geometry import Point, Polygon, LineString
from datetime import timedelta
from tqdm import tqdm


def clip_nc_by_region(nc_path, region_gdf):
    """
    读取 CryoSat-2 .nc 文件，只保留落在 region 范围内的点位
    """
    try:
        ds = xr.open_dataset(nc_path)
        lats = ds['lat_01'].values
        lons = ds['lon_01'].values

        # 构建点的 GeoSeries
        points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(lons, lats)], crs="EPSG:4326")
        
        # region_gdf 已经是 GeoDataFrame，可能是多多边形，做 union
        region_union = region_gdf.unary_union
        
        # 使用 contains 向量判断哪些点在区域内
        mask = points.within(region_union)
        
        # 若无点落在区域中，返回 None
        if not mask.any():
            return None

        return {
            "lat": lats[mask],
            "lon": lons[mask],
            "time": ds['time_cor_01'].values[mask],
            "path": nc_path
        }
    except Exception as e:
        print(f"❌ 读取失败 {nc_path}: {e}")
        return None


def build_s1_polygon(row):
    """
    根据 Sentinel-1 元数据构建足迹多边形
    """
    coords = [
        (row['Near Start Lon'], row['Near Start Lat']),
        (row['Far Start Lon'], row['Far Start Lat']),
        (row['Far End Lon'], row['Far End Lat']),
        (row['Near End Lon'], row['Near End Lat']),
        (row['Near Start Lon'], row['Near Start Lat'])  # 闭合
    ]
    return Polygon(coords)


def generate_cs2_s1_pairs(s1_metadata_path, cs2_base_dir, region_shapefile, output_path, year=None):
    """
    生成 CryoSat-2 和 Sentinel-1 的有效配对
    
    参数:
    s1_metadata_path: Sentinel-1 元数据 CSV 文件路径
    cs2_base_dir: CryoSat-2 数据根目录
    region_shapefile: 研究区域 shapefile 路径
    output_path: 输出结果 CSV 文件路径
    year: 可选，指定年份筛选
    """
    print("开始生成 CS2-S1 配对...")
    
    # 1. 加载研究区域 shapefile
    print("加载研究区域 shapefile...")
    region = gpd.read_file(region_shapefile)
    
    # 2. 加载 Sentinel-1 元数据
    print("加载 Sentinel-1 元数据...")
    s1_df = pd.read_csv(s1_metadata_path, parse_dates=["Start Time", "End Time"])
    # 移除时区信息，避免与 CryoSat-2 时间冲突
    s1_df["Start Time"] = s1_df["Start Time"].dt.tz_localize(None)
    s1_df["End Time"] = s1_df["End Time"].dt.tz_localize(None)
    
    # 如果指定了年份，筛选对应年份的数据
    if year:
        s1_df = s1_df[s1_df["Start Time"].dt.year == year]
        print(f"筛选 {year} 年的 Sentinel-1 数据，共 {len(s1_df)} 条记录")
    
    # 3. 按区域筛选 CryoSat-2 数据
    print("按区域筛选 CryoSat-2 数据...")
    cs2_clipped_summary = []
    
    # 确定要处理的 CS2 目录
    if year:
        cs2_dirs = [os.path.join(cs2_base_dir, str(year))]
    else:
        cs2_dirs = [os.path.join(cs2_base_dir, d) for d in os.listdir(cs2_base_dir) 
                   if os.path.isdir(os.path.join(cs2_base_dir, d))]
    
    for cs2_dir in cs2_dirs:
        print(f"处理目录: {cs2_dir}")
        for root, _, files in os.walk(cs2_dir):
            for f in tqdm(files, desc=f"处理 {os.path.basename(root)}"):
                if f.endswith(".nc"):
                    full_path = os.path.join(root, f)
                    clipped = clip_nc_by_region(full_path, region)
                    if clipped is not None:
                        times = pd.to_datetime(clipped["time"])
                        cs2_clipped_summary.append({
                            "cs2_path": full_path,
                            "cs2_start": times[0],
                            "cs2_end": times[-1],
                            "lat": clipped["lat"],
                            "lon": clipped["lon"],
                            "time": clipped["time"]
                        })
    
    print(f"✅ 区域内 CS2 文件数：{len(cs2_clipped_summary)}")
    
    # 4. 第一步：时间重合判断（coarse temporal filter）
    print("第一步：时间重合判断...")
    time_matches = []
    
    for _, s1_row in tqdm(s1_df.iterrows(), total=len(s1_df), desc="时间匹配"):
        s1_start, s1_end = s1_row["Start Time"], s1_row["End Time"]
        s1_lon, s1_lat = s1_row["Center Lon"], s1_row["Center Lat"]
        
        # 构造 ±2 小时窗口
        time_window_start = s1_start - timedelta(hours=2)
        time_window_end = s1_end + timedelta(hours=2)
        
        for cs2 in cs2_clipped_summary:
            # 判断是否时间重叠（±2小时窗口）
            if (cs2["cs2_end"] >= time_window_start) and (cs2["cs2_start"] <= time_window_end):
                # 简易空间粗筛（S1 中心点在 CS2 点 bbox 内）
                if (cs2["lat"].min() <= s1_lat <= cs2["lat"].max()) and \
                   (cs2["lon"].min() <= s1_lon <= cs2["lon"].max()):
                    time_matches.append({
                        "sceneName": s1_row["Granule Name"],
                        "s1_start": s1_start,
                        "s1_end": s1_end,
                        "s1_lon": s1_lon,
                        "s1_lat": s1_lat,
                        "cs2_path": cs2["cs2_path"],
                        "cs2_start": cs2["cs2_start"],
                        "cs2_end": cs2["cs2_end"]
                    })
    
    print(f"✅ 时间匹配成功数：{len(time_matches)}")
    
    # 5. 第二步：空间重合判断（polygon intersection）
    print("第二步：空间重合判断...")
    spatial_matches = []
    
    for match in tqdm(time_matches, desc="空间匹配"):
        # 找到对应的 Sentinel-1 行信息
        s1_row = s1_df[s1_df["Granule Name"] == match["sceneName"]].iloc[0]
        
        # 构建 S1 足迹多边形
        s1_poly = build_s1_polygon(s1_row)
        
        # 打开对应 CS2 文件
        try:
            ds = xr.open_dataset(match["cs2_path"])
            lat = ds["lat_01"].values
            lon = ds["lon_01"].values
            time = pd.to_datetime(ds["time_cor_01"].values)
            
            # 构建 CS2 轨迹线
            cs2_points = [Point(lo, la) for lo, la in zip(lon, lat)]
            if len(cs2_points) >= 2:  # 至少需要两个点才能构成线
                cs2_line = LineString(cs2_points)
                
                # 判断轨迹线是否与多边形相交
                if cs2_line.intersects(s1_poly):
                    # 第三步（可选）：CS2 有效穿越点个数统计
                    # 找出同时满足时间和空间条件的点
                    time_window_start = match["s1_start"] - timedelta(hours=2)
                    time_window_end = match["s1_end"] + timedelta(hours=2)
                    
                    valid_points = []
                    for lo, la, t in zip(lon, lat, time):
                        if time_window_start <= t <= time_window_end:
                            pt = Point(lo, la)
                            if s1_poly.contains(pt):
                                valid_points.append({
                                    "cs2_time": t,
                                    "cs2_lat": la,
                                    "cs2_lon": lo
                                })
                    
                    if valid_points:  # 只有有效点才添加到结果中
                        valid_times = [p["cs2_time"] for p in valid_points]
                        spatial_matches.append({
                            "sceneName": match["sceneName"],
                            "s1_start": match["s1_start"],
                            "s1_end": match["s1_end"],
                            "s1_url": s1_row["URL"] if "URL" in s1_row else "",
                            "cs2_path": match["cs2_path"],
                            "cs2_time_min": min(valid_times),
                            "cs2_time_max": max(valid_times),
                            "num_matched_points": len(valid_points)
                        })
        except Exception as e:
            print(f"❌ 处理失败 {match['cs2_path']}: {e}")
    
    print(f"✅ 空间匹配成功数：{len(spatial_matches)}")
    
    # 6. 保存结果
    result_df = pd.DataFrame(spatial_matches)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"✅ 结果已保存至：{output_path}")
    
    return result_df


if __name__ == "__main__":
    # 示例用法
    s1_metadata_path = r"E:\NWP\Sentinel1\Metadata\2023\sentinel1_2023_all.csv"
    cs2_base_dir = r"E:\NWP\CS2"
    region_shapefile = r"C:\Users\TJ002\Desktop\code\Cal_code_data\NWP_orbit_processing\Arctic_Canada_North.shp"
    output_path = r"E:\NWP\CS2_S1_matched\cs2_s1_pairs_2023.csv"
    
    # 生成 2023 年的配对
    pairs_df = generate_cs2_s1_pairs(
        s1_metadata_path=s1_metadata_path,
        cs2_base_dir=cs2_base_dir,
        region_shapefile=region_shapefile,
        output_path=output_path,
        year=2023
    )
    
    print(pairs_df.head())
    print(f"总共找到 {len(pairs_df)} 个有效配对")