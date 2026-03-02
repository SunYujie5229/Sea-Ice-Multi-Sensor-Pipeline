import pandas as pd
import xarray as xr
import os
import glob
import geopandas as gpd
from shapely.geometry import Polygon, LineString

# --- 路径配置 ---
MATCHED_PAIRS_CSV = r'C:\Users\TJ002\Desktop\CS2_S1_result\overlap\matched_pairs_final.csv'
S1_METADATA_CSV = r'F:\NWP\Sentinel1 metadata\Metadata\filtered_GRD_results.csv'
CS2_ROOT = r'F:\NWP\CS2_L1'
OUTPUT_DIR = r'E:\Manuscript\boundary' # 输出根目录

def export_by_month():
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 读取数据
    pairs_df = pd.read_csv(MATCHED_PAIRS_CSV)
    s1_meta = pd.read_csv(S1_METADATA_CSV).set_index('Granule Name')
    
    # 按月份存储字典
    monthly_data = {m: {'s1': [], 'cs2': []} for m in range(1, 13)}

    print("开始解析数据并按月份归类...")

    for idx, row in pairs_df.iterrows():
        s1_name = row['sceneName']
        try:
            month = int(s1_name.split('_')[4][4:6])
        except:
            continue
        
        # --- A. 生成 S1 Polygon ---
        if s1_name in s1_meta.index:
            meta = s1_meta.loc[s1_name]
            if isinstance(meta, pd.DataFrame): meta = meta.iloc[0]
            try:
                coords = [
                    (float(meta['Near Start Lon']), float(meta['Near Start Lat'])),
                    (float(meta['Far Start Lon']), float(meta['Far Start Lat'])),
                    (float(meta['Far End Lon']), float(meta['Far End Lat'])),
                    (float(meta['Near End Lon']), float(meta['Near End Lat'])),
                    (float(meta['Near Start Lon']), float(meta['Near Start Lat']))
                ]
                monthly_data[month]['s1'].append({
                    'geometry': Polygon(coords),
                    'SceneName': s1_name,
                    'Date': s1_name.split('_')[4][:8]
                })
            except:
                pass

        # --- B. 生成 CS2 LineString ---
        raw_path = row['cs2_path']
        try:
            date_str = raw_path.split('__')[2].split('_')[0] 
            year_str = date_str[:4]
            search_pattern = os.path.join(CS2_ROOT, year_str, f"*{date_str}*.nc")
            found_files = glob.glob(search_pattern)
            
            if found_files:
                with xr.open_dataset(found_files[0]) as ds:
                    lats, lons = ds.lat_20_ku.values, ds.lon_20_ku.values
                    valid = ~(pd.isna(lats) | pd.isna(lons))
                    if valid.any():
                        monthly_data[month]['cs2'].append({
                            'geometry': LineString(zip(lons[valid], lats[valid])),
                            'S1_Match': s1_name
                        })
        except:
            pass

    # --- 2. 写入文件 (每个月一个 Shapefile) ---
    for m in range(1, 13):
        m_str = f"{m:02d}"
        month_path = os.path.join(OUTPUT_DIR, f"Month_{m_str}")
        if not os.path.exists(month_path): os.makedirs(month_path)

        # 保存 S1
        if monthly_data[m]['s1']:
            gdf_s1 = gpd.GeoDataFrame(monthly_data[m]['s1'], crs="EPSG:4326")
            gdf_s1.to_file(os.path.join(month_path, f"S1_Boundaries_M{m_str}.shp"))

        # 保存 CS2
        if monthly_data[m]['cs2']:
            gdf_cs2 = gpd.GeoDataFrame(monthly_data[m]['cs2'], crs="EPSG:4326")
            gdf_cs2.to_file(os.path.join(month_path, f"CS2_Tracks_M{m_str}.shp"))

    print(f"转换完成！所有文件已保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    export_by_month()