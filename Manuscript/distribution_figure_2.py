import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon
import os
import glob
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np

# --- 路径配置 ---
MATCHED_PAIRS_CSV = r'C:\Users\TJ002\Desktop\CS2_S1_result\overlap\matched_pairs_final.csv'
S1_METADATA_CSV = r'F:\NWP\Sentinel1 metadata\Metadata\filtered_GRD_results.csv'
CS2_ROOT = r'F:\NWP\CS2_L1'

# --- 月份颜色与名称映射 ---
month_colors = {
    1: '#1f77b4', 2: '#aec7e8', 3: '#ff7f0e', 4: '#ffbb78',
    5: '#2ca02c', 6: '#98df8a', 7: '#d62728', 8: '#ff9896',
    9: '#9467bd', 10: '#c5b0d5', 11: '#8c564b', 12: '#c49c94'
}
month_names = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}

def add_north_arrow(ax, location=(0.05, 0.95), size=40):
    """添加指北针"""
    x, y = location
    ax.annotate('N', xy=(x, y), xytext=(x, y-0.05),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=15, fontweight='bold',
                xycoords='axes fraction')

def add_scale_bar(ax, length=500, location=(0.1, 0.05)):
    """
    添加比例尺 (默认500km)
    location: (x, y) 轴分数坐标
    """
    # 获取当前地图的投影系统
    crs = ax.projection
    
    # 在地图底部中心计算单位长度
    x0, x1, y0, y1 = ax.get_extent()
    center_x = (x0 + x1) / 2
    center_y = y0 + (y1 - y0) * location[1]
    
    # 转换 500km 到投影坐标单位 (米)
    length_m = length * 1000
    
    # 绘制比例尺线段
    sb_x = [center_x - length_m/2, center_x + length_m/2]
    sb_y = [center_y, center_y]
    
    ax.plot(sb_x, sb_y, color='black', linewidth=2, transform=crs, zorder=5)
    ax.text(center_x, center_y + (y1-y0)*0.02, f'{length} km', 
            transform=crs, ha='center', va='bottom', fontsize=10, fontweight='bold')

def plot_nwp_distribution():
    # 1. 读取数据
    pairs_df = pd.read_csv(MATCHED_PAIRS_CSV)
    s1_meta = pd.read_csv(S1_METADATA_CSV).set_index('Granule Name')
    
    # 2. 初始化地图
    fig = plt.figure(figsize=(14, 11)) # 稍微增加高度
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-100))
    ax.set_extent([-140, -50, 65, 85], ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor='#e0e0e0', edgecolor='black', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5, zorder=3)
    gl.top_labels = False # 避免标签与标题重叠
    gl.right_labels = False

    print("开始按日期检索并绘制轨道...")
    
    active_months = set()

    for idx, row in pairs_df.iterrows():
        s1_name = row['sceneName']
        
        # --- A. CryoSat-2 轨道绘制 ---
        raw_path = row['cs2_path']
        try:
            date_str = raw_path.split('__')[2].split('_')[0] 
            year_str = date_str[:4]
            search_pattern = os.path.join(CS2_ROOT, year_str, f"*{date_str}*.nc")
            found_files = glob.glob(search_pattern)
            
            if found_files:
                cs2_file = found_files[0]
                with xr.open_dataset(cs2_file) as ds:
                    lats = ds.lat_20_ku.values
                    lons = ds.lon_20_ku.values
                    ax.plot(lons, lats, color='gray', linewidth=0.4, 
                            alpha=0.3, transform=ccrs.Geodetic(), zorder=3)
        except:
            pass

        # --- B. Sentinel-1 边框绘制 ---
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
                
                month = int(s1_name.split('_')[4][4:6])
                active_months.add(month)
                
                ax.add_geometries([Polygon(coords)], crs=ccrs.PlateCarree(), 
                                 facecolor='none', edgecolor=month_colors.get(month, 'black'), 
                                 linewidth=0.8, alpha=0.8, zorder=4)
            except:
                continue

    # --- 3. 辅助组件 ---
    add_north_arrow(ax)
    add_scale_bar(ax, length=500)

    # --- 4. 完善图例 ---
    legend_elements = [
        Line2D([0], [0], color='gray', lw=1.5, alpha=0.6, label='CryoSat-2 Tracks'),
    ]
    
    # 按照月份顺序添加 S1 图例
    for m in sorted(list(active_months)):
        legend_elements.append(
            mpatches.Patch(color=month_colors[m], label=f'S1 {month_names[m]}')
        )

    ax.legend(handles=legend_elements, loc='lower right', 
              title="Data Layers & Months", fontsize='small', ncol=2, 
              frameon=True, facecolor='white', framealpha=0.9, edgecolor='gray')

    # --- 5. 标题设置 ---
    # 使用 pad 增加标题与地图的间距
    plt.title("Spatial Distribution of 594 Matched Pairs (NWP) 2015-2023", 
              fontsize=16, fontweight='bold', pad=30)

    print("绘制完成！")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_nwp_distribution()