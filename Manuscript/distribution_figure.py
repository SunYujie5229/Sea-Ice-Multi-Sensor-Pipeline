import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon, LineString
import os

# --- 路径配置 ---
# 1. 之前步骤生成的匹配对汇总表
MATCHED_PAIRS_CSV = r'C:\Users\TJ002\Desktop\CS2_S1_result\overlap\matched_pairs_final.csv'
# 2. 包含四角坐标的元数据汇总表
S1_METADATA_CSV = r'F:\NWP\Sentinel1 metadata\Metadata\filtered_GRD_results.csv'
# 3. CryoSat-2 L1b 数据根目录
CS2_ROOT = r'F:\NWP\CS2_L1'

# --- 颜色映射：不同月份对应不同颜色 ---
month_colors = {
    1: '#1f77b4', 2: '#aec7e8', 3: '#ff7f0e', 4: '#ffbb78',
    5: '#2ca02c', 6: '#98df8a', 7: '#d62728', 8: '#ff9896',
    9: '#9467bd', 10: '#c5b0d5', 11: '#8c564b', 12: '#c49c94'
}

def load_data():
    # 读取数据
    pairs_df = pd.read_csv(MATCHED_PAIRS_CSV)
    s1_meta = pd.read_csv(S1_METADATA_CSV)
    
    # 将元数据中的 Granule Name 作为索引方便快速查询
    s1_meta = s1_meta.set_index('Granule Name')
    return pairs_df, s1_meta

def plot_nwp_distribution():
    pairs_df, s1_meta = load_data()
    
    # 创建北极投影绘图
    fig = plt.figure(figsize=(15, 12))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-100))
    ax.set_extent([-140, -50, 65, 85], ccrs.PlateCarree()) # 聚焦西北航道区域
    
    # 添加地理底图
    ax.add_feature(cfeature.OCEAN, facecolor='#f0f0f0')
    ax.add_feature(cfeature.LAND, facecolor='#e0e0e0', edgecolor='black', linewidth=0.5)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    print("开始绘制地理分布图...")

    for idx, row in pairs_df.iterrows():
        s1_name = row['sceneName']
        cs2_rel_path = row['cs2_path']
        
        # 1. 获取并绘制 Sentinel-1 边界
       # 1. 获取并绘制 Sentinel-1 边界
        if s1_name in s1_meta.index:
            # 关键修改：使用 .loc[s1_name] 后接 .iloc[0] 确保获取单行，避免重复行导致返回 Series 集合
            meta = s1_meta.loc[s1_name]
            if isinstance(meta, pd.DataFrame):
                meta = meta.iloc[0]
            
            # 显式转换为 float，确保 shapely 能够识别
            try:
                coords = [
                    (float(meta['Near Start Lon']), float(meta['Near Start Lat'])),
                    (float(meta['Far Start Lon']), float(meta['Far Start Lat'])),
                    (float(meta['Far End Lon']), float(meta['Far End Lat'])),
                    (float(meta['Near End Lon']), float(meta['Near End Lat'])),
                    (float(meta['Near Start Lon']), float(meta['Near Start Lat'])) # 闭合点
                ]
                poly = Polygon(coords)
                
                # 绘制代码保持不变...
                month = int(s1_name.split('_')[4][4:6])
                color = month_colors.get(month, 'black')
                
                ax.add_geometries([poly], crs=ccrs.PlateCarree(), 
                                 facecolor='none', edgecolor=color, 
                                 linewidth=0.8, alpha=0.7)
            except (ValueError, TypeError) as e:
                print(f"跳过数据 {s1_name}: 坐标格式错误 - {e}")

        # 2. 获取并绘制 CryoSat-2 轨道线
        # 注意：此处需根据你的实际存储结构拼接路径，下为示例
        cs2_full_path = os.path.join(CS2_ROOT, cs2_rel_path.split('NWP\\')[-1]) 
        
        if os.path.exists(cs2_full_path):
            try:
                with xr.open_dataset(cs2_full_path) as ds:
                    lon = ds.lon_20_ku.values
                    lat = ds.lat_20_ku.values
                    # 绘制灰色轨道线
                    ax.plot(lon, lat, color='gray', linewidth=0.4, 
                            alpha=0.4, transform=ccrs.Geodetic())
            except Exception as e:
                pass # 忽略读取错误的NC文件

    plt.title("Spatial Distribution of Matched S1-CS2 Pairs (Northwest Passage)", fontsize=15)
    # 手动创建图例句柄
# --- 在 plt.show() 之前添加以下图例生成代码 ---
    
    from matplotlib.lines import Line2D
    
    # 1. 自动获取数据中实际存在的月份（从 pairs_df 的 sceneNam 列提取）
    # 假设文件名格式为：S1A_EW_GRDM_1SDH_20150107T...
    def extract_month(name):
        try:
            return int(name.split('_')[4][4:6])
        except:
            return None

    existing_months = sorted(pairs_df['sceneName'].apply(extract_month).dropna().unique())

    # 2. 创建图例句柄 (仅包含数据中存在的月份)
    legend_elements = [
        Line2D([0], [0], color=month_colors[m], lw=2, label=f'Month {m:02d}') 
        for m in existing_months if m in month_colors
    ]
    
    # 3. 添加 CryoSat-2 的轨道线图例说明
    legend_elements.append(Line2D([0], [0], color='gray', lw=1, alpha=0.6, label='CryoSat-2 Tracks'))

    # 4. 绘制图例
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower right', title="Legend", 
                  fontsize='small', ncol=2, frameon=True, facecolor='white')

    print(f"检测到数据分布在以下月份: {existing_months}")
    
    # --- 在 plt.show() 之前，确保这部分代码包含了 CS2 的示意线 ---
    from matplotlib.lines import Line2D
    
    # 1. 自动获取数据中实际存在的月份
    def extract_month(name):
        try:
            return int(name.split('_')[4][4:6])
        except:
            return None

    existing_months = sorted(pairs_df['sceneName'].apply(extract_month).dropna().unique())

    # 2. 构建图例句柄列表
    legend_elements = []
    
    # 添加 Sentinel-1 月份颜色说明 (使用方形或线框)
    for m in existing_months:
        if m in month_colors:
            legend_elements.append(Line2D([0], [0], color=month_colors[m], lw=2, 
                                          label=f'S1 Month {m:02d}'))

    # 【关键补充】：添加 CryoSat-2 轨道线的示意线（灰色细线）
    # 使用与绘图时相同的颜色(gray)和透明度(0.5)
    legend_elements.append(Line2D([0], [0], color='gray', lw=1, alpha=0.8, 
                                  linestyle='-', label='CryoSat-2 Tracks'))

    # 3. 绘制图例
    if legend_elements:
        # 将图例放在合适的位置，ncol=2 可以让图例排成两列，节省空间
        ax.legend(handles=legend_elements, loc='lower right', title="Data Layers", 
                  fontsize='small', ncol=2, frameon=True, facecolor='white', framealpha=0.9)

    print(f"图例已更新，包含 {len(existing_months)} 个月份和 1 条 CS2 轨道线示意。")

    
    
    plt.show()
    
if __name__ == "__main__":
    plot_nwp_distribution()