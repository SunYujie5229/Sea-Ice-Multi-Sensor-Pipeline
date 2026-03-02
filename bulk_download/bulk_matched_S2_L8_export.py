import pandas as pd
import ee
import geemap
import time
from datetime import datetime

# 初始化 Earth Engine
ee.Initialize()

def export_optical_images():
    """
    从CSV文件读取匹配的影像对，筛选并导出符合条件的光学影像到Google Drive
    """
    
    # 1. 读取CSV文件并筛选时间差异<7小时的pairs
    csv_path = r"F:\NWP\S1_S2_matched\filtered_best_matches.csv"
    
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取CSV文件，共有 {len(df)} 条记录")
        
        # 筛选时间差异小于7小时的记录
        filtered_df = df[df['time_difference_hours'] < 7]
        print(f"筛选后剩余 {len(filtered_df)} 条记录（时间差异<7小时）")
        
        # 进一步筛选状态为success的记录
        success_df = filtered_df[filtered_df['status'] == 'success']
        print(f"状态为success的记录：{len(success_df)} 条")
        
    except Exception as e:
        print(f"读取CSV文件时出错：{e}")
        return
    
    # 2. 处理每个匹配的影像ID
    export_count = 0
    failed_exports = []
    
    for index, row in success_df.iterrows():
        matched_image_id = row['matched_image_id']
        matched_sensor = row['matched_sensor']
        time_diff = row['time_difference_hours']
        
        print(f"\n处理第 {index+1}/{len(success_df)} 个影像：{matched_image_id}")
        print(f"传感器类型：{matched_sensor}，时间差异：{time_diff:.2f}小时")
        
        try:
            # 3. 根据传感器类型获取影像并应用云量过滤
            if 'S2' in matched_image_id or 'Sentinel-2' in matched_sensor:
                # Sentinel-2 影像处理
                image = ee.Image(matched_image_id)
                
                # 获取云量信息（Sentinel-2使用CLOUDY_PIXEL_PERCENTAGE）
                cloud_cover = image.get('CLOUDY_PIXEL_PERCENTAGE')
                
                # 创建一个虚拟的ImageCollection来应用filter
                collection = ee.ImageCollection([image])
                filtered_collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                
                if filtered_collection.size().getInfo() == 0:
                    print(f"跳过：云量>=30% 的Sentinel-2影像")
                    continue
                    
                # 选择可见光和近红外波段进行导出
                bands = ['B2', 'B3', 'B4', 'B8']  # Blue, Green, Red, NIR
                image_export = image.select(bands)
                
            elif 'LC08' in matched_image_id or 'Landsat' in matched_sensor:
                # Landsat-8 影像处理
                image = ee.Image(matched_image_id)
                
                # 获取云量信息（Landsat使用CLOUD_COVER）
                cloud_cover = image.get('CLOUD_COVER')
                
                # 创建一个虚拟的ImageCollection来应用filter
                collection = ee.ImageCollection([image])
                filtered_collection = collection.filter(ee.Filter.lt('CLOUD_COVER', 30))
                
                if filtered_collection.size().getInfo() == 0:
                    print(f"跳过：云量>=30% 的Landsat-8影像")
                    continue
                    
                # 选择可见光和近红外波段进行导出
                bands = ['B2', 'B3', 'B4', 'B5']  # Blue, Green, Red, NIR
                image_export = image.select(bands)
                
            else:
                print(f"未知的传感器类型，跳过：{matched_image_id}")
                continue
            
            # 4. 设置导出参数
            # 从影像ID中提取有用信息作为文件名
            image_id_parts = matched_image_id.split('/')[-1]  # 获取最后一部分作为文件名
            export_name = f"optical_{matched_sensor.replace(' ', '_').replace('&', 'and')}_{image_id_parts}"
            
            # 限制文件名长度（Google Drive有限制）
            if len(export_name) > 100:
                export_name = export_name[:100]
            
            # 获取影像的几何边界
            geometry = image.geometry()
            
            print(f"开始导出：{export_name}")
            
            # 导出到Google Drive
            task = ee.batch.Export.image.toDrive(**{
                'image': image_export,
                'description': export_name,
                'folder': '',  # 空字符串表示根目录
                'scale': 30,  # 30米分辨率
                'region': geometry,
                'maxPixels': 1e9,  # 增加最大像素数限制
                'crs': 'EPSG:4326'
            })
            
            task.start()
            export_count += 1
            print(f"导出任务已启动：{export_name}")
            
            # 避免过快提交任务，稍微延迟
            time.sleep(2)
            
        except Exception as e:
            error_msg = f"处理影像 {matched_image_id} 时出错：{e}"
            print(error_msg)
            failed_exports.append((matched_image_id, str(e)))
            continue
    
    # 5. 输出结果统计
    print(f"\n=== 导出任务统计 ===")
    print(f"成功启动导出任务数量：{export_count}")
    print(f"失败的导出数量：{len(failed_exports)}")
    
    if failed_exports:
        print(f"\n失败的导出详情：")
        for img_id, error in failed_exports:
            print(f"  - {img_id}: {error}")
    
    print(f"\n请在Google Earth Engine Tasks页面查看导出进度：")
    print(f"https://code.earthengine.google.com/tasks")
    
    # 6. 可选：监控任务状态
    print(f"\n正在检查前几个任务状态...")
    tasks = ee.batch.Task.list()
    recent_tasks = tasks[:min(5, export_count)]  # 检查最近的5个任务
    
    for task in recent_tasks:
        print(f"任务 {task.config['description']}: {task.state}")

def check_authentication():
    """检查Earth Engine认证状态"""
    try:
        ee.Initialize()
        print("Earth Engine 认证成功")
        return True
    except Exception as e:
        print(f"Earth Engine 认证失败：{e}")
        print("请运行 ee.Authenticate() 进行认证")
        return False

if __name__ == "__main__":
    print("开始批量导出光学影像...")
    
    # 检查认证状态
    if check_authentication():
        export_optical_images()
    else:
        print("请先完成Earth Engine认证后再运行脚本")
        
    print("脚本执行完成！")