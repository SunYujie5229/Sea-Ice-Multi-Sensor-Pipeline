import ee
import geemap
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SatelliteImageMatcher:
    """卫星影像匹配类"""
    
    def __init__(self, time_window_hours=2, cloud_threshold=20):
        """
        初始化匹配器
        
        Args:
            time_window_hours: 时间窗口（小时）
            cloud_threshold: 云量阈值（百分比）
        """
        # 初始化Earth Engine
        try:
            ee.Initialize()
            print("✓ Earth Engine 初始化成功")
        except Exception as e:
            print(f"✗ Earth Engine 初始化失败: {e}")
            raise
        
        self.time_window_hours = time_window_hours
        self.cloud_threshold = cloud_threshold
        self.results = []
        
    def load_csv_asset(self, asset_path, scene_name_column='sceneName'):
        """
        加载CSV资产并提取场景名称
        
        Args:
            asset_path: GEE中CSV资产的路径
            scene_name_column: 场景名称列的名称
            
        Returns:
            list: 场景名称列表
        """
        try:
            # 加载CSV作为FeatureCollection
            csv_fc = ee.FeatureCollection(asset_path)
            
            # 获取场景名称列表
            scene_names = csv_fc.aggregate_array(scene_name_column).getInfo()
            
            print(f"✓ 成功加载CSV资产: {len(scene_names)} 个场景名称")
            return scene_names
            
        except Exception as e:
            logger.error(f"加载CSV资产失败: {e}")
            raise
    
    def load_csv_file(self, csv_file_path, scene_name_column='sceneName'):
        """
        从本地CSV文件加载场景名称，自动筛选EW模式
        
        Args:
            csv_file_path: 本地CSV文件路径
            scene_name_column: 场景名称列的名称
            
        Returns:
            list: EW模式的场景名称列表
        """
        try:
            df = pd.read_csv(csv_file_path)
            
            # 检查列是否存在
            if scene_name_column not in df.columns:
                raise ValueError(f"CSV中未找到列：{scene_name_column}")
            
            print("筛选EW模式的S1数据...")
            # 筛选EW模式的S1数据
            ew_scenes = df[df[scene_name_column].apply(self.is_ew_mode)][scene_name_column].dropna().astype(str).drop_duplicates().tolist()
            
            print(f"✓ 成功加载本地CSV文件: 总共 {len(df)} 条记录")
            print(f"✓ 筛选出EW模式场景: {len(ew_scenes)} 个")
            return ew_scenes
            
        except Exception as e:
            logger.error(f"加载CSV文件失败: {e}")
            raise
    
    def is_ew_mode(self, scene_name):
        """
        判断场景名称是否为EW模式
        
        Args:
            scene_name: 场景名称
            
        Returns:
            bool: 是否为EW模式
        """
        if pd.isna(scene_name):
            return False
        
        scene_str = str(scene_name)
        # Sentinel-1场景名称格式检查EW模式
        # 格式: S1A_EW_GRDM_1SDH_20230101T120000_20230101T120025_046123_058456_1234
        if '_EW_' in scene_str:
            return True
        return False
    
    def find_sentinel1_image(self, scene_name, roi=None):
        """
        查找Sentinel-1影像（已知为EW模式）
        
        Args:
            scene_name: 场景名称（已筛选为EW模式）
            roi: 感兴趣区域
            
        Returns:
            dict: Sentinel-1影像信息
        """
        try:
            # 构建Sentinel-1集合，直接查找EW模式
            s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filter(ee.Filter.eq('instrumentMode', 'EW'))
            
            # 尝试不同的筛选方式
            filters = [
                ee.Filter.eq('system:index', scene_name),
                ee.Filter.stringContains('system:id', scene_name),
                ee.Filter.stringContains('PRODUCT_ID', scene_name)
            ]
            
            s1_image = None
            for filter_condition in filters:
                temp_collection = s1_collection.filter(filter_condition)
                
                if roi:
                    temp_collection = temp_collection.filterBounds(roi)
                
                if temp_collection.size().getInfo() > 0:
                    s1_image = temp_collection.first()
                    break
            
            if s1_image is None:
                return {'error': f'未找到场景名称为 {scene_name} 的Sentinel-1影像'}
            
            # 获取影像信息
            image_info = s1_image.getInfo()
            acquisition_time = ee.Date(s1_image.get('system:time_start'))
            
            return {
                'image': s1_image,
                'image_id': image_info['id'],
                'acquisition_time': acquisition_time,
                'acquisition_time_str': acquisition_time.format('YYYY-MM-dd HH:mm:ss').getInfo(),
                'properties': image_info.get('properties', {}),
                'geometry': s1_image.geometry()
            }
            
        except Exception as e:
            return {'error': f'处理场景 {scene_name} 时出错: {str(e)}'}
    
    def find_matching_images(self, s1_info, sensor_type='sentinel2'):
        """
        查找时间窗口内的匹配影像
        
        Args:
            s1_info: Sentinel-1影像信息
            sensor_type: 'sentinel2' 或 'landsat8'
            
        Returns:
            list: 匹配的影像列表
        """
        if 'error' in s1_info:
            return []
        
        try:
            acquisition_time = s1_info['acquisition_time']
            geometry = s1_info['geometry']
            
            # 计算时间窗口
            time_start = acquisition_time.advance(-self.time_window_hours, 'hour')
            time_end = acquisition_time.advance(self.time_window_hours, 'hour')
            
            if sensor_type == 'sentinel2':
                collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                    .filterDate(time_start, time_end) \
                    .filterBounds(geometry) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', self.cloud_threshold))
                    
            elif sensor_type == 'landsat8':
                collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                    .filterDate(time_start, time_end) \
                    .filterBounds(geometry) \
                    .filter(ee.Filter.lt('CLOUD_COVER', self.cloud_threshold))
            else:
                return []
            
            collection_size = collection.size().getInfo()
            
            if collection_size == 0:
                return []
            
            # 限制返回数量以避免内存问题
            max_images = min(collection_size, 10)
            image_list = collection.limit(max_images).getInfo()
            
            matched_images = []
            for img_info in image_list['features']:
                img_time = ee.Date(img_info['properties']['system:time_start'])
                time_diff = img_time.difference(acquisition_time, 'hour').getInfo()
                
                matched_images.append({
                    'image_id': img_info['id'],
                    'acquisition_time': img_time.format('YYYY-MM-dd HH:mm:ss').getInfo(),
                    'time_difference_hours': round(time_diff, 2),
                    'properties': img_info['properties'],
                    'sensor_type': sensor_type
                })
            
            return matched_images
            
        except Exception as e:
            logger.error(f"查找匹配影像时出错: {e}")
            return []
    
    def process_single_scene(self, scene_name, roi=None):
        """
        处理单个场景
        
        Args:
            scene_name: 场景名称
            roi: 感兴趣区域
            
        Returns:
            list: 处理结果列表
        """
        scene_results = []
        
        # 查找Sentinel-1影像
        s1_info = self.find_sentinel1_image(scene_name, roi)
        
        if 'error' in s1_info:
            scene_results.append({
                's1_scene_name': scene_name,
                'error': s1_info['error'],
                'status': 'failed'
            })
            return scene_results
        
        # 查找Sentinel-2匹配
        s2_matches = self.find_matching_images(s1_info, 'sentinel2')
        
        # 查找Landsat-8匹配
        l8_matches = self.find_matching_images(s1_info, 'landsat8')
        
        # 记录Sentinel-2匹配结果
        if s2_matches:
            for s2_match in s2_matches:
                scene_results.append({
                    's1_scene_name': scene_name,
                    's1_image_id': s1_info['image_id'],
                    's1_acquisition_time': s1_info['acquisition_time_str'],
                    'matched_image_id': s2_match['image_id'],
                    'matched_acquisition_time': s2_match['acquisition_time'],
                    'time_difference_hours': s2_match['time_difference_hours'],
                    'sensor_pair': 'Sentinel-1 & Sentinel-2',
                    'matched_sensor': 'Sentinel-2',
                    'status': 'success'
                })
        
        # 记录Landsat-8匹配结果
        if l8_matches:
            for l8_match in l8_matches:
                scene_results.append({
                    's1_scene_name': scene_name,
                    's1_image_id': s1_info['image_id'],
                    's1_acquisition_time': s1_info['acquisition_time_str'],
                    'matched_image_id': l8_match['image_id'],
                    'matched_acquisition_time': l8_match['acquisition_time'],
                    'time_difference_hours': l8_match['time_difference_hours'],
                    'sensor_pair': 'Sentinel-1 & Landsat-8',
                    'matched_sensor': 'Landsat-8',
                    'status': 'success'
                })
        
        # 如果没有找到任何匹配
        if not s2_matches and not l8_matches:
            scene_results.append({
                's1_scene_name': scene_name,
                's1_image_id': s1_info['image_id'],
                's1_acquisition_time': s1_info['acquisition_time_str'],
                'matched_image_id': 'No match found',
                'sensor_pair': 'Sentinel-1 only',
                'status': 'no_match'
            })
        
        return scene_results
    
    def process_batch(self, scene_names, roi=None, batch_size=50, delay=0.5):
        """
        批量处理场景名称
        
        Args:
            scene_names: 场景名称列表
            roi: 感兴趣区域
            batch_size: 批处理大小
            delay: 批次间延迟（秒）
            
        Returns:
            list: 所有处理结果
        """
        all_results = []
        total_scenes = len(scene_names)
        
        print(f"开始批量处理 {total_scenes} 个场景...")
        
        # 分批处理
        for i in tqdm(range(0, total_scenes, batch_size), desc="批次处理进度"):
            batch_scenes = scene_names[i:i+batch_size]
            batch_results = []
            
            # 处理当前批次
            for scene_name in tqdm(batch_scenes, desc=f"批次 {i//batch_size + 1}", leave=False):
                try:
                    scene_results = self.process_single_scene(scene_name, roi)
                    batch_results.extend(scene_results)
                    
                    # 小延迟避免API限制
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"处理场景 {scene_name} 时出错: {e}")
                    batch_results.append({
                        's1_scene_name': scene_name,
                        'error': str(e),
                        'status': 'failed'
                    })
            
            all_results.extend(batch_results)
            
            # 批次间延迟
            if i + batch_size < total_scenes:
                time.sleep(delay)
            
            # 定期输出进度
            if (i // batch_size + 1) % 10 == 0:
                print(f"已完成 {len(all_results)} 个结果")
        
        self.results = all_results
        return all_results
    
    def save_results(self, output_path='satellite_matching_results.csv'):
        """
        保存结果到CSV文件
        
        Args:
            output_path: 输出文件路径
        """
        if not self.results:
            print("没有结果可保存")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✓ 结果已保存到: {output_path}")
        
        # 输出统计信息
        self.print_statistics()
    
    def print_statistics(self):
        """输出统计信息"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n=== 处理统计 ===")
        print(f"总处理数量: {len(df)}")
        
        status_counts = df['status'].value_counts()
        for status, count in status_counts.items():
            print(f"{status}: {count}")
        
        if 'sensor_pair' in df.columns:
            print("\n=== 传感器配对统计 ===")
            pair_counts = df['sensor_pair'].value_counts()
            for pair, count in pair_counts.items():
                print(f"{pair}: {count}")

# 使用示例
def main():
    """主函数示例"""
    
    # 创建匹配器实例
    matcher = SatelliteImageMatcher(time_window_hours=12, cloud_threshold=20)
    
    # 方法1: 从GEE资产加载CSV
    # scene_names = matcher.load_csv_asset('users/your_username/your_csv_asset')
    CSV_file = r"E:\NWP\CS2_S1_matched\time_match_2023.csv"
    # 方法2: 从本地CSV文件加载
    scene_names = matcher.load_csv_file(CSV_file)
    
    # 方法3: 手动提供场景名称列表（用于测试）
    # scene_names = [
    #     'S1A_EW_GRDM_1SDH_20230101T120000_20230101T120025_046123_058456_1234',
    #     'S1B_EW_GRDM_1SDH_20230102T120000_20230102T120025_030123_036789_5678',
    #     # 添加更多场景名称...
    # ]
    
    # 定义研究区域（可选）
    # roi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    roi = None  # 不限制区域
    
    # 批量处理
    results = matcher.process_batch(
        scene_names=scene_names,
        roi=roi,
        batch_size=20,  # 每批处理20个
        delay=1.0       # 批次间延迟1秒
    )
    
    matched_file = r"F:\NWP\S1_S2_matched\matched_satellite_images.csv"
    
    # 保存结果
    matcher.save_results(matched_file)
    
    print("处理完成！")

if __name__ == "__main__":
    # 运行示例
    main()