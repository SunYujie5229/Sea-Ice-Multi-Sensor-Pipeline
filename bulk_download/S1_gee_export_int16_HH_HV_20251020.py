# =============================================
# Sentinel-1 EW HH/HV 导出脚本（去除背景值，nodata=-9999）
# =============================================
import time
import re
import ee
import geemap

# ========= 1) 配置 =========
CSV_ASSET_ID ="projects/calm-drive-469805-i9/assets/time_match_2018" # <-- 替换为你的CSV表格资产ID
SCALE_M = 40                                      # EW 建议 ~40 m
MULTIPLIER = 100                                  # float*100 -> Int16
MAX_EXPORTS = None                                # 限制导出数量；None 表示全部
SLEEP_AFTER_FIRST = 8.0    # 首个任务RUNNING后等待秒数
SLEEP_BETWEEN     = 1.0    # 任务间隔（秒）

# ========= 2) 初始化 =========
ee.Initialize()

# ========= 3) 读取 CSV 并筛选 =========
table = ee.FeatureCollection(CSV_ASSET_ID)
table = table.filter(ee.Filter.gt('num_matched_points', 50))
scene_names = ee.List(table.aggregate_array('sceneName'))
print('CSV rows after filter:', scene_names.size().getInfo())

# ========= 4) 过滤 Sentinel-1 影像 =========
s1 = (
    ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.eq('instrumentMode', 'EW'))
    .filter(ee.Filter.eq('resolution_meters', 40))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'HH'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'HV'))
    .filter(ee.Filter.inList('system:index', scene_names))
)
print('Matched EW HH/HV scenes =', s1.size().getInfo())

# ========= 5) 波段选择、空值处理与转 Int16 =========
def to_int16_scaled(img):
    # 只选择 HH / HV 波段
    sel = img.select(['HH', 'HV'])
    
    # 去除背景值（0值不参与）
    sel = sel.updateMask(sel.neq(0))
    
    # 转 Int16 并缩放
    scaled = sel.multiply(MULTIPLIER).round().toInt16()
    
    # 导出前指定 nodata=-9999
    scaled = scaled.unmask(-9999)
    scaled = scaled.setDefaultProjection(sel.projection()).set('nodata', -9999)
    
    return scaled.copyProperties(img, img.propertyNames())

s1_int16 = s1.map(to_int16_scaled)

# ========= 6) 命名清洗 =========
def sanitize_name(name):
    name = re.sub(r'[\\/:\*\?"<>\|]+', '_', name)
    return name[:180]

# ========= 7) 提交导出任务 =========
img_list = s1_int16.toList(s1_int16.size())
count = s1_int16.size().getInfo()
if MAX_EXPORTS is None:
    MAX_EXPORTS = count
n_submit = min(count, MAX_EXPORTS)

print('Scheduling exports =', n_submit)

def poll_until_running_or_done(task, timeout_sec=180, interval=2.0):
    """轮询任务状态直到 RUNNING/COMPLETED/FAILED/CANCELLED 或超时。"""
    waited = 0.0
    while waited < timeout_sec:
        st = task.status().get('state')
        if st in ('RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'):
            return st
        time.sleep(interval)
        waited += interval
    return task.status().get('state')

if n_submit == 0:
    print('没有满足条件的影像可导出。请检查 CSV 的 sceneName 与筛选条件。')
else:
    for i in range(n_submit):
        img = ee.Image(img_list.get(i))
        scene_id = img.get('system:index').getInfo()
        out_name = f"{sanitize_name(scene_id)}_EW_HH_HV_int16x{MULTIPLIER}_nodata-9999"

        # ✅ 导出时声明 crs 与 noData 值
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=out_name,
            fileNamePrefix=out_name,
            region=img.geometry(),
            scale=SCALE_M,
            crs='EPSG:4326',  # ✅ 坐标系声明
            maxPixels=1e13,
            fileFormat='GeoTIFF',
            formatOptions={
                'noData': -9999  # ✅ 导出时指定 nodata 值
            }
        )
        task.start()

        if i == 0:
            state = poll_until_running_or_done(task, timeout_sec=180, interval=2.0)
            print(f'[info] First task state = {state}. Sleeping {SLEEP_AFTER_FIRST}s...')
            time.sleep(SLEEP_AFTER_FIRST)
        else:
            time.sleep(SLEEP_BETWEEN)

    print('✅ All export tasks submitted. Monitor progress in EE Tasks.')
