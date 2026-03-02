import time
import re
import ee
import geemap

# ========= 1) 配置 =========
CSV_ASSET_ID = 'projects/vaulted-hawk-469808-i5/assets/time_match_2019'  # <-- 替换为你的CSV表格资产ID
# DRIVE_FOLDER = 'Project_Yujie'                    # 目标Google Drive文件夹
SCALE_M = 40                                      # EW建议 ~40 m
MULTIPLIER = 100                                  # float*100 -> Int16
MAX_EXPORTS = None                                # 限制最多导出多少景；None表示全部
# —— 限流与等待（仅用sleep，不访问Drive）——
SLEEP_AFTER_FIRST = 8.0    # 首个任务进入RUNNING后再等待的秒数（给文件夹创建与可见性传播留缓冲）
SLEEP_BETWEEN     = 1.0    # 其余任务之间的间隔（秒）


# ========= 2) 初始化 =========
ee.Initialize()

# ========= 3) 读取CSV并筛选 =========
table = ee.FeatureCollection(CSV_ASSET_ID)

# 要求：num_matched_points > 40
table = table.filter(ee.Filter.gt('num_matched_points', 50))

# 提取sceneName列表
scene_names = ee.List(table.aggregate_array('sceneName'))

print('CSV rows after filter:', scene_names.size().getInfo())

# ========= 4) 过滤Sentinel-1影像 =========
s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
      .filter(ee.Filter.eq('instrumentMode', 'EW'))
      .filter(ee.Filter.eq('resolution_meters', 40))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'HH'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'HV'))
      .filter(ee.Filter.inList('system:index', scene_names))
     )

print('Matched EW HH/HV scenes =', s1.size().getInfo())

# ========= 5) 波段选择与转 Int16 =========
def to_int16_scaled(img):
    # 个别影像可能缺angle，这里做个兜底：若缺angle则创建一个空的angle以免报错（也可选择跳过）
    band_names = img.bandNames()
    has_angle = band_names.contains('angle')
    def select_three():
        return img.select(['HH', 'HV', 'angle'])
    def add_angle_nan():
        # 新建一个与HH同尺寸的全NaN角度波段（或使用常数角度），这里用NaN
        ref = img.select('HH')
        angle_nan = ref.multiply(ee.Image(0)).updateMask(ref.mask()).rename('angle')
        return img.addBands(angle_nan, overwrite=True).select(['HH', 'HV', 'angle'])
    sel = ee.Image(ee.Algorithms.If(has_angle, select_three(), add_angle_nan()))
    scaled = sel.multiply(MULTIPLIER).round().toInt16()
    return scaled.copyProperties(img, img.propertyNames())

s1_int16 = s1.map(to_int16_scaled)

# ========= 6) 命名清洗，避免产生多层文件夹 =========
def sanitize_name(name):
    # 去掉可能被Drive当作路径或非法的字符（/、\、:、?、* 等）
    # 同时避免过长：Drive限制不严，但GEE任务名过长不友好
    name = re.sub(r'[\\/:\*\?"<>\|]+', '_', name)
    return name[:180]  # 截断以安全

# ========= 7) 创建导出任务到同一个Drive文件夹 =========
img_list = s1_int16.toList(s1_int16.size())
count = s1_int16.size().getInfo()
if MAX_EXPORTS is None:
    MAX_EXPORTS = count

print('Scheduling exports =', min(count, MAX_EXPORTS))

# ========== 7) 仅用sleep的“串行+首任务等待”提交器 ==========
def poll_until_running_or_done(task, timeout_sec=180, interval=2.0):
    """轮询任务状态直到 RUNNING/COMPLETED/FAILED/CANCELLED 或超时，返回状态字符串。"""
    waited = 0.0
    while waited < timeout_sec:
        st = task.status().get('state')
        if st in ('RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'):
            return st
        time.sleep(interval)
        waited += interval
    return task.status().get('state')

img_list = s1_int16.toList(s1_int16.size())
count = s1_int16.size().getInfo()
if MAX_EXPORTS is None:
    MAX_EXPORTS = count

n_submit = min(count, MAX_EXPORTS)
print('Scheduling exports =', n_submit)

if n_submit == 0:
    print('没有满足条件的影像可导出。请检查 CSV 的 sceneName 与筛选条件。')
else:
    for i in range(n_submit):
        img = ee.Image(img_list.get(i))
        scene_id = img.get('system:index').getInfo()
        out_name = f"{sanitize_name(scene_id)}_EW_HH_HV_angle_int16x{MULTIPLIER}"

        task = ee.batch.Export.image.toDrive(
            image=img,
            description=out_name,          # 任务名
            # folder=DRIVE_FOLDER,           # 统一目标文件夹（只按名字）
            fileNamePrefix=out_name,       # 不含 “/”
            region=img.geometry(),         # 使用产品覆盖范围
            scale=SCALE_M,
            maxPixels=1e13,
            fileFormat='GeoTIFF',
            formatOptions={'cloudOptimized': True}
        )
        task.start()

        if i == 0:
            # 等到首个任务至少进入RUNNING（或完成/失败/被取消）
            state = poll_until_running_or_done(task, timeout_sec=180, interval=2.0)
            print(f'[info] First task state = {state}. Sleeping {SLEEP_AFTER_FIRST}s to stabilize folder visibility...')
            # 再额外等待一段时间，降低“找不到文件夹→各自创建”的并发概率
            time.sleep(SLEEP_AFTER_FIRST)
        else:
            # 其余任务之间加短暂停顿，避免并发创建竞态
            time.sleep(SLEEP_BETWEEN)

    print('All export tasks submitted. Monitor their progress in your EE Tasks.')