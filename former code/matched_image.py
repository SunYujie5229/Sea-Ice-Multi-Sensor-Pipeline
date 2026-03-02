# matched_image.py
import ee
import pandas as pd
from datetime import timedelta
import time

def ee_init():
    try:
        ee.Initialize()
        print("✓ Earth Engine 初始化成功")
    except Exception as e:
        raise RuntimeError(f"Earth Engine 初始化失败: {e}")

def _is_ew(scene_name: str) -> bool:
    if not isinstance(scene_name, str):
        return False
    return "_EW_" in scene_name

def _find_s1_by_scene(scene_name: str, roi=None):
    """返回 ee.Image 或 None"""
    s1_col = ee.ImageCollection('COPERNICUS/S1_GRD').filter(
        ee.Filter.eq('instrumentMode', 'EW')
    )
    # 三种兜底查找方式
    candidates = [
        s1_col.filter(ee.Filter.eq('system:index', scene_name)),
        s1_col.filter(ee.Filter.stringContains('system:id', scene_name)),
        s1_col.filter(ee.Filter.stringContains('PRODUCT_ID', scene_name)),
    ]
    for col in candidates:
        if roi is not None:
            col = col.filterBounds(roi)
        if col.size().getInfo() > 0:
            return col.first()
    return None

def _find_optical_matches(s1_img, hours: int, cloud_thr: int, sensor: str):
    """返回列表，每项是 dict（image_id, acquisition_time, time_difference_hours, sensor_type）"""
    acq = ee.Date(s1_img.get('system:time_start'))
    geom = s1_img.geometry()
    t0 = acq.advance(-hours, 'hour')
    t1 = acq.advance(+hours, 'hour')

    if sensor.lower() in ["s2", "sentinel2", "sentinel-2"]:
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterDate(t0, t1)
               .filterBounds(geom)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_thr)))
        tag = 'Sentinel-2'
    elif sensor.lower() in ["l8", "landsat8", "landsat-8"]:
        col = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
               .filterDate(t0, t1)
               .filterBounds(geom)
               .filter(ee.Filter.lt('CLOUD_COVER', cloud_thr)))
        tag = 'Landsat-8'
    else:
        return []

    size = col.size().getInfo()
    if size == 0:
        return []

    # 限制最多取 10 张，避免 getInfo 压力
    feats = col.limit(min(size, 10)).getInfo()['features']
    out = []
    for f in feats:
        img_time = ee.Date(f['properties']['system:time_start'])
        dt_h = img_time.difference(acq, 'hour').getInfo()
        out.append({
            "image_id": f['id'],
            "acquisition_time": img_time.format('YYYY-MM-dd HH:mm:ss').getInfo(),
            "time_difference_hours": round(dt_h, 2),
            "sensor_type": tag
        })
    return out

def find_matched_images(
    csv_path: str,
    scene_col: str = "sceneName",
    time_window_hours: int = 12,
    cloud_threshold: int = 20,
    roi=None,
    batch_size: int = 20,
    batch_delay_sec: float = 1.0,
    per_item_delay_sec: float = 0.1,
    save_csv_path: str | None = None
) -> pd.DataFrame:
    """
    从本地 CSV 读取 S1 场景名（自动筛选 EW），对每个 S1 在时间窗口内找 S2 / L8 匹配。
    返回 DataFrame；若指定 save_csv_path 则同时保存。
    """
    ee_init()
    df = pd.read_csv(csv_path)
    if scene_col not in df.columns:
        raise ValueError(f"CSV中未找到列：{scene_col}")

    scenes = (df[scene_col]
              .dropna().astype(str).drop_duplicates()
              .tolist())
    ew_scenes = [s for s in scenes if _is_ew(s)]
    print(f"共 {len(scenes)} 条场景；EW 模式 {len(ew_scenes)} 条。")

    rows = []
    total = len(ew_scenes)
    for i in range(0, total, batch_size):
        batch = ew_scenes[i:i+batch_size]
        for scene in batch:
            try:
                s1 = _find_s1_by_scene(scene, roi)
                if s1 is None:
                    rows.append({
                        "s1_scene_name": scene,
                        "status": "failed",
                        "error": "未找到对应的 S1 EW 影像"
                    })
                    continue

                # 基本属性
                s1_id = s1.getInfo()['id']
                s1_time = ee.Date(s1.get('system:time_start')).format(
                    'YYYY-MM-dd HH:mm:ss'
                ).getInfo()

                s2_matches = _find_optical_matches(s1, time_window_hours, cloud_threshold, "s2")
                l8_matches = _find_optical_matches(s1, time_window_hours, cloud_threshold, "l8")

                if s2_matches:
                    for m in s2_matches:
                        rows.append({
                            "s1_scene_name": scene,
                            "s1_image_id": s1_id,
                            "s1_acquisition_time": s1_time,
                            "matched_image_id": m["image_id"],
                            "matched_acquisition_time": m["acquisition_time"],
                            "time_difference_hours": m["time_difference_hours"],
                            "matched_sensor": m["sensor_type"],
                            "sensor_pair": "Sentinel-1 & Sentinel-2",
                            "status": "success"
                        })

                if l8_matches:
                    for m in l8_matches:
                        rows.append({
                            "s1_scene_name": scene,
                            "s1_image_id": s1_id,
                            "s1_acquisition_time": s1_time,
                            "matched_image_id": m["image_id"],
                            "matched_acquisition_time": m["acquisition_time"],
                            "time_difference_hours": m["time_difference_hours"],
                            "matched_sensor": m["sensor_type"],
                            "sensor_pair": "Sentinel-1 & Landsat-8",
                            "status": "success"
                        })

                if not s2_matches and not l8_matches:
                    rows.append({
                        "s1_scene_name": scene,
                        "s1_image_id": s1_id,
                        "s1_acquisition_time": s1_time,
                        "matched_image_id": None,
                        "matched_acquisition_time": None,
                        "time_difference_hours": None,
                        "matched_sensor": None,
                        "sensor_pair": "Sentinel-1 only",
                        "status": "no_match"
                    })

            except Exception as e:
                rows.append({
                    "s1_scene_name": scene,
                    "status": "failed",
                    "error": str(e)
                })
            finally:
                time.sleep(per_item_delay_sec)

        # 批次间歇
        if i + batch_size < total:
            time.sleep(batch_delay_sec)

    out_df = pd.DataFrame(rows)
    if save_csv_path:
        out_df.to_csv(save_csv_path, index=False, encoding="utf-8")
        print(f"✓ 结果已保存: {save_csv_path}")

    return out_df
