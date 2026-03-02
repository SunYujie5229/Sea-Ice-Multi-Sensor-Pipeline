# visualize_matched.py
import ee
import geemap

def ee_init():
    try:
        ee.Initialize()
        print("✓ Earth Engine 初始化成功")
    except Exception as e:
        raise RuntimeError(f"Earth Engine 初始化失败: {e}")

def _load_s1(image_id: str):
    return ee.Image(image_id)

def _load_optical(image_id: str, sensor: str):
    if "COPERNICUS/S2" in image_id or sensor.lower().startswith("sentinel-2"):
        return ee.Image(image_id).select(["B4","B3","B2"])  # True color
    # Landsat-8 L2: 需要把反射率缩放
    if "LANDSAT/LC08" in image_id or sensor.lower().startswith("landsat-8"):
        img = ee.Image(image_id)
        # 参考 USGS L2 系列缩放因子：反射率= SR_Bx * 0.0000275 - 0.2
        rgb = img.select(["SR_B4","SR_B3","SR_B2"]).multiply(0.0000275).add(-0.2)
        return rgb
    # 默认直接返回
    return ee.Image(image_id)

def visualize_pair(
    s1_image_id: str,
    matched_image_id: str,
    matched_sensor: str,
    center_on="s1",
    map_height=800,
    map_width=800
):
    """
    在 geemap 里可视化一对 S1 与匹配到的光学影像。
    center_on: "s1" | "optical"
    """
    ee_init()
    m = geemap.Map(height=map_height, width=map_width)

    s1 = _load_s1(s1_image_id)
    # S1 可视化：VV 或 HH；EW 常见 HH/HV，此处示例取强度影像
    s1_viz = {"min": -20, "max": 0}
    # 取存在的后向散射通道（顺序兜底）
    bands_try = ["VV","HH","VH","HV"]
    s1_bands = [b for b in bands_try if b in s1.bandNames().getInfo()]
    if s1_bands:
        m.addLayer(s1.select(s1_bands[0]).log10().multiply(10), s1_viz, f"S1 ({s1_bands[0]})")
    else:
        m.addLayer(s1, {}, "S1 (raw)")

    opt = _load_optical(matched_image_id, matched_sensor)
    opt_viz = {"min": 0.0, "max": 0.3}  # S2/L8 反射率可用此范围快速预览
    m.addLayer(opt, opt_viz, f"{matched_sensor} RGB")

    # 视图居中
    target = s1 if center_on == "s1" else opt
    geom = target.geometry()
    m.centerObject(geom, 7)  # 缩放等级可按需调整

    return m
