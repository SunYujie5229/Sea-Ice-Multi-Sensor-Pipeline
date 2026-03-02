import pandas as pd
from pathlib import Path

# ===============================
# 1. 读取 CSV
# ===============================
csv_path = r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\overlapping_S1_MultiSensor_all.csv"
df = pd.read_csv(csv_path)

# ===============================
# 2. S1 scene → GEE Image ID
# ===============================
def build_s1_id(scene_name: str) -> str:
    return f"COPERNICUS/S1_GRD/{scene_name}"

# ===============================
# 3. 生成 JS 代码
# ===============================
js_blocks = []

for s1_scene, group in df.groupby("s1_scene"):
    s1_id = build_s1_id(s1_scene)

    # Sentinel-2
    s2_ids = group.loc[
        group["matched_sensor"] == "Sentinel2", "matched_id"
    ].tolist()

    # Landsat (LC08 + LC09 全部放一起)
    l8_ids = group.loc[
        group["matched_sensor"].str.contains("Landsat"), "matched_id"
    ].tolist()

    block = []

    block.append("// S1 Image ID")

    block.append(f"var s1_id = '{s1_id}';\n")


    block.append("// Corresponding Sentinel-2 Images")

    block.append("var s2_id = [")
    for i in s2_ids:
        block.append(f"  '{i}',")
    block.append("];\n")


    block.append("// Corresponding Landsat Images")

    block.append("var l8_id = [")
    for i in l8_ids:
        block.append(f"  '{i}',")
    block.append("];\n\n")

    js_blocks.append("\n".join(block))

# ===============================
# 4. 写出 JS 文件
# ===============================
output_path = Path(r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\pairs_from_csv_all.js")
output_path.write_text("\n".join(js_blocks), encoding="utf-8")

print(f"JS file written to: {output_path.resolve()}")
