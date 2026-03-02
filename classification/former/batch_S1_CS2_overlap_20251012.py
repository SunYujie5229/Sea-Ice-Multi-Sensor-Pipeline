# -*- coding: utf-8 -*-
"""
Batch overlap analysis for all matched CS2–S1 pairs.
CSV columns used: sceneName, cs2_path
"""

import os, re, warnings
import numpy as np, pandas as pd, geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt

# -------------------- USER CONFIG --------------------
CS2_DIR = r"E:\NWP\CS2_S1_matched\CS2\CS2 class point"
S1_DIR = r"F:\NWP\Classification Result\2023_maskSIC"
MATCH_CSV = r"E:\NWP\CS2_S1_matched\time_match_2023_filter.csv"
OUTPUT_DIR = r"F:\NWP\S1_CS2_overlap\2023"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CS2_CLASS_FIELD = "class"
S1_CLASS_MAP = {1: "ice", 2: "lead", 3: "refrozen"}
ICE_MATCH_RULE = "A"   # or "B"  (ice->1 vs ice->1&2)
SAVE_PLOTS = True
# -----------------------------------------------------

def read_cs2_points(path):
    gdf = gpd.read_file(path)
    gdf[CS2_CLASS_FIELD] = gdf[CS2_CLASS_FIELD].astype(str).str.lower().str.strip()
    gdf = gdf[gdf[CS2_CLASS_FIELD].isin(["ice", "lead"]) & gdf.geometry.notna()]
    return gdf

def polygonize_valid_region(rds):
    band = rds.read(1, masked=False)
    mask = np.ones_like(band, dtype=bool) if rds.nodata is None else (band != rds.nodata)
    shapes = features.shapes(mask.astype(np.uint8), transform=rds.transform)
    polys = [shape(g) for g,v in shapes if v == 1]
    if not polys:
        l,b,r,t = rds.bounds
        return Polygon([(l,b),(l,t),(r,t),(r,b)])
    merged = unary_union(polys)
    if isinstance(merged, MultiPolygon):
        merged = max(merged.geoms, key=lambda p: p.area)
    return merged

def sample_raster(rds, gdf):
    coords = [(p.x, p.y) for p in gdf.geometry]
    vals = [v[0] if v is not None and len(v)>0 else None for v in rds.sample(coords)]
    return [int(v) if v is not None and not np.isnan(v) else None for v in vals]

def apply_match(cs2_label, s1_val, rule):
    if s1_val is None:
        return "ignore"
    if cs2_label == "ice":
        return "match" if (s1_val==1 or (rule=="B" and s1_val==2)) else "mismatch"
    if cs2_label == "lead":
        return "match" if s1_val==2 else "mismatch"
    return "ignore"

def analyze_pair(cs2_file, s1_file, out_dir, rule):
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(s1_file) as rds:
        s1_crs = rds.crs
        s1_poly = polygonize_valid_region(rds)
        cs2 = read_cs2_points(cs2_file).to_crs(s1_crs)
        s1_vals = sample_raster(rds, cs2)
    cs2["S1_value"] = s1_vals
    cs2["S1_class"] = cs2["S1_value"].map(S1_CLASS_MAP).fillna("unknown")
    cs2["match_state"] = [apply_match(c, v, rule) for c,v in zip(cs2[CS2_CLASS_FIELD], cs2["S1_value"])]

    df = cs2[[CS2_CLASS_FIELD,"S1_class","match_state"]].copy()
    cross = pd.crosstab(df[CS2_CLASS_FIELD], df["S1_class"])
    total = len(df)
    m = (df["match_state"]=="match").sum()
    mm = (df["match_state"]=="mismatch").sum()
    ig = (df["match_state"]=="ignore").sum()
    acc = round(m/(m+mm)*100,2) if (m+mm)>0 else np.nan

    cross.to_csv(os.path.join(out_dir,"cross_tab.csv"),encoding="utf-8-sig")
    cs2.drop(columns="geometry").to_csv(os.path.join(out_dir,"points_result.csv"),index=False,encoding="utf-8-sig")

    if SAVE_PLOTS:
        fig,ax=plt.subplots(figsize=(8,8))
        gpd.GeoSeries([s1_poly],crs=s1_crs).boundary.plot(ax=ax,color='k')
        cs2.plot(ax=ax,column="match_state",markersize=10,legend=True)
        ax.set_title(os.path.basename(s1_file))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,"preview.png"),dpi=300)
        plt.close(fig)

    return dict(pair=os.path.basename(s1_file),
                N=total,match=m,mismatch=mm,ignore=ig,acc_percent=acc)

def find_file(dir_path, key):
    for f in os.listdir(dir_path):
        if key in f:
            return os.path.join(dir_path,f)
    return None

def main():
    df = pd.read_csv(MATCH_CSV)
    print(f"Loaded {len(df)} match entries.")
    all_stats=[]
    for _,row in df.iterrows():
        scene=row["sceneName"]
        cs2_base=os.path.splitext(os.path.basename(row["cs2_path"]))[0]
        s1_file=find_file(S1_DIR,scene)
        cs2_file=find_file(CS2_DIR,cs2_base)
        if s1_file and cs2_file:
            print(f"Processing {scene} ↔ {cs2_base}")
            outdir=os.path.join(OUTPUT_DIR,f"{scene}__{cs2_base}")
            try:
                stats=analyze_pair(cs2_file,s1_file,outdir,ICE_MATCH_RULE)
                all_stats.append(stats)
            except Exception as e:
                print(f"Error {scene}: {e}")
        else:
            print(f"⚠ Missing file for {scene}")
    if all_stats:
        total=pd.DataFrame(all_stats)
        total.to_csv(os.path.join(OUTPUT_DIR,"total_summary.csv"),index=False,encoding="utf-8-sig")
        print("\n=== Total Summary ===")
        print(total)
    else:
        print("No pairs processed.")

if __name__=="__main__":
    main()
