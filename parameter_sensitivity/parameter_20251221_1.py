import os
import glob
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# 年份列表（排除缺少 S1 分类数据的年份）
YEARS = [2015, 2016, 2017, 2018,2019, 2020, 2021,2022,2023]  # 暂时跳过 2018, 2022

 
# 参数空间定义
PP_FLOE_THRESHOLDS = [6, 7, 8, 9, 10, 11, 12, 13]
PP_LEAD_THRESHOLDS = [14, 15, 16, 17, 18, 19, 20, 21]
STD_THRESHOLDS = [2.29, 3.29, 4.62, 5.29, 6.29]

# 实验组定义
EXPERIMENTS = {
    "A_STD_sensitivity": {
        "description": "STD sensitivity (fixed PP_floe=9, PP_lead=18)",
        "fixed_params": {"pp_floe": 9, "pp_lead": 18},
        "vary_param": "std",
        "values": STD_THRESHOLDS
    },
    "B_PPfloe_sensitivity": {
        "description": "PP_floe sensitivity (fixed PP_lead=18, STD=4.62)",
        "fixed_params": {"pp_lead": 18, "std": 4.62},
        "vary_param": "pp_floe",
        "values": PP_FLOE_THRESHOLDS
    },
    "C_PPlead_sensitivity": {
        "description": "PP_lead sensitivity (fixed PP_floe=9, STD=4.62)",
        "fixed_params": {"pp_floe": 9, "std": 4.62},
        "vary_param": "pp_lead",
        "values": PP_LEAD_THRESHOLDS
    }
}

# 基础路径配置
BASE_INPUT_DIR = r"F:\NWP\CS2_L1"  # CS2 L1B 原始数据根目录
BASE_OUTPUT_DIR = r"F:\NWP\CS2_S1_matched\parameter_experiments"  # 实验输出根目录
BASE_MATCH_CSV_DIR = r"F:\NWP\CS2_S1_matched"  # time_match CSV 文件目录

# S1 数据路径（用于 overlap 计算）
BASE_S1_DIR = r"C:\Users\TJ002\Desktop\CS2_S1_result\filter"

# 窗口半径（保持不变）
WINDOW_RADIUS = 12


# =============================================================================
# HELPER FUNCTIONS (从原始代码中提取)
# =============================================================================

def calculate_pulse_peakiness(waveform: np.ndarray) -> float:
    """计算脉冲峰度值"""
    if not isinstance(waveform, np.ndarray):
        waveform = np.array(waveform)
    
    if waveform.size == 0 or np.all(waveform == 0):
        return np.nan
    
    b_max = np.argmax(waveform)
    start_index = b_max - 50
    end_index = b_max + 77
    
    cropped_waveform = np.zeros(128)
    src_start = max(0, start_index)
    src_end = min(waveform.size, end_index + 1)
    
    dest_start = max(0, -start_index)
    dest_end = dest_start + (src_end - src_start)
    
    cropped_waveform[dest_start:dest_end] = waveform[src_start:src_end]
    
    p_max = np.max(cropped_waveform)
    p_mean = np.mean(cropped_waveform)
    
    if p_mean == 0:
        return np.nan
    
    return p_max / p_mean


# =============================================================================
# CS2 分类函数（参数化版本）
# =============================================================================

def classify_cs2_with_params(input_folder, output_folder, pp_lead_thresh, pp_floe_thresh, std_thresh):
    """
    使用指定参数对 CS2 数据进行分类
    
    Parameters:
    -----------
    input_folder : str
        CS2 L1B 数据文件夹
    output_folder : str
        分类结果输出文件夹
    pp_lead_thresh : float
        Lead 的 PP 阈值下限
    pp_floe_thresh : float
        Floe 的 PP 阈值上限
    std_thresh : float
        STD 阈值
    """
    import xarray as xr
    import geopandas as gpd
    from datetime import datetime
    
    print(f"\n  Classifying with params: PP_lead≥{pp_lead_thresh}, PP_floe<{pp_floe_thresh}, STD≤{std_thresh}")
    
    # 搜索文件
    search_patterns = [
        "CS_OFFL_SIR_SIN_1B_*.nc",
        "CS_LTA__SIR_SIN_1B_*.nc",
        "CS_NRT__SIR_SIN_1B_*.nc"
    ]
    
    nc_files = []
    for pattern in search_patterns:
        files = glob.glob(os.path.join(input_folder, pattern))
        nc_files.extend(files)
    
    if not nc_files:
        print(f"    ✗ No CS2 files found in: {input_folder}")
        return False
    
    os.makedirs(output_folder, exist_ok=True)
    
    processed_count = 0
    
    for file_path in nc_files:
        file_name = os.path.basename(file_path)
        
        try:
            with xr.open_dataset(file_path) as ds:
                # 提取变量
                lat, lon, time_raw, waveforms, stack_std, stack_kurt = [
                    ds[var].values for var in
                    [
                        "lat_20_ku", "lon_20_ku", "time_20_ku",
                        "pwr_waveform_20_ku", "stack_std_20_ku", "stack_kurtosis_20_ku"
                    ]
                ]
                
                # 基础过滤
                valid_mask = (lat < 90) & (lon > -181)
                if not np.any(valid_mask):
                    continue
                
                lat = lat[valid_mask]
                lon = lon[valid_mask]
                time_raw = time_raw[valid_mask]
                waveforms = waveforms[valid_mask]
                stack_std = stack_std[valid_mask]
                stack_kurt = stack_kurt[valid_mask]
                
                # 时间处理
                if np.issubdtype(time_raw.dtype, np.datetime64):
                    utc_time = time_raw
                else:
                    utc_time = pd.to_datetime(
                        time_raw,
                        origin=datetime(2000, 1, 1),
                        unit="s"
                    )
                
                # 计算 PP
                pp_vals = np.array([calculate_pulse_peakiness(wf) for wf in waveforms])
                
                df_classify = pd.DataFrame({
                    "pp": pp_vals,
                    "std": stack_std
                })
                
                df_classify["type"] = np.nan
                
                # 使用自定义参数进行分类
                mask_lead = (df_classify["pp"] >= pp_lead_thresh) & (df_classify["std"] <= std_thresh)
                mask_ice = (df_classify["pp"] < pp_floe_thresh) & (df_classify["std"] > std_thresh)
                
                mask_combined = mask_lead | mask_ice
                df_classify.loc[mask_combined, "type"] = np.where(
                    df_classify.loc[mask_combined, "pp"] >= pp_lead_thresh,
                    "lead",
                    "ice"
                )
                
                # 构建最终 DataFrame
                df = pd.DataFrame({
                    "latitude": lat,
                    "longitude": lon,
                    "utc_time": utc_time,
                    "pp": pp_vals,
                    "std": stack_std,
                    "kurtosis": stack_kurt,
                    "class": df_classify["type"]
                })
                
                df.dropna(subset=["class"], inplace=True)
                
                if df.empty:
                    continue
                
                # 转换为 GeoDataFrame
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df.longitude, df.latitude),
                    crs="EPSG:4326"
                )
                
                # 导出
                base_name = os.path.splitext(file_name)[0]
                output_path = os.path.join(output_folder, f"{base_name}_classified.gpkg")
                gdf.to_file(output_path, driver="GPKG")
                
                processed_count += 1
                
        except Exception as e:
            print(f"    ✗ Error processing {file_name}: {e}")
            continue
    
    print(f"    ✓ Processed {processed_count}/{len(nc_files)} files")
    return processed_count > 0


# =============================================================================
# OVERLAP DENSITY 计算（简化版）
# =============================================================================

def run_overlap_density_for_year(year, cs2_dir, s1_dir, match_csv_path, output_dir, window_radius=12):
    """
    为单个年份运行 overlap density 计算
    """
    import rasterio
    import geopandas as gpd
    import re
    from shapely.geometry import Point
    from rasterio.transform import rowcol
    import warnings
    warnings.filterwarnings("ignore")
    
    print(f"\n  Computing overlap density for year {year}...")
    
    if not os.path.exists(match_csv_path):
        print(f"    ✗ Match CSV not found: {match_csv_path}")
        return None
    
    match_df = pd.read_csv(match_csv_path)
    results = []
    
    # S1 分类值定义
    S1_BACKGROUND = 0
    S1_ICE = 1
    S1_LEAD = 2
    S1_REFR = 3
    
    for idx, row in match_df.iterrows():
        scene_name = row['sceneName']
        
        # 查找文件
        cs2_key = re.search(r"(\d{8}T\d{6}_\d{8}T\d{6})", Path(row["cs2_path"]).name)
        if not cs2_key:
            continue
        
        # 查找 CS2 文件
        cs2_pattern = cs2_key.group(1)
        cs2_file = None
        for file in os.listdir(cs2_dir):
            if cs2_pattern in file and file.lower().endswith('.gpkg'):
                cs2_file = os.path.join(cs2_dir, file)
                break
        
        # 查找 S1 文件
        s1_file = None
        for file in os.listdir(s1_dir):
            if scene_name in file and file.lower().endswith('.tif'):
                s1_file = os.path.join(s1_dir, file)
                break
        
        if not cs2_file or not s1_file:
            continue
        
        try:
            # 读取 CS2
            cs2_gdf = gpd.read_file(cs2_file)
            
            # 检测分类列
            class_col = None
            for c in ["class", "Class", "CLASS", "classification"]:
                if c in cs2_gdf.columns:
                    class_col = c
                    break
            
            if not class_col:
                continue
            
            # 读取 S1
            with rasterio.open(s1_file) as src:
                s1_data = src.read(1)
                s1_crs = src.crs
                s1_transform = src.transform
                s1_rows, s1_cols = src.shape
            
            # 投影 CS2 到 S1 坐标系
            cs2_proj = cs2_gdf.to_crs(s1_crs)
            
            # 坐标转换
            x_coords = cs2_proj.geometry.x.tolist()
            y_coords = cs2_proj.geometry.y.tolist()
            rows_list, cols_list = rowcol(s1_transform, x_coords, y_coords)
            rows_arr = np.array(rows_list)
            cols_arr = np.array(cols_list)
            
            # 矩形边界过滤
            row_mask = (rows_arr >= 0) & (rows_arr < s1_rows)
            col_mask = (cols_arr >= 0) & (cols_arr < s1_cols)
            rect_mask = row_mask & col_mask
            
            # 提取 S1 像素值
            valid_indices = np.where(rect_mask)[0]
            s1_values = np.full(len(cs2_gdf), S1_BACKGROUND, dtype=s1_data.dtype)
            
            if valid_indices.size > 0:
                s1_values[valid_indices] = s1_data[rows_arr[valid_indices], cols_arr[valid_indices]]
            
            # 最终过滤：必须在有效区域且非背景
            value_mask = (s1_values != S1_BACKGROUND)
            final_filter_mask = rect_mask & value_mask
            
            cs2_gdf_overlapped = cs2_gdf[final_filter_mask].copy()
            
            if len(cs2_gdf_overlapped) == 0:
                continue
            
            # 计算 CS2 密度
            cs2_gdf_overlapped[class_col] = cs2_gdf_overlapped[class_col].str.lower().str.strip()
            count_lead = (cs2_gdf_overlapped[class_col] == "lead").sum()
            count_ice = (cs2_gdf_overlapped[class_col] == "ice").sum()
            count_refrozen = (cs2_gdf_overlapped[class_col] == "refrozen").sum()
            total_cs2 = count_lead + count_ice + count_refrozen
            
            if total_cs2 == 0:
                continue
            
            density_cs2_lead = count_lead / total_cs2
            density_cs2_ice = count_ice / total_cs2
            
            # 计算 S1 密度（窗口采样）
            cs2_proj_overlapped = cs2_gdf_overlapped.to_crs(s1_crs)
            unique_pixels = set()
            
            for pt in cs2_proj_overlapped.geometry:
                if not isinstance(pt, Point):
                    continue
                
                x, y = pt.x, pt.y
                col, row = (~s1_transform) * (x, y)
                row, col = int(row), int(col)
                
                r1, r2 = max(row - window_radius, 0), min(row + window_radius, s1_rows)
                c1, c2 = max(col - window_radius, 0), min(col + window_radius, s1_cols)
                
                for rr in range(r1, r2):
                    for cc in range(c1, c2):
                        unique_pixels.add((rr, cc))
            
            if len(unique_pixels) == 0:
                continue
            
            # 统计 S1 像素
            lead_total = 0
            ice_total = 0
            refr_total = 0
            total_pixels = 0
            
            for rr, cc in unique_pixels:
                val = s1_data[rr, cc]
                if val == S1_LEAD:
                    lead_total += 1
                elif val == S1_ICE:
                    ice_total += 1
                elif val == S1_REFR:
                    refr_total += 1
                
                if val != S1_BACKGROUND:
                    total_pixels += 1
            
            if total_pixels == 0:
                continue
            
            density_s1_lead = lead_total / total_pixels
            density_s1_ice = ice_total / total_pixels
            
            # 保存结果
            results.append({
                "scene_name": scene_name,
                "total_CS2_overlap": total_cs2,
                "density_CS2_lead": density_cs2_lead,
                "density_CS2_ice": density_cs2_ice,
                "S1_total_pixels": total_pixels,
                "density_S1_lead": density_s1_lead,
                "density_S1_ice": density_s1_ice,
                "diff_lead_density": density_cs2_lead - density_s1_lead,
                "diff_ice_density": density_cs2_ice - density_s1_ice
            })
            
        except Exception as e:
            print(f"    ✗ Error processing scene {scene_name}: {e}")
            continue
    
    if results:
        os.makedirs(output_dir, exist_ok=True)
        outcsv = os.path.join(output_dir, f"density_summary_{year}.csv")
        pd.DataFrame(results).to_csv(outcsv, index=False)
        print(f"    ✓ Saved: {outcsv}")
        return results
    else:
        print(f"    ✗ No valid results for year {year}")
        return None


# =============================================================================
# 主实验流程
# =============================================================================

def run_parameter_experiment():
    """
    主实验函数：按实验组遍历参数
    """
    
    print("=" * 80)
    print("CS2–S1 PARAMETER SENSITIVITY EXPERIMENT")
    print("=" * 80)
    print(f"\nYears: {YEARS}")
    print(f"\nExperiment Groups:")
    for exp_name, exp_config in EXPERIMENTS.items():
        print(f"  {exp_name}: {exp_config['description']}")
        print(f"    → Testing {len(exp_config['values'])} values for {exp_config['vary_param']}")
    
    total_experiments = sum(len(exp['values']) for exp in EXPERIMENTS.values())
    print(f"\nTotal experiments: {total_experiments}")
    
    # 存储所有实验组的结果
    all_experiment_groups = {}
    
    # 遍历每个实验组
    for exp_group_name, exp_config in EXPERIMENTS.items():
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT GROUP: {exp_group_name}")
        print(f"{exp_config['description']}")
        print(f"{'='*80}")
        
        # 创建实验组输出目录
        exp_group_dir = os.path.join(BASE_OUTPUT_DIR, exp_group_name)
        os.makedirs(exp_group_dir, exist_ok=True)
        
        # 获取固定参数和变化参数
        fixed_params = exp_config['fixed_params']
        vary_param = exp_config['vary_param']
        values_to_test = exp_config['values']
        
        # 存储本实验组的结果
        group_results = []
        
        # 遍历要测试的参数值
        for test_idx, test_value in enumerate(values_to_test, 1):
            
            # 构建完整的参数集
            if vary_param == "std":
                pp_floe = fixed_params["pp_floe"]
                pp_lead = fixed_params["pp_lead"]
                std_thresh = test_value
            elif vary_param == "pp_floe":
                pp_floe = test_value
                pp_lead = fixed_params["pp_lead"]
                std_thresh = fixed_params["std"]
            elif vary_param == "pp_lead":
                pp_floe = fixed_params["pp_floe"]
                pp_lead = test_value
                std_thresh = fixed_params["std"]
            
            print(f"\n{'-'*80}")
            print(f"Test {test_idx}/{len(values_to_test)} in {exp_group_name}")
            print(f"Parameters: PP_floe<{pp_floe}, PP_lead≥{pp_lead}, STD≤{std_thresh}")
            print(f"{'-'*80}")
            
            # 创建本次测试的输出目录
            test_name = f"PPfloe{pp_floe}_PPlead{pp_lead}_STD{std_thresh:.2f}"
            test_output_dir = os.path.join(exp_group_dir, test_name)
            
            # 存储所有年份的所有场景结果（用于 global RMSD）
            all_scenes_results = []
            
            # 存储本次测试的年度结果（用于 yearly RMSD）
            yearly_rmsd_results = []
            
            # 对每个年份进行处理
            for year in YEARS:
                print(f"\n[Year {year}]")
                
                # 路径配置
                cs2_input_dir = os.path.join(BASE_INPUT_DIR, str(year))
                cs2_output_dir = os.path.join(test_output_dir, "cs2_classified", str(year))
                match_csv = os.path.join(BASE_MATCH_CSV_DIR, f"time_match_{year}_filter.csv")
                s1_dir = os.path.join(BASE_S1_DIR, str(year), "tif")
                overlap_output_dir = os.path.join(test_output_dir, "overlap_results", str(year))
                
                # 检查所有必需的输入文件/目录是否存在
                missing_data = []
                if not os.path.exists(cs2_input_dir):
                    missing_data.append(f"CS2 input: {cs2_input_dir}")
                if not os.path.exists(match_csv):
                    missing_data.append(f"Match CSV: {match_csv}")
                if not os.path.exists(s1_dir):
                    missing_data.append(f"S1 classification: {s1_dir}")
                
                if missing_data:
                    print(f"  ⚠️ Skipping year {year} - Missing data:")
                    for item in missing_data:
                        print(f"     - {item}")
                    continue
                
                # Step 1: CS2 分类
                print(f"  Step 1: CS2 Classification")
                classify_success = classify_cs2_with_params(
                    cs2_input_dir,
                    cs2_output_dir,
                    pp_lead,
                    pp_floe,
                    std_thresh
                )
                
                if not classify_success:
                    print(f"  ✗ Classification failed for year {year}")
                    continue
                
                # Step 2: Overlap Density 计算
                print(f"  Step 2: Overlap Density Calculation")
                density_results = run_overlap_density_for_year(
                    year,
                    cs2_output_dir,
                    s1_dir,
                    match_csv,
                    overlap_output_dir,
                    WINDOW_RADIUS
                )
                
                if density_results is None:
                    print(f"  ✗ Density calculation failed for year {year}")
                    continue
                
                # 将本年度的场景结果添加到全局场景列表
                all_scenes_results.extend(density_results)
                
                # Step 3: 计算年度 RMSD (保留用于分析)
                df_density = pd.DataFrame(density_results)
                
                rmsd_lead = np.sqrt(np.mean(df_density["diff_lead_density"]**2))
                rmsd_ice = np.sqrt(np.mean(df_density["diff_ice_density"]**2))
                
                mae_lead = np.mean(np.abs(df_density["diff_lead_density"]))
                mae_ice = np.mean(np.abs(df_density["diff_ice_density"]))
                
                yearly_rmsd_results.append({
                    "year": year,
                    "n_scenes": len(df_density),
                    "rmsd_lead": rmsd_lead,
                    "rmsd_ice": rmsd_ice,
                    "mae_lead": mae_lead,
                    "mae_ice": mae_ice,
                    "mean_diff_lead": df_density["diff_lead_density"].mean(),
                    "mean_diff_ice": df_density["diff_ice_density"].mean()
                })
                
                print(f"  ✓ Year {year} completed. RMSD_lead={rmsd_lead:.4f}, RMSD_ice={rmsd_ice:.4f}")
            
            # Step 4: 计算 Global RMSD (所有年份所有场景池化)
            if all_scenes_results:
                df_all_scenes = pd.DataFrame(all_scenes_results)
                
                # Global RMSD: 对所有场景直接计算
                global_rmsd_lead = np.sqrt(np.mean(df_all_scenes["diff_lead_density"]**2))
                global_rmsd_ice = np.sqrt(np.mean(df_all_scenes["diff_ice_density"]**2))
                
                global_mae_lead = np.mean(np.abs(df_all_scenes["diff_lead_density"]))
                global_mae_ice = np.mean(np.abs(df_all_scenes["diff_ice_density"]))
                
                total_scenes = len(df_all_scenes)
                
                print(f"\n  📊 Global Statistics (all {total_scenes} scenes pooled):")
                print(f"     Global RMSD (Lead): {global_rmsd_lead:.4f}")
                print(f"     Global RMSD (Ice):  {global_rmsd_ice:.4f}")
                print(f"     Global MAE (Lead):  {global_mae_lead:.4f}")
                print(f"     Global MAE (Ice):   {global_mae_ice:.4f}")
                
                # 保存本次测试的年度 RMSD 汇总 (用于分析)
                if yearly_rmsd_results:
                    df_yearly = pd.DataFrame(yearly_rmsd_results)
                    yearly_summary_path = os.path.join(test_output_dir, "yearly_rmsd_summary.csv")
                    df_yearly.to_csv(yearly_summary_path, index=False)
                    
                    # 计算多年平均 RMSD (仅用于参考)
                    avg_rmsd_lead = df_yearly["rmsd_lead"].mean()
                    avg_rmsd_ice = df_yearly["rmsd_ice"].mean()
                    
                    print(f"\n  📈 Yearly Average (for reference only):")
                    print(f"     Average RMSD (Lead): {avg_rmsd_lead:.4f}")
                    print(f"     Average RMSD (Ice):  {avg_rmsd_ice:.4f}")
                
                # 保存所有场景的详细结果
                all_scenes_path = os.path.join(test_output_dir, "all_scenes_density.csv")
                df_all_scenes.to_csv(all_scenes_path, index=False)
                
                # 添加到实验组结果
                group_results.append({
                    "test_name": test_name,
                    "pp_floe_thresh": pp_floe,
                    "pp_lead_thresh": pp_lead,
                    "std_thresh": std_thresh,
                    "varying_param": vary_param,
                    "varying_value": test_value,
                    
                    # Global RMSD (主要指标)
                    "global_rmsd_lead": global_rmsd_lead,
                    "global_rmsd_ice": global_rmsd_ice,
                    "global_mae_lead": global_mae_lead,
                    "global_mae_ice": global_mae_ice,
                    
                    # Yearly Average RMSD (参考指标)
                    "yearly_avg_rmsd_lead": avg_rmsd_lead if yearly_rmsd_results else np.nan,
                    "yearly_avg_rmsd_ice": avg_rmsd_ice if yearly_rmsd_results else np.nan,
                    
                    "total_scenes": total_scenes,
                    "years_processed": len(yearly_rmsd_results)
                })
                
                print(f"\n  ✓ Test {test_idx} completed.")
        
        # 保存本实验组的汇总结果
        if group_results:
            df_group = pd.DataFrame(group_results)
            
            # 按 Global RMSD 排序
            df_group_sorted = df_group.sort_values("global_rmsd_lead")
            
            group_summary_path = os.path.join(exp_group_dir, f"{exp_group_name}_summary.csv")
            df_group_sorted.to_csv(group_summary_path, index=False)
            
            print(f"\n{'='*80}")
            print(f"EXPERIMENT GROUP {exp_group_name} COMPLETED")
            print(f"{'='*80}")
            print(f"Results saved to: {group_summary_path}")
            print(f"\nTop 3 parameter values (by Global Lead RMSD):")
            top_cols = ["varying_value", "global_rmsd_lead", "global_rmsd_ice", "total_scenes"]
            print(df_group_sorted[top_cols].head(3).to_string(index=False))
            
            # 找到本组最佳参数
            best_test = df_group_sorted.iloc[0]
            print(f"\n🏆 BEST in {exp_group_name}:")
            print(f"   {vary_param.upper()} = {best_test['varying_value']}")
            print(f"   Global RMSD (Lead): {best_test['global_rmsd_lead']:.4f}")
            print(f"   Global RMSD (Ice):  {best_test['global_rmsd_ice']:.4f}")
            print(f"   Total scenes: {best_test['total_scenes']}")
            
            # 保存到总结果字典
            all_experiment_groups[exp_group_name] = df_group_sorted
    
    # 生成总报告
    print("\n" + "="*80)
    print("ALL EXPERIMENT GROUPS COMPLETED")
    print("="*80)
    
    print("\n📋 Summary of Best Parameters from Each Group:")
    for exp_name, df_group in all_experiment_groups.items():
        if not df_group.empty:
            best = df_group.iloc[0]
            print(f"\n{exp_name}:")
            print(f"  Best {best['varying_param']}: {best['varying_value']}")
            print(f"  Global RMSD (Lead): {best['global_rmsd_lead']:.4f}")
            print(f"  Global RMSD (Ice):  {best['global_rmsd_ice']:.4f}")
    
    print("\n✅ All processing complete!")
    print(f"Results saved in: {BASE_OUTPUT_DIR}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_parameter_experiment()