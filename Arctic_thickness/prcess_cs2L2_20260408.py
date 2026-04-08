import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import multiprocessing as mp

def process_one_l2_file(file_path, output_dir):
    track_id = os.path.basename(file_path)
    try:
        ds = xr.open_dataset(file_path, decode_timedelta=True)
        
        # 1. 获取 20Hz 基础维度数据
        time_20 = ds['time_20_ku'].values
        lat_poca = ds['lat_poca_20_ku'].values
        lon_poca = ds['lon_poca_20_ku'].values
        N_20 = len(time_20)
        
        # 2. 准备 1Hz 到 20Hz 的广播索引
        # 使用与 L1 脚本完全一致的索引逻辑：ind_first_meas_20hz_01
        ind_first = ds["ind_first_meas_20hz_01"].values.astype(int)
        
        # 3. 定义需要提取并广播的 1Hz 变量
        vars_01 = [
            'sea_ice_concentration_01', 
            'snow_density_01', 
            'snow_depth_01', 
            'mean_sea_surf_sea_ice_01'
        ]
        
        # 定义直接读取的 20Hz 变量
        vars_20 = [
            'radar_freeboard_20_ku',
            'surf_type_20_ku',
            'snow_depth_cor_20_ku',
            'ssha_interp_20_ku'
        ]
        
        # 4. 执行广播逻辑
        broadcasted_vars = {v: np.full(N_20, np.nan) for v in vars_01}
        for i, start_idx in enumerate(ind_first):
            end_idx = min(start_idx + 20, N_20)
            for v in vars_01:
                if v in ds:
                    broadcasted_vars[v][start_idx:end_idx] = ds[v].values[i]

        # 5. 构造 DataFrame
        df_l2 = pd.DataFrame({
            'track_id': track_id,
            'time_20_ku': time_20,
            'lat': lat_poca,  # 对应 L1 的 lat
            'lon': lon_poca,  # 对应 L1 的 lon
        })
        
        # 添加 20Hz 变量
        for v in vars_20:
            if v in ds:
                df_l2[v] = ds[v].values
            else:
                df_l2[v] = np.nan
        
        # 添加广播后的 1Hz 变量
        for v in vars_01:
            df_l2[v] = broadcasted_vars[v]

        # 6. 时间转换 (对齐 L1 的 utc_time)
        tai_epoch = datetime(2000, 1, 1)
        tai_sec = (time_20 - np.datetime64('2000-01-01T00:00:00')) / np.timedelta64(1, 's')
        df_l2['utc_time'] = [tai_epoch + pd.Timedelta(seconds=t) for t in tai_sec]

        # 7. 计算 L2 产品的海冰厚度 (用于对照你的 L1 thickness)
        # 采用 L2 官方公式：SIT = Freeboard * (rho_w / (rho_w - rho_i)) + SnowDepth * (rho_s / (rho_w - rho_i))
        # 这里 rho 数据如果不全，可保留 raw 参数后续对比
        
        # 导出
        out_path = os.path.join(output_dir, track_id.replace('.nc', '_L2.parquet'))
        df_l2.to_parquet(out_path, index=False)
        return True, track_id

    except Exception as e:
        return False, f"{track_id}: {str(e)}"

# # 批量处理逻辑 (保持一致性)
# def batch_process_l2(input_dir, output_dir, n_workers=4):
#     os.makedirs(output_dir, exist_ok=True)
#     files = glob.glob(os.path.join(input_dir, "*.nc"))
#     tasks = [(f, output_dir) for f in files]
#     with mp.Pool(n_workers) as pool:
#         results = pool.starmap(process_one_l2_file, tasks)
#         for success, msg in results:
#             print(f"{'SUCCESS' if success else 'FAILED'}: {msg}")

# if __name__ == "__main__":
#     L2_INPUT = r"E:\Arctic\SAR L2\2024\01"
#     L2_OUTPUT = r"E:\Arctic\SAR L2\2024\01\Processed_Parquet"
#     batch_process_l2(L2_INPUT, L2_OUTPUT)
# =====================================================================
# 修改后的批量处理逻辑 (支持 yyyy/mm 递归遍历)
# =====================================================================
def batch_process_l2(root_input_dir, root_output_dir, n_workers=4):
    """
    递归遍历 root_input_dir 下的所有子文件夹，处理 .nc 文件并保持目录结构
    """
    # 1. 使用 **/*.nc 开启递归搜索
    search_pattern = os.path.join(root_input_dir, "**", "*.nc")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"在 {root_input_dir} 及其子目录中没有找到 .nc 文件。")
        return
        
    print(f"发现 {len(files)} 个 L2 文件。启动 {n_workers} 个 Worker 进行处理...")
    
    tasks = []
    for f in files:
        # 2. 计算相对路径以保留 yyyy/mm 结构
        rel_dir = os.path.relpath(os.path.dirname(f), root_input_dir)
        current_output_dir = os.path.join(root_output_dir, rel_dir)
        
        # 3. 预先创建对应的输出子文件夹
        os.makedirs(current_output_dir, exist_ok=True)
        
        # 将具体的文件路径和对应的输出路径加入任务队列
        tasks.append((f, current_output_dir))
    
    # 4. 执行并行处理
    with mp.Pool(n_workers) as pool:
        # 使用 starmap 处理带多个参数的函数
        results = pool.starmap(process_one_l2_file, tasks)
        
        # 统计结果
        for success, msg in results:
            if not success:
                print(f"FAILED: {msg}")
    
    print("所有文件处理完毕。")

if __name__ == "__main__":
    # ==== 配置区 ====
    # 指向 L2 数据的根目录，例如 E:\Arctic\SAR L2
    # 脚本会自动处理其下的 2024\01, 2024\02 等所有子目录
    L2_INPUT = r"E:\Arctic\SAR L2"
    L2_OUTPUT = r"E:\Arctic\SAR L2_Processed_Results"
    
    # 建议根据 CPU 核心数设置，通常为 4 或 8
    N_WORKERS = 8 
    # ================
    
    batch_process_l2(L2_INPUT, L2_OUTPUT, n_workers=N_WORKERS)