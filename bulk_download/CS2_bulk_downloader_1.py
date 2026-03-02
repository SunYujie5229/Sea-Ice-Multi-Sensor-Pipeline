import os
import pandas as pd
from ftplib import FTP
from pathlib import Path
import requests
from tqdm import tqdm
import re
import time
import logging
from ftplib import error_perm
import sys



# === 日志配置 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cryosat2_download.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# === 配置部分 ===
CONFIG = {
    # 文件路径配置
    "csv_path": r"F:\NWP\CS2_S1_matched\time_match_2020_filter.csv",
    # "sentinel1_dir": r"E:\NWP\CS2_S1_matched\Sentinel-1",
    "cryosat2_dir": r"C:\Users\TJ002\Desktop\CS2\2020",
    
    # CryoSat-2 FTP 设置
    "ftp": {
        "host": "science-pds.cryosat.esa.int",
        "user": "cryosat507",
        "pass": "qOnliGbm",
        "timeout": 30
    },
    
    # 下载设置
    "download": {
        "retry_limit": 3,     # 每个文件最多重试次数
        "delay": 2,          # 每次下载后的延迟秒数
        "retry_delay": 5     # 重试前的等待秒数
    }
}


# def generate_cs2_L1_name(cs2_filename):
#     """将 CryoSat-2 L2 文件名转换为对应的 L1B 文件名
    
#     Args:
#         cs2_filename: CryoSat-2 L2 文件名
        
#     Returns:
#         对应的 L1B 文件名
#     """
#     return cs2_filename.replace("SIN_2__", "SIN_1B_")

def generate_cs2_L1_name(cs2_filename):
    name = cs2_filename

    # 若包含 "LTA__", 则修成 "LTA_"
    # if "LTA__" in name:
    #     name = name.replace("LTA__", "LTA_")

    # 永远需要把 Level-2 的 SIN_2__ → Level-1 的 SIN_1B_
    name = name.replace("SIN_2__", "SIN_1B_")

    return name



def get_file_list_from_csv(csv_path):
    """从CSV文件中提取并生成需要下载的CryoSat-2 L1B文件列表
    
    Args:
        csv_path: 包含CryoSat-2文件信息的CSV文件路径
        
    Returns:
        需要下载的L1B文件名列表
    """
    logging.info(f"正在从CSV文件加载数据: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"成功加载CSV文件，共有{len(df)}条记录")
        
        cs2_l1_name_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="处理文件名"):
            cs2_original_name = os.path.basename(row['cs2_path'])
            cs2_l1_name = generate_cs2_L1_name(cs2_original_name)
            cs2_l1_name_list.append(cs2_l1_name)
            
        logging.info(f"共生成{len(cs2_l1_name_list)}个CryoSat-2 L1B文件名")
        return cs2_l1_name_list
    except Exception as e:
        logging.error(f"处理CSV文件时出错: {e}")
        raise


def connect_ftp(config):
    """连接到CryoSat-2 FTP服务器
    
    Args:
        config: 包含FTP连接信息的配置字典
        
    Returns:
        已连接的FTP对象
    """
    try:
        ftp = FTP(config["ftp"]["host"], timeout=config["ftp"]["timeout"])
        ftp.login(config["ftp"]["user"], config["ftp"]["pass"])
        logging.info(f"成功连接到FTP服务器: {config['ftp']['host']}")
        return ftp
    except Exception as e:
        logging.error(f"FTP连接失败: {e}")
        raise


def download_file_with_progress(ftp, filename, local_path):
    """下载单个文件并显示进度条
    
    Args:
        ftp: 已连接的FTP对象
        filename: 要下载的文件名
        local_path: 本地保存路径
        
    Returns:
        bool: 下载是否成功
    """
    try:
        # 获取文件大小
        file_size = ftp.size(filename)
        if file_size is None:
            file_size = 0  # 如果无法获取大小，设为0
            
        # 创建进度条
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename) as pbar:
            with open(local_path, 'wb') as f:
                # 定义回调函数来更新进度条
                def callback(data):
                    f.write(data)
                    pbar.update(len(data))
                
                # 下载文件
                ftp.retrbinary(f"RETR {filename}", callback)
        return True
    except Exception as e:
        logging.error(f"下载文件时出错: {e}")
        # 如果下载失败，删除可能部分下载的文件
        if local_path.exists():
            local_path.unlink()
        return False


def download_cryosat2_batch_resumable(file_list, config):
    """
    批量下载 CryoSat-2 数据，具备断点续传和容错机制。
    
    Args:
        file_list: 所有需要下载的文件名列表
        config: 包含下载配置的字典
    """
    # 配置参数
    output_dir = Path(config["cryosat2_dir"])
    retry_limit = config["download"]["retry_limit"]
    delay = config["download"]["delay"]
    retry_delay = config["download"]["retry_delay"]
    
    # 确保目标路径存在
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"下载目录: {output_dir}")
    
    # 统计信息
    total_files = len(file_list)
    downloaded = 0
    skipped = 0
    failed = 0
    
    logging.info(f"开始下载 {total_files} 个文件")
    
    ftp = None
    for i, filename in enumerate(file_list, 1):
        # 显示总体进度
        progress_msg = f"[{i}/{total_files}] 处理: {filename}"
        logging.info(progress_msg)
        
        # 检查本地文件是否已存在
        local_file_path = output_dir / filename
        if local_file_path.exists():
            logging.info(f"✅ 已存在，跳过: {filename}")
            skipped += 1
            continue

        # 提取年份和月份
        match = re.match(r'CS_OFFL_SIR_SIN_1B_(\d{4})(\d{2})\d{2}T', filename)
        if not match:
            logging.warning(f"⚠️ 格式不匹配，跳过: {filename}")
            failed += 1
            continue
            
        year, month = match.group(1), match.group(2)
        remote_path = f"/SIR_SIN_L1/{year}/{month}"

        attempt = 0
        while attempt < retry_limit:
            try:
                if ftp is None:
                    ftp = connect_ftp(config)

                # 切换到远程目录
                ftp.cwd(remote_path)
                
                # 检查文件是否存在
                files = ftp.nlst()
                if filename not in files:
                    logging.error(f"❌ 文件未找到: {filename} in {remote_path}")
                    failed += 1
                    break

                # 下载文件并显示进度
                if download_file_with_progress(ftp, filename, local_file_path):
                    logging.info(f"✅ 成功下载: {filename}")
                    downloaded += 1
                    time.sleep(delay)  # 下载间隔
                    break  # 成功后退出重试循环
                else:
                    raise Exception("下载进度中断")

            except (OSError, EOFError, error_perm) as e:
                logging.warning(f"⚠️ 下载失败（第 {attempt + 1} 次）: {filename} -> {e}")
                attempt += 1
                time.sleep(retry_delay)
                # 尝试重新连接 FTP
                try:
                    if ftp:
                        ftp.quit()
                except:
                    pass
                ftp = None  # 触发重连
        else:
            logging.error(f"❌ 放弃下载: {filename} 在 {retry_limit} 次尝试后")
            failed += 1

        # 显示当前统计信息
        stats_msg = f"当前统计 - 已下载: {downloaded}, 已跳过: {skipped}, 失败: {failed}, 剩余: {total_files - i}"
        logging.info(stats_msg)

    # 最后关闭 FTP 连接
    if ftp:
        try:
            ftp.quit()
            logging.info("FTP连接已关闭")
        except:
            logging.warning("关闭FTP连接时出错")
    
    # 显示最终统计信息
    logging.info(f"下载完成 - 总计: {total_files}, 成功: {downloaded}, 已跳过: {skipped}, 失败: {failed}")
    return downloaded, skipped, failed

def main():
    """主函数，协调整个下载过程"""
    try:
        # 显示配置信息
        logging.info("=== CryoSat-2 批量下载工具 ===")
        logging.info(f"CSV文件: {CONFIG['csv_path']}")
        logging.info(f"下载目录: {CONFIG['cryosat2_dir']}")
        
        # 获取文件列表
        file_list = get_file_list_from_csv(CONFIG["csv_path"])
        
        # 开始下载
        downloaded, skipped, failed = download_cryosat2_batch_resumable(file_list, CONFIG)
        
        # 显示结果摘要
        success_rate = (downloaded + skipped) / len(file_list) * 100 if file_list else 0
        logging.info(f"下载成功率: {success_rate:.2f}%")
        
        return 0
    except Exception as e:
        logging.error(f"程序执行出错: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())