import os
import pandas as pd
from ftplib import FTP, error_perm
from pathlib import Path
from tqdm import tqdm
import re
import time
import logging
import sys


# === 日志配置：去掉 emoji，指定 utf-8 写日志文件 ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cryosat2_download.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)


# === 配置部分 ===
CONFIG = {
    # 文件路径配置
    "csv_path": r"F:\NWP\CS2_S1_matched\time_match_2018_filter.csv",
    "cryosat2_dir": r"F:\NWP\CS2_L1\2018",

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
        "delay": 2,           # 每次下载后的延迟秒数
        "retry_delay": 5      # 重试前的等待秒数
    }
}


def generate_cs2_L1_name(cs2_filename):
    """将 CryoSat-2 L2 文件名转换为对应的 L1B 文件名"""
    return cs2_filename.replace("SIN_2__", "SIN_1B_")


def get_file_list_from_csv(csv_path):
    """从CSV文件中提取并生成需要下载的CryoSat-2 L1B文件列表"""
    logging.info(f"正在从CSV文件加载数据: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"成功加载CSV文件，共有 {len(df)} 条记录")

        cs2_l1_name_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="处理文件名"):
            cs2_original_name = os.path.basename(row['cs2_path'])
            cs2_l1_name = generate_cs2_L1_name(cs2_original_name)
            cs2_l1_name_list.append(cs2_l1_name)

        logging.info(f"共生成 {len(cs2_l1_name_list)} 个CryoSat-2 L1B文件名")
        return cs2_l1_name_list
    except Exception as e:
        logging.error(f"处理CSV文件时出错: {e}")
        raise


def connect_ftp(config):
    """连接到CryoSat-2 FTP服务器"""
    try:
        ftp = FTP(config["ftp"]["host"], timeout=config["ftp"]["timeout"])
        ftp.login(config["ftp"]["user"], config["ftp"]["pass"])
        logging.info(f"成功连接到FTP服务器: {config['ftp']['host']}")
        return ftp
    except Exception as e:
        logging.error(f"FTP连接失败: {e}")
        raise


def download_file_with_progress(ftp, filename, local_path: Path):
    """下载单个文件并显示进度条"""
    try:
        # 获取文件大小
        try:
            file_size = ftp.size(filename)
        except Exception:
            file_size = 0

        with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename) as pbar:
            with open(local_path, 'wb') as f:
                def callback(data):
                    f.write(data)
                    pbar.update(len(data))

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
    """
    output_dir = Path(config["cryosat2_dir"])
    retry_limit = config["download"]["retry_limit"]
    delay = config["download"]["delay"]
    retry_delay = config["download"]["retry_delay"]

    # 确保目标路径存在
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"下载目录: {output_dir}")

    total_files = len(file_list)
    downloaded = 0
    skipped = 0
    failed = 0

    logging.info(f"开始下载 {total_files} 个文件")

    ftp = None
    for i, filename in enumerate(file_list, 1):
        logging.info(f"[{i}/{total_files}] 处理: {filename}")

        local_file_path = output_dir / filename
        if local_file_path.exists():
            logging.info(f"文件已存在，跳过: {filename}")
            skipped += 1
            continue

        # ===== 仅解析日期，不关心 OFFL/LTA/OPER 标签 =====
        match = re.search(r'(\d{4})(\d{2})\d{2}T', filename)
        if not match:
            logging.warning(f"文件名格式无法解析日期，跳过: {filename}")
            failed += 1
            continue

        year, month = match.group(1), match.group(2)
        # 路径和你 FTP 截图一致：/SIR_SIN_L1/年/月
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
                    logging.error(f"文件未在目录中找到: {filename} in {remote_path}")
                    failed += 1
                    break

                # 下载文件
                if download_file_with_progress(ftp, filename, local_file_path):
                    logging.info(f"成功下载: {filename}")
                    downloaded += 1
                    time.sleep(delay)
                    break
                else:
                    raise Exception("下载进度中断")

            except (OSError, EOFError, error_perm, Exception) as e:
                logging.warning(f"下载失败（第 {attempt + 1} 次）: {filename} -> {e}")
                attempt += 1
                time.sleep(retry_delay)

                # 重连 FTP
                try:
                    if ftp:
                        ftp.quit()
                except Exception:
                    pass
                ftp = None

        if attempt >= retry_limit:
            logging.error(f"多次重试后仍失败，放弃下载: {filename}")
            failed += 1

        logging.info(
            f"当前统计 - 已下载: {downloaded}, 已跳过: {skipped}, 失败: {failed}, 剩余: {total_files - i}"
        )

    if ftp:
        try:
            ftp.quit()
            logging.info("FTP连接已关闭")
        except Exception:
            logging.warning("关闭FTP连接时出错")

    logging.info(f"下载完成 - 总计: {total_files}, 成功: {downloaded}, 已跳过: {skipped}, 失败: {failed}")
    return downloaded, skipped, failed


def main():
    """主函数"""
    try:
        logging.info("=== CryoSat-2 批量下载工具 ===")
        logging.info(f"CSV文件: {CONFIG['csv_path']}")
        logging.info(f"下载目录: {CONFIG['cryosat2_dir']}")

        file_list = get_file_list_from_csv(CONFIG["csv_path"])
        downloaded, skipped, failed = download_cryosat2_batch_resumable(file_list, CONFIG)

        success_rate = (downloaded + skipped) / len(file_list) * 100 if file_list else 0
        logging.info(f"下载成功率: {success_rate:.2f}%")

        return 0
    except Exception as e:
        logging.error(f"程序执行出错: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
