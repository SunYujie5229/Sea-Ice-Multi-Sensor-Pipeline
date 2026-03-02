import os
import re
from ftplib import FTP
import pandas as pd
from datetime import datetime

# ===== 用户配置 =====
s1_folder = r"E:\NWP\CS2_S1_matched\Sentinel-1"   # 用不到也保留
save_dir  = r"D:\S1_CS2_data\SIC\2018"          # 保存SIC的根目录
ftp_host  = "data.seaice.uni-bremen.de"
username  = "anonymous"
password  = ""

csv_path = r"E:\NWP\CS2_S1_matched\time_match_2018.csv"
col_with_cs2_name = "cs2_path"   # CSV里存放CS2文件名/路径的列

# CryoSat-2 文件名中提取 YYYYMMDD 的正则（如 CS_OFFL_SIR_SIN_2__20230629T124858_...）
CS2_DATE_RE = re.compile(r"(\d{8})T?\d{0,6}")

MON3 = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]


# ========= 基础工具 =========
def month_to_mon3(yyyymmdd: str):
    dt = datetime.strptime(yyyymmdd, "%Y%m%d")
    return dt.year, MON3[dt.month-1]

def extract_dates_from_csv(csv_path: str, col_with_cs2_name: str):
    df = pd.read_csv(csv_path)
    dates = []
    for s in df[col_with_cs2_name].astype(str):
        m = CS2_DATE_RE.search(s)
        if m:
            dates.append(m.group(1))
    dates = sorted(set(dates))
    return dates

def ftp_connect(host: str, user: str, pwd: str) -> FTP:
    ftp = FTP(host, timeout=60)
    ftp.login(user, pwd)
    return ftp

def ftp_listdir(ftp: FTP, path: str):
    try:
        ftp.cwd(path)
        return ftp.nlst()
    except Exception as e:
        print(f"[WARN] listdir fail: {path} -> {e}")
        return []

def ftp_download(ftp: FTP, remote_dir: str, filename: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    local_path = os.path.join(out_dir, filename)
    if os.path.exists(local_path):
        print(f"[SKIP] exists: {local_path}")
        return local_path
    ftp.cwd(remote_dir)
    with open(local_path, "wb") as f:
        ftp.retrbinary(f"RETR {filename}", f.write)
    print(f"[OK] {remote_dir}/{filename} -> {local_path}")
    return local_path

def pick_best_file(files_in_month, yyyymmdd, prefer_ext=(".tif", ".nc")):
    """
    在当月清单里选择与 yyyymmdd 匹配的文件；
    优先使用 prefer_ext 中靠前的扩展名（如 .tif），
    若同一扩展有多个版本，取名字排序最大的那个（通常是较新版本）。
    """
    buckets = {ext: [] for ext in prefer_ext}
    for fn in files_in_month:
        for ext in prefer_ext:
            if fn.endswith(ext) and yyyymmdd in fn:
                buckets[ext].append(fn)
                break
    for ext in prefer_ext:
        cand = buckets[ext]
        if cand:
            cand.sort()
            return cand[-1]
    return None


# ========= 核心批量下载器 =========
def batch_download_sic_dates(
    dates,
    product_root,           # 例如 "/amsr2/asi_daygrid_swath/n6250" 或 "/amsr2/asi_daygrid_swath/n3125"
    region=None,            # 例如 "Arctic" 或 "NorthWestPassage"；为 None 表示不进入子目录
    prefer_ext=(".tif", ".nc"),
    out_root=save_dir
):
    """
    从给定 dates (YYYYMMDD 列表) 下载指定网格的日 SIC 产品。
    - 如果 region 非空：优先进入 {product_root}/{yyyy}/{mon3}/{region}；
      若该目录为空或列失败，则自动回退到 {product_root}/{yyyy}/{mon3}
    - 下载文件保存在 out_root/<grid>/ 子目录下（grid 从 product_root 末段提取，如 n6250/n3125）
    """
    grid = os.path.basename(product_root.rstrip("/"))
    out_dir = os.path.join(out_root, grid)
    os.makedirs(out_dir, exist_ok=True)

    ftp = ftp_connect(ftp_host, username, password)
    downloaded = []

    try:
        # dates -> 按月分组，减少FTP往返
        by_month = {}
        for ymd in dates:
            yyyy, mon3 = month_to_mon3(ymd)
            by_month.setdefault((yyyy, mon3), []).append(ymd)

        for (yyyy, mon3), ymd_list in sorted(by_month.items()):
            # 1) 首选：带 region 的月份目录
            month_dir_region = f"{product_root}/{yyyy}/{mon3}/{region}" if region else None
            # 2) 备选：不带 region 的月份目录
            month_dir_plain  = f"{product_root}/{yyyy}/{mon3}"

            month_files = []
            if month_dir_region:
                month_files = ftp_listdir(ftp, month_dir_region)

            used_dir = month_dir_region
            if not month_files:  # region目录为空或不存在，回退到plain
                month_files = ftp_listdir(ftp, month_dir_plain)
                used_dir = month_dir_plain

            if not month_files:
                print(f"[WARN] EMPTY month dir: {month_dir_region or month_dir_plain}")
                continue

            for ymd in sorted(set(ymd_list)):
                fn = pick_best_file(month_files, ymd, prefer_ext=prefer_ext)
                if fn is None:
                    print(f"[MISS] {ymd} not found in {used_dir}")
                    continue
                p = ftp_download(ftp, used_dir, fn, out_dir)
                downloaded.append(p)
    finally:
        try:
            ftp.quit()
        except:
            pass

    print(f"[DONE] {grid} downloaded: {len(downloaded)} files -> {out_dir}")
    return downloaded


# ========= 一键入口（按你的CSV抽日期后，分别下 n6250与n3125） =========
def main():
    dates = extract_dates_from_csv(csv_path, col_with_cs2_name)
    print(f"[INFO] unique dates from CSV: {len(dates)}")

    # A) n6250: 全北极（Arctic），优先下载 GeoTIFF
    n6250_root = "/amsr2/asi_daygrid_swath/n6250"
    batch_download_sic_dates(
        dates=dates,
        product_root=n6250_root,
        region="Arctic",              # 若该目录不存在，会自动回退到无region的月份目录
        prefer_ext=(".tif", ".nc"),
        out_root=save_dir
    )

    # # B) n3125: NorthWestPassage 子区，优先下载 GeoTIFF
    # n3125_root = "/amsr2/asi_daygrid_swath/n3125"
    # batch_download_sic_dates(
    #     dates=dates,
    #     product_root=n3125_root,
    #     region="NorthWestPassage",    # n3125 下常见分区；若不可用将回退到无region月份目录
    #     prefer_ext=(".tif", ".nc"),
    #     out_root=save_dir
    # )

if __name__ == "__main__":
    main()
