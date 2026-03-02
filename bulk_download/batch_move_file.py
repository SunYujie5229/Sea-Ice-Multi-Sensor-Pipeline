# -*- coding: utf-8 -*-

import hashlib
import shutil
from pathlib import Path

# ===== 配置区 =====
SRC_ROOT = Path(r"J:\我的云端硬盘")            # 源根目录（Google Drive 桌面盘）
DEST_DIR = Path(r"F:\NWP\sentinel1 gee\2023")  # 目标目录（新地址）
DRY_RUN  = False                                # True=试跑仅打印，不复制

def unique_name(dest_dir: Path, src_path: Path) -> Path:
    """避免同名覆盖：基于完整源路径生成8位hash附加到文件名。"""
    h = hashlib.sha1(str(src_path).encode("utf-8")).hexdigest()[:8]
    stem, suf = src_path.stem, src_path.suffix
    candidate = dest_dir / f"{stem}_{h}{suf}"
    i = 1
    while candidate.exists():
        candidate = dest_dir / f"{stem}_{h}_{i}{suf}"
        i += 1
    return candidate

def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"源目录不存在：{SRC_ROOT}")
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # 找所有名为 Project_Yujie 的目录
    pj_dirs = [p for p in SRC_ROOT.rglob("*") if p.is_dir() and ("project_yujie" in p.name.lower())]
    print(f"发现 {len(pj_dirs)} 个名为 'Project_Yujie' 的文件夹。")

    total = copied = errors = 0
    for d in pj_dirs:
        tif_list = list(d.glob("*.tif")) + list(d.glob("*.TIF")) + \
                   list(d.glob("*.tiff")) + list(d.glob("*.TIFF"))
        for src in tif_list:
            total += 1
            dst = unique_name(DEST_DIR, src)
            print(f"[COPY] {src}  ->  {dst}")
            if not DRY_RUN:
                try:
                    shutil.copy2(src, dst)  # 保留时间戳等元数据
                    copied += 1
                except Exception as e:
                    errors += 1
                    print(f"  [ERROR] {e}")

    print(f"\n完成：发现 {total} 个tiff；成功复制 {copied}，错误 {errors}。")
    print(f"目标文件夹：{DEST_DIR.resolve()}")

if __name__ == "__main__":
    main()
