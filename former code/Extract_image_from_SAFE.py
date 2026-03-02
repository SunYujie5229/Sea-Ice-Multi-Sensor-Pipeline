import zipfile
from pathlib import Path

# ===== 用户配置 =====
s1_zip_dir = r"E:\NWP\CS2_S1_matched\Sentinel-1"      # S1 zip所在目录
s1_ew_dir  = r"E:\NWP\CS2_S1_matched\Sentinel-1_EW"   # 输出EW主影像目录

def extract_hh_hv_from_zip(zpath: Path, out_dir: Path):
    with zipfile.ZipFile(zpath, 'r') as zf:
        # 只找 measurement 下的 HH/HV tif
        members = [name for name in zf.namelist()
                   if "measurement" in name.lower() and name.lower().endswith((".tif",".tiff"))
                   and ("hh" in name.lower() or "hv" in name.lower())]

        if not members:
            print(f"[MISS] no HH/HV tif in {zpath.name}")
            return

        for src_member in members:
            pol = "HH" if "hh" in src_member.lower() else "HV"
            out_name = f"{zpath.stem}_{pol}.tif"
            out_path = out_dir / out_name
            if out_path.exists():
                print(f"[SKIP] exists: {out_name}")
                continue
            with zf.open(src_member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            print(f"[OK] extracted {Path(src_member).name} -> {out_name}")

def main():
    out_dir = Path(s1_ew_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for z in Path(s1_zip_dir).glob("*.zip"):
        extract_hh_hv_from_zip(z, out_dir)

if __name__ == "__main__":
    main()
