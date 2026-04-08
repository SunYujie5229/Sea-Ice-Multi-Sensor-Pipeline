"""
statistic_SIC_binned.py
========================
针对 v3 pipeline 输出（density_SIC_binned_v3_*.csv）的统计分析脚本。

两条分析路线
-----------
Route A  →  月份 × SIC bin 联合统计
            groupby(month, sic_bin)：count 聚合 → 重算 density
            输出：monthly_SIC_stats_ALL_YEARS.csv / .xlsx

Route B  →  纯 SIC 关系（跨年份、跨月份）
            groupby(sic_bin)：count 聚合 → 重算 density → 拟合 Δdensity = f(SIC)
            输出：sic_density_relation.csv / .xlsx
                  sic_bias_fit_coefficients.csv
                  sic_bias_curve.png

核心原则
-------
  ✅  先 sum(count)，再算 density
  ✅  不用 mean(density)（scene 大小不同，mean 会偏）
  ✅  拟合时同时输出：二次多项式 + UnivariateSpline（两者都保留供选择）
  ✅  物理约束：SIC > 95 时 Δdensity 线性趋零
"""

import os
import glob
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import UnivariateSpline

warnings.filterwarnings("ignore")


# 1. 设置输入路径（匹配你图片中的 statistic 文件夹下的 csv）
INPUT_PATTERN = r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\201604021\statistic\density_SIC_binned_*.csv"

# 所有输出写入这个目录
OUTPUT_DIR = r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\201604021\statistic"

# Spline 平滑度（越小越贴数据，越大越平滑）
SPLINE_SMOOTHING = 0.0005

# 物理约束：SIC > 此阈值时 Δdensity 线性趋零
CONSTRAINT_SIC_UPPER = 95.0


# ========================= SIC bin 工具 =========================

def bin_to_mid(b: str) -> float:
    """
    "0-10"  → 5.0
    "90-100"→ 95.0
    兼容旧格式 "<60" → 30.0（v3 不再生成此标签，保留兼容）
    """
    b = str(b).strip()
    if "<" in b:
        return 30.0
    if "-" not in b:
        try:
            return float(b)
        except ValueError:
            return np.nan
    lo, hi = b.split("-", 1)
    return (float(lo) + float(hi)) / 2.0


# ========================= 读取 & 预处理 =========================

# 必须存在的计数列（分母必须有）
REQUIRED_COLS = [
    "scene_name", "sic_bin", "month",
    "count_CS2_lead", "count_CS2_ice", "count_CS2_refrozen",
    "count_CS2_ambi", "total_CS2_overlap",
    "S1_lead_pixels", "S1_ice_pixels", "S1_refrozen_pixels", "S1_total_pixels",
    "n_CS2_points",
]




def load_all_csv(pattern: str) -> pd.DataFrame:
    """
    读取所有年度 CSV，通过 scene_name 或文件名补全日期信息。
    """
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"未找到匹配的文件: {pattern}")
    
    print(f"找到 {len(files)} 个 CSV 文件。")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        
        # --- 步骤 A: 提取年份 (作为保底) ---
        # 从文件名提取数字，例如从 "density_SIC_binned_2015.csv" 提取 2015
        file_year = None
        year_match = re.search(r'(\d{4})', os.path.basename(f))
        if year_match:
            file_year = int(year_match.group(1))

        # --- 步骤 B: 处理 scene_name (获取精确日期) ---
        if "scene_name" in df.columns:
            # 提取格式如 _20210520T 中的日期
            extracted = df["scene_name"].str.extract(r"_(\d{8})T", expand=False)
            dates = pd.to_datetime(extracted, format="%Y%m%d", errors="coerce")
            
            # 如果 scene_name 解析成功则用解析的，否则用文件名的年份
            df["year"] = dates.dt.year.fillna(file_year)
            # 如果原表有 month 且有效则保留，否则从日期提取
            if "month" in df.columns:
                df["month"] = df["month"].fillna(dates.dt.month)
            else:
                df["month"] = dates.dt.month
        else:
            # 如果没有 scene_name，则强制使用文件名年份
            if "year" not in df.columns:
                df["year"] = file_year

        # --- 步骤 C: 筛选列并整合 ---
        # 确保 REQUIRED_COLS 包含你需要的列（如 'sic_bin', 'density' 等）
        # 这里加上 'year' 和 'month'
        current_required = list(set(REQUIRED_COLS + ["year", "month", "sic_bin"]))
        keep = [c for c in current_required if c in df.columns]
        
        frames.append(df[keep])

    # 合并所有年度数据
    merged = pd.concat(frames, ignore_index=True)

    # --- 步骤 D: 数据清洗与计算 ---
    # 移除关键列为空的行
    merged = merged.dropna(subset=["month", "sic_bin", "year"])
    
    # 强制转换类型
    merged["month"] = merged["month"].astype(int)
    merged["year"] = merged["year"].astype(int)

    # 计算 sic_mid (用于后续排序或拟合)
    if "sic_bin" in merged.columns:
        # 确保你已经定义了 bin_to_mid 函数
        merged["sic_mid"] = merged["sic_bin"].apply(bin_to_mid)

    # 按年份和月份排序，方便后续绘图
    merged = merged.sort_values(["year", "month"]).reset_index(drop=True)

    print(f"数据处理完成！")
    print(f"时间范围: {merged['year'].min()}年 到 {merged['year'].max()}年")
    print(f"总行数: {len(merged):,}")
    
    return merged

# 调用示例
# df_final = load_all_csv(INPUT_PATTERN)


# ========================= 核心聚合：count → density =========================

def count_agg_and_density(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    按 group_cols 对计数列 sum，然后重算 density 和 Δdensity。
    这是唯一正确的聚合方式（不用 mean(density)）。
    """
    agg = (
        df.groupby(group_cols, sort=True)
        .agg(
            # CS2 计数
            CS2_lead      = ("count_CS2_lead",     "sum"),
            CS2_floe      = ("count_CS2_ice",       "sum"),
            CS2_refrozen  = ("count_CS2_refrozen",  "sum"),
            CS2_ambi      = ("count_CS2_ambi",      "sum"),
            CS2_total     = ("total_CS2_overlap",   "sum"),
            # S1 像素
            S1_lead       = ("S1_lead_pixels",      "sum"),
            S1_floe       = ("S1_ice_pixels",       "sum"),
            S1_refrozen   = ("S1_refrozen_pixels",  "sum"),
            S1_total      = ("S1_total_pixels",     "sum"),
            # 样本量
            n_CS2_points  = ("n_CS2_points",        "sum"),
            n_scenes      = ("scene_name",          "count"),
        )
        .reset_index()
    )

    # ---- CS2 density ----
    t = agg["CS2_total"].replace(0, np.nan)
    agg["density_CS2_lead"]     = agg["CS2_lead"]     / t
    agg["density_CS2_floe"]     = agg["CS2_floe"]     / t
    agg["density_CS2_leadref"]  = (agg["CS2_lead"] + agg["CS2_refrozen"]) / t
    agg["density_CS2_floeref"]  = (agg["CS2_floe"] + agg["CS2_refrozen"]) / t
    agg["density_CS2_ambi"]     = agg["CS2_ambi"]     / t

    # ---- S1 density ----
    p = agg["S1_total"].replace(0, np.nan)
    agg["density_S1_lead"]      = agg["S1_lead"]      / p
    agg["density_S1_floe"]      = agg["S1_floe"]      / p
    agg["density_S1_leadref"]   = (agg["S1_lead"] + agg["S1_refrozen"]) / p
    agg["density_S1_floeref"]   = (agg["S1_floe"] + agg["S1_refrozen"]) / p

    # ---- Δdensity ----
    agg["diff_lead"]     = agg["density_CS2_lead"]    - agg["density_S1_lead"]
    agg["diff_floe"]     = agg["density_CS2_floe"]    - agg["density_S1_floe"]
    agg["diff_leadref"]  = agg["density_CS2_leadref"] - agg["density_S1_leadref"]
    agg["diff_floeref"]  = agg["density_CS2_floeref"] - agg["density_S1_floeref"]

    return agg


# ========================= Route A：月份 × SIC bin =========================

def route_A_monthly_sic(df: pd.DataFrame, output_dir: str):
    """
    groupby(month, sic_bin) → count agg → density
    每个月有 ≤10 行（对应各 SIC bin）。
    """
    print("\n[Route A] Monthly × SIC bin statistics...")

    result = count_agg_and_density(df, group_cols=["month", "sic_bin"])

    # 加 sic_mid 以便后续排序/绘图
    result["sic_mid"] = result["sic_bin"].apply(bin_to_mid)
    result = result.sort_values(["month", "sic_mid"])

    # ---- 输出 ----
    csv_path = os.path.join(output_dir, "monthly_SIC_stats_ALL_YEARS.csv")
    result.to_csv(csv_path, index=False)
    print(f"  [OK] CSV: {csv_path}")

    xlsx_path = os.path.join(output_dir, "monthly_SIC_stats_ALL_YEARS.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # 总表
        result.to_excel(writer, sheet_name="ALL", index=False)
        # 每个月一个 sheet
        for month, grp in result.groupby("month"):
            grp.to_excel(writer, sheet_name=f"M{int(month):02d}", index=False)
    print(f"  [OK] Excel (per-month sheets): {xlsx_path}")

    return result


# ========================= Route B：纯 SIC 关系 =========================

def _apply_physical_constraint(sic_arr: np.ndarray, val_arr: np.ndarray,
                                upper: float = CONSTRAINT_SIC_UPPER) -> np.ndarray:
    """
    SIC > upper 时，Δdensity 线性衰减至 SIC=100 处为 0。
    原理：val_constrained = val * max(0, (100 - sic) / (100 - upper))
    """
    factor = np.clip((100.0 - sic_arr) / (100.0 - upper), 0.0, 1.0)
    return val_arr * factor


def fit_sic_relation(agg: pd.DataFrame, target_col: str, output_dir: str,
                     label: str = "lead"):
    """
    对 sic_mid vs target_col 做两种拟合：
      1. 二次多项式（快，解析式可写入论文）
      2. UnivariateSpline（平滑，带物理约束）
    输出：系数 CSV + 拟合曲线图。
    """
    sub = agg.dropna(subset=["sic_mid", target_col]).copy()
    sub = sub.sort_values("sic_mid")
    x = sub["sic_mid"].to_numpy(dtype=float)
    y = sub[target_col].to_numpy(dtype=float)

    if len(x) < 3:
        print(f"  [SKIP] Not enough data points for {label} fit ({len(x)} pts).")
        return

    # ---- 1. 二次多项式 ----
    coef = np.polyfit(x, y, deg=2)
    poly = np.poly1d(coef)
    print(f"  [{label}] Poly2 fit: {poly}")

    # ---- 2. Spline ----
    try:
        spline = UnivariateSpline(x, y, k=min(3, len(x)-1), s=SPLINE_SMOOTHING)
    except Exception as e:
        print(f"  [{label}] Spline failed: {e}. Using poly only.")
        spline = None

    # ---- 拟合曲线（细密 x 轴）----
    x_fine = np.linspace(x.min(), x.max(), 300)
    y_poly  = poly(x_fine)
    y_spline_raw = spline(x_fine) if spline else None

    # 物理约束版 spline
    y_spline_constrained = None
    if y_spline_raw is not None:
        y_spline_constrained = _apply_physical_constraint(x_fine, y_spline_raw)

    # ---- 图 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color="steelblue", zorder=5, s=60,
               label=f"Observed Δdensity ({label})")
    ax.plot(x_fine, y_poly, "r--", linewidth=1.8, label="Poly-2 fit")
    if y_spline_constrained is not None:
        ax.plot(x_fine, y_spline_constrained, "g-", linewidth=2.0,
                label=f"Spline (constrained, s={SPLINE_SMOOTHING})")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("SIC bin midpoint (%)", fontsize=12)
    ax.set_ylabel(f"Δdensity ({label})  [CS2 − S1]", fontsize=12)
    ax.set_title(f"Bias = f(SIC)  —  {label}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"sic_bias_curve_{label}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  [OK] Curve plot: {fig_path}")

    # ---- 系数 CSV ----
    coef_df = pd.DataFrame({
        "term":  ["a (x²)", "b (x)", "c (const)"],
        "coef":  coef,
    })
    coef_df["poly_expr"] = f"{coef[0]:.6f}·SIC² + {coef[1]:.6f}·SIC + {coef[2]:.6f}"
    coef_path = os.path.join(output_dir, f"poly2_coef_{label}.csv")
    coef_df.to_csv(coef_path, index=False)
    print(f"  [OK] Poly coefs: {coef_path}")

    return poly, spline


def route_B_sic_relation(df: pd.DataFrame, output_dir: str):
    """
    groupby(sic_bin) → count agg → density → 拟合 Δdensity = f(SIC)
    """
    print("\n[Route B] Pure SIC → density relation (all years merged)...")

    agg = count_agg_and_density(df, group_cols=["sic_bin"])
    agg["sic_mid"] = agg["sic_bin"].apply(bin_to_mid)
    agg = agg.sort_values("sic_mid").reset_index(drop=True)

    # ---- 输出汇总表 ----
    csv_path = os.path.join(output_dir, "sic_density_relation.csv")
    agg.to_csv(csv_path, index=False)
    print(f"  [OK] SIC relation CSV: {csv_path}")

    xlsx_path = os.path.join(output_dir, "sic_density_relation.xlsx")
    agg.to_excel(xlsx_path, index=False)
    print(f"  [OK] SIC relation Excel: {xlsx_path}")

    # ---- 拟合四条偏差曲线 ----
    print("\n  Fitting bias curves...")
    fits = {}
    for col, label in [
        ("diff_lead",    "lead"),
        ("diff_floe",    "floe"),
        ("diff_leadref", "leadref"),
        ("diff_floeref", "floeref"),
    ]:
        result = fit_sic_relation(agg, col, output_dir, label=label)
        if result:
            fits[label] = result

    # ---- 综合对比图（4 条 Δdensity in one figure）----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    pairs = [
        ("diff_lead",    "density_CS2_lead",    "density_S1_lead",    "Lead"),
        ("diff_floe",    "density_CS2_floe",    "density_S1_floe",    "Floe"),
        ("diff_leadref", "density_CS2_leadref", "density_S1_leadref", "Lead+Ref"),
        ("diff_floeref", "density_CS2_floeref", "density_S1_floeref", "Floe+Ref"),
    ]
    for ax, (diff_col, cs2_col, s1_col, title) in zip(axes.flat, pairs):
        sub = agg.dropna(subset=["sic_mid", diff_col, cs2_col, s1_col])
        ax.plot(sub["sic_mid"], sub[cs2_col], "o-", color="#1f77b4",
                label="CS2", linewidth=1.5, markersize=5)
        ax.plot(sub["sic_mid"], sub[s1_col],  "s-", color="#d62728",
                label="S1",  linewidth=1.5, markersize=5)
        ax2 = ax.twinx()
        ax2.bar(sub["sic_mid"], sub[diff_col], width=6,
                color="gray", alpha=0.35, label="Δ (CS2−S1)")
        ax2.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax2.set_ylabel("Δdensity", fontsize=9, color="gray")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Density", fontsize=9)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")

    for ax in axes[1]:
        ax.set_xlabel("SIC bin midpoint (%)", fontsize=10)
    fig.suptitle("CS2 vs S1 Density by SIC bin  (all years merged)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    overview_path = os.path.join(output_dir, "sic_density_overview.png")
    plt.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [OK] Overview figure: {overview_path}")

    return agg, fits


# ========================= 便捷函数：生成可调用的 f_sic =========================

def make_f_sic(poly, spline=None, use_spline: bool = True,
               upper: float = CONSTRAINT_SIC_UPPER):
    """
    返回一个可调用的偏差函数 f_sic(sic) → Δdensity。
    - 输入 sic 兼容 0-1 和 0-100（自动识别）
    - 超出 [0,100] 的值被 clip 到 [0,100]
    - SIC > upper 时物理约束线性衰减至 0
    """
    def f_sic(sic):
        sic = np.asarray(sic, dtype=float)
        # 自动兼容 0-1 尺度
        if np.nanmax(sic) <= 1.0:
            sic = sic * 100.0
        sic = np.clip(sic, 0.0, 100.0)

        if use_spline and spline is not None:
            val = spline(sic)
        else:
            val = poly(sic)

        # 物理约束
        val = _apply_physical_constraint(sic, val, upper)
        return float(val) if val.ndim == 0 else val

    return f_sic


# ========================= MAIN =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("SIC-binned Density Statistics  (v3 pipeline)")
    print("=" * 70)
    print(f"Input  : {INPUT_PATTERN}")
    print(f"Output : {OUTPUT_DIR}\n")

    # ---- 读取所有 CSV ----
    df = load_all_csv(INPUT_PATTERN)

    # ---- Route A：月份 × SIC bin ----
    monthly_sic = route_A_monthly_sic(df, OUTPUT_DIR)

    # ---- Route B：纯 SIC 关系 ----
    sic_agg, fits = route_B_sic_relation(df, OUTPUT_DIR)

    # ---- 打印示例：f_sic 可用性验证 ----
    if "lead" in fits:
        poly_lead, spline_lead = fits["lead"]
        f_lead = make_f_sic(poly_lead, spline_lead, use_spline=True)
        print("\n[f_sic demo]  Δdensity_lead at SIC = 70, 80, 90, 95, 100:")
        for v in [70, 80, 90, 95, 100]:
            print(f"  SIC={v:3d} → Δdensity_lead = {f_lead(v):+.4f}")

    print("\nAll Done! ✓")
    print(f"Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
