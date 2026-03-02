import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Read data
# ===============================
csv_path = r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\20160112\statistic\monthly_stats_ALL_YEARS.csv"
df = pd.read_csv(csv_path)

# ===============================
# 2. Column names (AS PROVIDED)
# ===============================
col_month = "month"  # ⚠️ 如果你的月份列不是 Month，请改这里

# CryoSat-2
cs2_lead = "CS2_Lead_Density"
cs2_floe = "CS2_Floe_Density"
cs2_lead_std = "CS2_Lead_Std"
cs2_floe_std = "CS2_Floe_Std"

# Sentinel-1
s1_lead = "S1_Lead_Density"
s1_floe = "S1_Floe_Density"
s1_leadref = "S1_Leadref_density"

s1_lead_std = "S1_Lead_Std"
s1_floe_std = "S1_Floe_Std"
s1_leadref_std = "S1_Leadref_std"

# ===============================
# 3. Derived quantities
# ===============================
# Refrozen floe density (Sentinel-1 only)
df["S1_Refrozen"] = df[s1_leadref] - df[s1_floe]

# Monthly differences (Sentinel-1 − CryoSat-2)
df["d_lead"] = df[s1_lead] - df[cs2_lead]
df["d_floe"] = df[s1_floe] - df[cs2_floe]

# Error propagation for differences (independent assumption)
df["d_lead_std"] = np.sqrt(df[s1_lead_std]**2 + df[cs2_lead_std]**2)
df["d_floe_std"] = np.sqrt(df[s1_floe_std]**2 + df[cs2_floe_std]**2)

# ===============================
# 4. Plot setup
# ===============================
months = df[col_month]
x = np.arange(len(months))
width = 0.35

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(7, 6),
    sharex=True
)

# ===============================
# Panel (a): Density differences
# ===============================
ax = axes[0]

ax.bar(
    x - width/2,
    df["d_lead"],
    width,
    yerr=df["d_lead_std"],
    capsize=3,
    label="Δ Lead",
    alpha=0.9
)

ax.bar(
    x + width/2,
    df["d_floe"],
    width,
    yerr=df["d_floe_std"],
    capsize=3,
    label="Δ Floe",
    alpha=0.9
)

ax.axhline(0, linewidth=0.8)
ax.set_ylabel("Density difference (%)")
ax.set_title("(a) Sentinel-1 − CryoSat-2 monthly density differences")
ax.legend(frameon=False)

# ===============================
# Panel (b): Sentinel-1 composition
# ===============================
ax2 = axes[1]

ax2.bar(
    x,
    df[s1_lead],
    yerr=df[s1_lead_std],
    capsize=3,
    label="Lead",
    alpha=0.9
)

ax2.bar(
    x,
    df[s1_floe],
    bottom=df[s1_lead],
    yerr=df[s1_floe_std],
    capsize=3,
    label="Floe",
    alpha=0.9
)

ax2.bar(
    x,
    df["S1_Refrozen"],
    bottom=df[s1_lead] + df[s1_floe],
    yerr=df[s1_leadref_std],
    capsize=3,
    label="Refrozen floe",
    alpha=0.9
)

ax2.set_ylabel("Density (%)")
ax2.set_title("(b) Sentinel-1 monthly surface type composition")
ax2.legend(frameon=False)

# ===============================
# Shared x-axis
# ===============================
ax2.set_xticks(x)
ax2.set_xticklabels(months)
ax2.set_xlabel("Month")

# ===============================
# Final layout
# ===============================
plt.tight_layout()
plt.show()
