import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Read data
# ===============================
csv_path = r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\20160112\statistic\monthly_stats_ALL_YEARS.csv"
df = pd.read_csv(csv_path)

# ===============================
# 2. Column names
# ===============================
col_month = "month"   # 如果不是 Month，请改这里
# Sentinel-1 densities (fraction)
s1_lead = "S1_Lead_Density"
s1_floe = "S1_Floe_Density"
s1_floeref = "S1_Floeref_desity"

# CryoSat-2
cs2_floe = "CS2_Floe_Density"
cs2_ambig = "CS2_Ambi_Density"

# ===============================
# 3. Convert to percentage
# ===============================
for c in [s1_lead, s1_floe, s1_floeref, cs2_floe]:
    df[c] = df[c] * 100.0

# Individual class density (NOT cumulative)
df["S1_Refrozen"] = df[s1_floeref] - df[s1_floe]
df["S1_floe_ref"] = df[s1_floeref] 

# ===============================
# 4. X axis (compact months)
# ===============================
months = df[col_month]
x = np.arange(len(months))
width = 0.85
bar_centers = x

fig, ax = plt.subplots(figsize=(8.5, 4.2))

# ===============================
# 5. Stacked bars (cumulative height, individual thickness)
# ===============================

# --- S1 floe (base) ---
ax.bar(
    x,
    df[s1_floe],
    width,
    color="white",
    # edgecolor="black",
    label="S1 Floes"
)

# --- S1 refrozen (stacked on floe) ---
ax.bar(
    x,
    df["S1_Refrozen"],
    width,
    bottom=df[s1_floe],
    color="#b07cc6",
    label="S1 Refrozen"
)

# --- S1 lead (stacked on floe + refrozen) ---
ax.bar(
    x,
    df[s1_lead],
    width,
    bottom= df["S1_floe_ref"],
    color="#2b4f81",
    label="S1 Leads"
)

# ===============================
# 6. CryoSat-2 floes (points, not cumulative)
# ===============================
ax.scatter(
    bar_centers,
    df[cs2_floe],
    marker="x",
    color="red",
    s=60,          # 稍微大一点，视觉更稳
    linewidths=1.5,
    label="CS2 Floes",
    zorder=6
)


# ===============================
# 7. Axis & layout
# ===============================
ax.set_ylim(0, 100)
ax.set_ylabel("Density (%)")
ax.set_xlabel("Month")

ax.set_xticks(x)
ax.set_xticklabels(months)

ax.legend(
    ncol=4,
    frameon=False,
    fontsize=9,
    loc="upper left"
)

plt.tight_layout()
plt.show()