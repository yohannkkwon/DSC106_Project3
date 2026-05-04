"""
DSC 106 Project 3 – CMIP6 Exploratory Analysis
Generates 6 exploratory figures that motivate the central question:
  "How does projected surface temperature change vary across the globe
   under different emission scenarios, and how do regional trends diverge over time?"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── load data ──────────────────────────────────────────────────────────────────
ts = pd.read_csv("timeseries.csv")
sp = pd.read_csv("spatial.csv")

COLORS = {"ssp245": "#2196F3", "ssp585": "#F44336"}
LABELS = {"ssp245": "SSP2-4.5 (moderate)", "ssp585": "SSP5-8.5 (high)"}

# ── Figure 1: Global mean temperature trajectories ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for scen in ["ssp245", "ssp585"]:
    d = ts[ts["scenario"] == scen].sort_values("year")
    ax.plot(d["year"], d["tas_celsius"], color=COLORS[scen],
            linewidth=2.2, label=LABELS[scen])
    # 10-yr rolling mean overlay
    rm = d["tas_celsius"].rolling(10, center=True).mean()
    ax.plot(d["year"], rm, color=COLORS[scen], linewidth=1, linestyle="--", alpha=0.6)

ax.axvline(2024, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
ax.text(2025, ts["tas_celsius"].min() + 0.2, "2024", color="gray", fontsize=9)
ax.fill_between(ts[ts["scenario"]=="ssp245"]["year"],
                ts[ts["scenario"]=="ssp245"]["tas_celsius"],
                ts[ts["scenario"]=="ssp585"]["tas_celsius"],
                alpha=0.12, color="purple", label="Scenario spread")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Global Mean Surface Temp (°C)", fontsize=12)
ax.set_title("Fig 1 · Global Mean Temperature Projections 2015–2100\n"
             "Dashed = 10-yr rolling mean", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("fig1_global_timeseries.png", dpi=150)
plt.close()
print("Saved fig1_global_timeseries.png")

# ── Figure 2: Warming gap between scenarios ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
d245 = ts[ts["scenario"]=="ssp245"].sort_values("year").set_index("year")
d585 = ts[ts["scenario"]=="ssp585"].sort_values("year").set_index("year")
gap = d585["tas_celsius"] - d245["tas_celsius"]

ax.fill_between(gap.index, gap.values, color="#9C27B0", alpha=0.35, label="SSP585 − SSP245 gap")
ax.plot(gap.index, gap.values, color="#6A0080", linewidth=2)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Temperature Difference (°C)", fontsize=12)
ax.set_title("Fig 2 · Growing Divergence Between Emission Scenarios\n"
             "SSP5-8.5 minus SSP2-4.5 global mean temperature", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("fig2_scenario_gap.png", dpi=150)
plt.close()
print("Saved fig2_scenario_gap.png")

# ── Figure 3: Spatial temperature map – baseline 2020 ─────────────────────────
def pivot_spatial(scenario, decade):
    sub = sp[(sp["scenario"]==scenario) & (sp["decade"]==decade)]
    # pivot: rows=lat descending, cols=lon 0→360
    piv = sub.pivot_table(index="lat", columns="lon", values="tas_celsius", aggfunc="mean")
    return piv.sort_index(ascending=False)

baseline = pivot_spatial("ssp245", 2020)

fig, ax = plt.subplots(figsize=(13, 5))
cmap = plt.cm.RdBu_r
norm = mcolors.TwoSlopeNorm(vmin=-50, vcenter=10, vmax=38)
im = ax.imshow(baseline.values, cmap=cmap, norm=norm,
               extent=[0, 360, -90, 90], aspect="auto")
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Surface Temp (°C)", fontsize=10)
ax.set_xlabel("Longitude", fontsize=11)
ax.set_ylabel("Latitude", fontsize=11)
ax.set_title("Fig 3 · Spatial Surface Temperature – Baseline (2020, SSP2-4.5)\n"
             "Grid-cell decadal mean from CMIP6 ensemble", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("fig3_spatial_2020.png", dpi=150)
plt.close()
print("Saved fig3_spatial_2020.png")

# ── Figure 4: Spatial map 2080 SSP585 (high-warming endpoint) ─────────────────
future_hi = pivot_spatial("ssp585", 2080)

fig, ax = plt.subplots(figsize=(13, 5))
im = ax.imshow(future_hi.values, cmap=cmap, norm=norm,
               extent=[0, 360, -90, 90], aspect="auto")
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Surface Temp (°C)", fontsize=10)
ax.set_xlabel("Longitude", fontsize=11)
ax.set_ylabel("Latitude", fontsize=11)
ax.set_title("Fig 4 · Spatial Surface Temperature – End of Century (2080, SSP5-8.5)\n"
             "Dramatic warming visible in Arctic & continental interiors", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("fig4_spatial_2080_ssp585.png", dpi=150)
plt.close()
print("Saved fig4_spatial_2080_ssp585.png")

# ── Figure 5: Warming anomaly (2080 − 2020) by scenario ───────────────────────
base = sp[sp["decade"]==2020].set_index(["lat","lon","scenario"])["tas_celsius"]
end  = sp[sp["decade"]==2080].set_index(["lat","lon","scenario"])["tas_celsius"]
delta = (end - base).reset_index()
delta.columns = ["lat","lon","scenario","delta_c"]

def pivot_delta(scenario):
    sub = delta[delta["scenario"]==scenario]
    piv = sub.pivot_table(index="lat", columns="lon", values="delta_c", aggfunc="mean")
    return piv.sort_index(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
dnorm = mcolors.TwoSlopeNorm(vmin=0, vcenter=3, vmax=12)
dcmap = plt.cm.OrRd

for ax, scen in zip(axes, ["ssp245", "ssp585"]):
    d = pivot_delta(scen)
    im = ax.imshow(d.values, cmap=dcmap, norm=dnorm,
                   extent=[0, 360, -90, 90], aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("ΔTemp (°C)", fontsize=9)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(f"{LABELS[scen]}\n2080 minus 2020 warming", fontsize=11, fontweight="bold")

fig.suptitle("Fig 5 · Spatial Warming Anomaly (2080 − 2020) by Scenario\n"
             "Polar amplification & regional heterogeneity clearly visible",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("fig5_warming_anomaly.png", dpi=150)
plt.close()
print("Saved fig5_warming_anomaly.png")

# ── Figure 6: Latitudinal temperature profiles over decades ───────────────────
lat_bands = np.arange(-90, 91, 10)

def lat_profile(scenario, decade):
    sub = sp[(sp["scenario"]==scenario) & (sp["decade"]==decade)]
    sub = sub.copy()
    sub["lat_band"] = pd.cut(sub["lat"], bins=lat_bands, labels=lat_bands[:-1]+5)
    return sub.groupby("lat_band", observed=True)["tas_celsius"].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
decades = [2020, 2040, 2060, 2080]
palette = plt.cm.plasma(np.linspace(0.2, 0.9, len(decades)))

for ax, scen in zip(axes, ["ssp245", "ssp585"]):
    for dec, col in zip(decades, palette):
        prof = lat_profile(scen, dec)
        ax.plot(prof.index.astype(float), prof.values,
                color=col, linewidth=1.8, label=str(dec))
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Latitude (°N)", fontsize=11)
    ax.set_ylabel("Mean Surface Temp (°C)", fontsize=11)
    ax.set_title(f"{LABELS[scen]}", fontsize=11, fontweight="bold")
    ax.legend(title="Decade", fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle("Fig 6 · Latitudinal Temperature Profiles by Decade\n"
             "Arctic amplification accelerates under high emissions",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("fig6_lat_profiles.png", dpi=150)
plt.close()
print("Saved fig6_lat_profiles.png")

print("\nAll 6 exploratory figures saved.")
