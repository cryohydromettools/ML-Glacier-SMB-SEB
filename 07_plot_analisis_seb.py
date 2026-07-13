# ======================================================
# SMB comparison — COSIPY | XGB | NN
# 3 x 2 layout with external vertical colorbars
# ======================================================

import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

# ======================================================
# Paths
# ======================================================
out_dir = Path("/home/lacrio/ML-Glacier-SMB-SEB/outputs")
dir_figures = out_dir / "figures"
dir_figures.mkdir(exist_ok=True)

# ======================================================
# Load datasets
# ======================================================
ds_cosipy = xr.open_dataset(
    "/home/lacrio/DATA/CB/pros/Artesonraju_cpy_era5_l.nc"
)

ds_xgb = xr.open_dataset(
    "/home/lacrio/DATA/CB/pros/netcdf/ME_simulated_Artesonraju_ME.nc"
)

ds_nn = xr.open_dataset(
    "/home/lacrio/DATA/CB/pros/netcdf/ME_simulated_Artesonraju_NN_ME.nc"
)

# ======================================================
# Select period
# ======================================================
t_slice = slice("2022-01-01", "2022-12-01")

mb_cosipy = ds_cosipy["ME"].sel(time=t_slice).mean("time")
mb_xgb    = ds_xgb["ME_mean"].sel(time=t_slice).mean("time")
mb_nn     = ds_nn["ME_mean"].sel(time=t_slice).mean("time")

mask = ds_cosipy["MASK"]
# ======================================================
# Differences
# ======================================================
diff_cpy_xgb = mb_cosipy - mb_xgb
diff_cpy_nn  = mb_cosipy - mb_nn
diff_xgb_nn  = mb_xgb    - mb_nn

# ======================================================
# Color settings
# ======================================================
vmin, vmax = -10.0, 140.0
smb_levels = np.arange(vmin, vmax+1, 1)
norm_smb = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

vmax_diff = 30
norm_diff = colors.TwoSlopeNorm(
    vmin=-vmax_diff, vcenter=0.0, vmax=vmax_diff
)

# -----------------------------------------------------------------------------
# Plot configuration
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

# ======================================================
# Figure
# ======================================================
fig, axes = plt.subplots(
    nrows=2, ncols=3,
    figsize=(14, 9)
)

label_fs = 13
tick_fs  = 11
title_fs = 14
cbar_fs  = 12

# ======================================================
# --- ROW 1: SMB ---
# ======================================================
p0 = axes[0, 0].contourf(
    mb_cosipy.lon, mb_cosipy.lat, mb_cosipy.where(mask==1),
    cmap="RdBu_r", levels=smb_levels, norm=norm_smb, extend="both"
)
axes[0, 0].set_title("COSIPY SEB (2022)", fontsize=title_fs)
axes[0, 0].set_ylabel("Latitude", fontsize=label_fs)

p1 = axes[0, 1].contourf(
    mb_xgb.lon, mb_xgb.lat, mb_xgb.where(mask==1),
    cmap="RdBu_r", levels=smb_levels, norm=norm_smb, extend="both"
)
axes[0, 1].set_title("XGB SEB (2022)", fontsize=title_fs)

p2 = axes[0, 2].contourf(
    mb_nn.lon, mb_nn.lat, mb_nn.where(mask==1),
    cmap="RdBu_r", levels=smb_levels, norm=norm_smb, extend="both"
)
axes[0, 2].set_title("ANN SEB (2022)", fontsize=title_fs)

# ======================================================
# --- ROW 2: Differences ---
# ======================================================
p3 = axes[1, 0].pcolormesh(
    diff_cpy_xgb.lon, diff_cpy_xgb.lat, diff_cpy_xgb,
    cmap="bwr", norm=norm_diff
)
axes[1, 0].set_title("COSIPY − XGB", fontsize=title_fs)
axes[1, 0].set_ylabel("Latitude", fontsize=label_fs)

p4 = axes[1, 1].pcolormesh(
    diff_cpy_nn.lon, diff_cpy_nn.lat, diff_cpy_nn,
    cmap="bwr", norm=norm_diff
)
axes[1, 1].set_title("COSIPY − ANN", fontsize=title_fs)

p5 = axes[1, 2].pcolormesh(
    diff_xgb_nn.lon, diff_xgb_nn.lat, diff_xgb_nn,
    cmap="bwr", norm=norm_diff
)
axes[1, 2].set_title("XGB − ANN", fontsize=title_fs)

# ======================================================
# Axis formatting — ticks only left & bottom
# ======================================================
for i in range(2):
    for j in range(3):
        ax = axes[i, j]

        # Ticks
        ax.tick_params(axis="both", labelsize=tick_fs)

        # Only left column → Y labels
        if j == 0:
            ax.set_ylabel("Latitude", fontsize=label_fs)
        else:
            ax.set_yticklabels([])

        # Only bottom row → X labels
        if i == 1:
            ax.set_xlabel("Longitude", fontsize=label_fs)
            ax.tick_params(axis="x")
        else:
            ax.set_xticklabels([])

# ======================================================
# --- EXTERNAL COLORBARS (NO OVERLAP) ---
# ======================================================

# SMB colorbar (top row)
cax_smb = fig.add_axes([0.93, 0.55, 0.02, 0.33])
cbar_smb = fig.colorbar(p0, cax=cax_smb, extend="both")
cbar_smb.set_label("SEB (W m$^{-2}$)", fontsize=cbar_fs)
cbar_smb.ax.tick_params(labelsize=tick_fs)
#cbar_smb.set_ticks(
#    [-60.0, -40.0, -20, 0.0, 20, 40, 60 ]
#)

# Difference colorbar (bottom row)
cax_diff = fig.add_axes([0.93, 0.12, 0.02, 0.33])
cbar_diff = fig.colorbar(p3, cax=cax_diff, extend="both")
cbar_diff.set_label("Δ SEB (W m$^{-2}$)", fontsize=cbar_fs)
cbar_diff.ax.tick_params(labelsize=tick_fs)

# ======================================================
# Layout + save
# ======================================================
fig.subplots_adjust(
    left=0.07, right=0.90,
    bottom=0.08, top=0.92,
    wspace=0.15, hspace=0.25
)

fig.savefig(
    dir_figures / "ME_Arteson_2022_3x2_COSIPY_XGB_NN.png",
    dpi=300,
    facecolor="w",
    bbox_inches="tight"
)

#plt.show()

