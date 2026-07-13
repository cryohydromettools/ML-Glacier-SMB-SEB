import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ======================================================
# Paths
# ======================================================
out_dir = Path("/home/lacrio/ML-Glacier-SMB-SEB/outputs")
dir_figures = out_dir / "figures"
dir_figures.mkdir(exist_ok=True)

# ======================================================
# Load datasets
# ======================================================

# COSIPY (contains MB and ME)
ds_cosipy = xr.open_dataset(
    "/home/lacrio/DATA/CB/pros/Artesonraju_cpy_era5_l.nc"
)

# SMB (ML)
ds_xgb = xr.open_dataset(
    "/home/lacrio/DATA/CB/pros/netcdf/SMB_simulated_Artesonraju_MB.nc"
)

ds_nn = xr.open_dataset(
    "/home/lacrio/DATA/CB/pros/netcdf/SMB_simulated_Artesonraju_NN_MB.nc"
)

# ME (ML)
ds_xgb_me = xr.open_dataset(
    "/home/lacrio/DATA/CB/pros/netcdf/ME_simulated_Artesonraju_ME.nc"
)

ds_nn_me = xr.open_dataset(
    "/home/lacrio/DATA/CB/pros/netcdf/ME_simulated_Artesonraju_NN_ME.nc"
)


# ======================================================
# Matplotlib style
# ======================================================

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 22,
})

# ======================================================
# Time period
# ======================================================
t_full = slice("2000-01-01", "2023-12-31")



# ======================================================
# YEARLY SMB
# ======================================================

mb_cpy = (
    ds_cosipy["MB"]
    .sel(time=t_full)
    .resample(time="A-AUG")
    .sum("time")
    .mean(("lat", "lon"))
)

mb_xgb = (
    ds_xgb["SMB_mean"]
    .sel(time=t_full)
    .resample(time="A-AUG")
    .sum("time")
    .mean(("lat", "lon"))
)

mb_nn = (
    ds_nn["SMB_mean"]
    .sel(time=t_full)
    .resample(time="A-AUG")
    .sum("time")
    .mean(("lat", "lon"))
)

cpy_year = mb_cpy.to_series()
xgb_year = mb_xgb.to_series()
nn_year = mb_nn.to_series()

cpy_year.index = cpy_year.index.year
xgb_year.index = xgb_year.index.year
nn_year.index = nn_year.index.year

# ======================================================
# YEARLY ME (Surface Energy Balance)
# ======================================================

me_cpy = (
    ds_cosipy["ME"]
    .sel(time=t_full)
    .resample(time="A-AUG")
    .mean("time")
    .mean(("lat", "lon"))
)

me_xgb = (
    ds_xgb_me["ME_mean"]
    .sel(time=t_full)
    .resample(time="A-AUG")
    .mean("time")
    .mean(("lat", "lon"))
)

me_nn = (
    ds_nn_me["ME_mean"]
    .sel(time=t_full)
    .resample(time="A-AUG")
    .mean("time")
    .mean(("lat", "lon"))
)

me_cpy = me_cpy.to_series()
me_xgb = me_xgb.to_series()
me_nn = me_nn.to_series()

me_cpy.index = me_cpy.index.year
me_xgb.index = me_xgb.index.year
me_nn.index = me_nn.index.year

# ======================================================
# Years
# ======================================================

years_all = xgb_year.index.values
years_cpy = cpy_year.index.values

x = np.arange(len(years_all))
width = 0.28

# ======================================================
# Figure
# ======================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 10),
    sharex=True
)

# ======================================================
# (a) Surface Energy Balance
# ======================================================

ax = axes[0]

ax.bar(
    x - width/2,
    me_xgb.values,
    width,
    label="XGB",
    color="tab:blue"
)

ax.bar(
    x + width/2,
    me_nn.values,
    width,
    label="ANN",
    color="tab:orange"
)

x_cpy = np.where(np.isin(years_all, me_cpy.index.values))[0]

ax.bar(
    x_cpy,
    me_cpy.values,
    width * 0.9,
    label="COSIPY",
    color="tab:green"
)

ax.set_ylabel("Annual ME (W m$^{-2}$)", fontsize=12)
ax.set_title("(a) Surface Energy Balance", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.legend(ncol=3)

# ======================================================
# (b) Surface Mass Balance
# ======================================================

ax = axes[1]

ax.bar(
    x - width/2,
    xgb_year.values,
    width,
    label="XGB",
    color="tab:blue"
)

ax.bar(
    x + width/2,
    nn_year.values,
    width,
    label="ANN",
    color="tab:orange"
)

x_cpy = np.where(np.isin(years_all, years_cpy))[0]

ax.bar(
    x_cpy,
    cpy_year.values,
    width * 0.9,
    label="COSIPY",
    color="tab:green"
)

ax.set_ylabel("Annual SMB (m w.e.)", fontsize=12)
ax.set_title("(b) Surface Mass Balance", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

ax.set_xticks(x)
ax.set_xticklabels(years_all, rotation=90)
ax.set_xlabel("Hydrological Year (Sep–Aug)", fontsize=12)

# ======================================================
# Save
# ======================================================

fig.tight_layout()

fig.savefig(
    dir_figures / "Figure_Annual_ME_SMB_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("✔ Figure saved:", dir_figures / "Figure_Annual_ME_SMB_Comparison.png")