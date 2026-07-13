# ============================================================
# SMB simulation for a NEW glacier using ML emulators (xarray)
# ============================================================

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import joblib

# ============================================================
# Paths
# ============================================================
data_dir = Path("/home/lacrio/DATA/CB/pros")
out_dir  = Path("/home/lacrio/ML-Glacier-SMB-SEB/outputs")

dir_models  = out_dir / "models"

dir_results = data_dir / "netcdf"
dir_results.mkdir(parents=True, exist_ok=True)

target = "MB"
new_glacier_file = data_dir / "Artesonraju_cpy_era5_l.nc"
out_nc = dir_results / f"SMB_simulated_Artesonraju_{target}.nc"

# ============================================================
# Features (MUST match training)
# ============================================================
features = [
    'HGT', 'SLOPE', 'ASPECT',
    'u10', 'v10', 't2m',
    'slhf', 'str', 'sshf', 'ssrd', 'tp'
]

# ============================================================
# Load trained LOGO models
# ============================================================
model_files = sorted(dir_models.glob(f"xgb_LOGO_*{target}.joblib"))
models = {}

if len(model_files) == 0:
    raise RuntimeError("No trained models found.")

for mf in model_files:
    key = mf.stem.replace("xgb_LOGO_", "")
    models[key] = joblib.load(mf)

print(f"Loaded {len(models)} LOGO models")

# ============================================================
# Load glacier data
# ============================================================
ds = xr.open_dataset(new_glacier_file).sel(time=slice('20000101', '20231201'))
ds = ds.sortby(["lat", "lon"])

assert "time" in ds.dims, "Dataset must contain a time dimension"

# ============================================================
# Prepare output DataArrays (FULL GRID)
# ============================================================
template = ds[features[3]]

smb_models = {
    key: xr.full_like(template, np.nan, dtype="float32")
    for key in models
}

smb_mean = xr.full_like(template, np.nan, dtype="float32")
smb_std  = xr.full_like(template, np.nan, dtype="float32")

# ============================================================
# Loop over time
# ============================================================
for t in ds.time.values:
    print(f"Simulating time {pd.to_datetime(t).date()}")

    ds_t = ds.sel(time=t)

    # ---- DataFrame with spatial identity ----
    df = ds_t[features].to_dataframe()

    df["sp_id"] = list(
        zip(
            df.index.get_level_values("lat"),
            df.index.get_level_values("lon")
        )
    )

    # ---- Drop NaNs safely ----
    valid = df.dropna(subset=features)
    if valid.empty:
        continue

    X = valid[features]

    # ---- Predict with all models ----
    preds = {}
    for key, model in models.items():
        preds[key] = model.predict(X)

    preds_df = pd.DataFrame(preds, index=valid.index)
    preds_df["mean"] = preds_df.mean(axis=1)
    preds_df["std"]  = preds_df.std(axis=1)

    # ========================================================
    # Rebuild FULL GRID safely using spatial ID
    # ========================================================
    for key in models:
        da_out = xr.full_like(
            ds_t[features[0]],
            np.nan,
            dtype="float32"
        )

        for (lat, lon), val in zip(valid["sp_id"], preds_df[key]):
            da_out.loc[dict(lat=lat, lon=lon)] = val

        smb_models[key].loc[dict(time=t)] = da_out

    # ---- Ensemble mean ----
    da_mean = xr.full_like(ds_t[features[0]], np.nan, dtype="float32")
    da_std  = xr.full_like(ds_t[features[0]], np.nan, dtype="float32")

    for (lat, lon), m, s in zip(
        valid["sp_id"],
        preds_df["mean"],
        preds_df["std"]
    ):
        da_mean.loc[dict(lat=lat, lon=lon)] = m
        da_std.loc[dict(lat=lat, lon=lon)]  = s

    smb_mean.loc[dict(time=t)] = da_mean
    smb_std.loc[dict(time=t)]  = da_std

# ============================================================
# Build output Dataset
# ============================================================
out_ds = xr.Dataset()

for key, da in smb_models.items():
    out_ds[f"SMB_{key}"] = da

out_ds["SMB_mean"] = smb_mean
out_ds["SMB_std"]  = smb_std

out_ds = out_ds.assign_coords(ds.coords)

# ============================================================
# Metadata
# ============================================================
out_ds["SMB_mean"].attrs.update({
    "long_name": "Surface mass balance (ML ensemble mean)",
    "units": "m w.e."
})

out_ds["SMB_std"].attrs.update({
    "long_name": "Surface mass balance uncertainty (ML ensemble std)",
    "units": "m w.e."
})

out_ds.attrs["description"] = (
    "Surface mass balance simulated using an ensemble of "
    "leave-one-glacier-out trained XGBoost emulators"
)


# ============================================================
# Save NetCDF
# ============================================================
out_ds.where(ds['MASK']==1).to_netcdf(out_nc)

print("\n✅ SMB simulation completed")
print(f"Saved NetCDF to: {out_nc}")
