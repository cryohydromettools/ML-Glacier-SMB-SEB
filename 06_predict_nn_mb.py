# ============================================================
# SMB simulation for a NEW glacier using NN emulators (xarray)
# ============================================================

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import joblib

from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# PyTorch MLP (IDENTICAL TO TRAINING)
# ============================================================
class MLP(nn.Module):
    def __init__(self, n_features, n_hidden=2, n_units=64, dropout=0.2):
        super().__init__()

        layers = []
        in_f = n_features

        for _ in range(n_hidden):
            layers.append(nn.Linear(in_f, n_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_f = n_units

        layers.append(nn.Linear(in_f, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze()


# ============================================================
# sklearn-compatible TorchRegressor (FOR UNPICKLING)
# ============================================================
class TorchRegressor(BaseEstimator, RegressorMixin):

    def __init__(self,
                 n_features,
                 n_hidden=2,
                 n_units=64,
                 dropout=0.2,
                 lr=1e-3,
                 batch_size=256,
                 epochs=300,
                 weight_decay=1e-4,
                 patience=20):

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.patience = patience

    def fit(self, X, y):
        raise RuntimeError("Inference-only script")

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(device)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(X).cpu().numpy()


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
out_nc = dir_results / f"SMB_simulated_Artesonraju_NN_{target}.nc"

# ============================================================
# Features (MUST match training)
# ============================================================
features = [
    "HGT", "SLOPE", "ASPECT",
    "u10", "v10", "t2m",
    "slhf", "str", "sshf", "ssrd", "tp"
]

# ============================================================
# Load trained LOGO NN models
# ============================================================
model_files = sorted(dir_models.glob(f"NN_LOGO_*{target}.joblib"))
models = {}

if len(model_files) == 0:
    raise RuntimeError("No trained NN models found.")

for mf in model_files:
    key = mf.stem.replace("NN_LOGO_", "")
    models[key] = joblib.load(mf)

print(f"Loaded {len(models)} NN LOGO models")

# ============================================================
# Load glacier data
# ============================================================
ds = xr.open_dataset(new_glacier_file).sel(time=slice("2000-01-01", "2023-12-01"))
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
    "long_name": "Surface mass balance (NN ensemble mean)",
    "units": "m w.e."
})

out_ds["SMB_std"].attrs.update({
    "long_name": "Surface mass balance uncertainty (NN ensemble std)",
    "units": "m w.e."
})

out_ds.attrs["description"] = (
    "Surface mass balance simulated using an ensemble of "
    "leave-one-glacier-out trained neural network emulators"
)

# ============================================================
# Save NetCDF
# ============================================================
out_ds.where(ds["MASK"] == 1).to_netcdf(out_nc)

print("\n✅ SMB simulation completed with NN emulators")
print(f"Saved NetCDF to: {out_nc}")
