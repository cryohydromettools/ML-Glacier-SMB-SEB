# ============================================================
# SCATTER PLOTS FROM SAVED LOGO MODELS (XGB + PYTORCH NN)
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, gaussian_kde

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
        raise RuntimeError("Inference only.")

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(device)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(X).cpu().numpy()

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

# ============================================================
# Paths
# ============================================================
base_dir = Path("/home/lacrio/DATA/CB")
data_dir = base_dir / "pros"
out_dir  = Path("/home/lacrio/ML-Glacier-SMB-SEB/outputs")

dir_models  = out_dir / "models"
dir_figures = out_dir / "figures"
dir_figures.mkdir(exist_ok=True)

# ============================================================
# Plot utilities
# ============================================================
def prepare_plot_data(df, n=5000):

    obs, sim = df["OBS"].values, df["SIM"].values
    idx = np.random.permutation(len(obs))[:min(n, len(obs))]
    obs, sim = obs[idx], sim[idx]

    z = gaussian_kde(np.vstack([obs, sim]))(np.vstack([obs, sim]))
    s = z.argsort()
    obs, sim, z = obs[s], sim[s], z[s]

    r = pearsonr(df["OBS"], df["SIM"])[0]
    rmse = np.sqrt(mean_squared_error(df["OBS"], df["SIM"]))
    bias = np.mean(df["SIM"] - df["OBS"])

    txt = f"$r={r:.2f}$\n$RMSE={rmse:.3f}$\n$Bias={bias:.3f}$"
    return obs, sim, z, txt


# ============================================================
# Load data
# ============================================================
glaciers = ["G1", "G2", "G3"]
data = pd.read_csv(data_dir / "data_train_test_glas.csv")
data = data[data["GLACIER"].isin(glaciers)]
data["time"] = pd.to_datetime(data["time"])
data = data.set_index("time").dropna()

features = [
    "HGT", "SLOPE", "ASPECT",
    "u10", "v10", "t2m",
    "slhf", "str", "sshf", "ssrd", "tp"
]

target = "ME"
groups = data["GLACIER"].values
logo = LeaveOneGroupOut()

# ============================================================
# Collect predictions
# ============================================================
df_all_nn  = []
df_all_xgb = []

for _, te_idx in logo.split(data, groups=groups):

    test = data.iloc[te_idx]
    gtest = test["GLACIER"].iloc[0]

    print(f"Processing glacier: {gtest}")

    Xte = test[features]
    yte = test[target]

    # ---------------- NN ----------------
    nn_path = dir_models / f"NN_LOGO_{gtest}_{target}.joblib"
    if nn_path.exists():
        nn_model = joblib.load(nn_path)
        pred_nn = nn_model.predict(Xte)

        df_all_nn.append(pd.DataFrame({
            "OBS": yte.values,
            "SIM": pred_nn
        }))

    # ---------------- XGB ----------------
    xgb_path = dir_models / f"xgb_LOGO_{gtest}_{target}.joblib"
    if xgb_path.exists():
        xgb_model = joblib.load(xgb_path)
        pred_xgb = xgb_model.predict(Xte)

        df_all_xgb.append(pd.DataFrame({
            "OBS": yte.values,
            "SIM": pred_xgb
        }))

# Concatenate
df_all_nn  = pd.concat(df_all_nn,  ignore_index=True)
df_all_xgb = pd.concat(df_all_xgb, ignore_index=True)

# Prepare plot data
obs_nn, sim_nn, z_nn, txt_nn = prepare_plot_data(df_all_nn)
obs_xgb, sim_xgb, z_xgb, txt_xgb = prepare_plot_data(df_all_xgb)

# ============================================================
# 1x2 Figure (FIXED colorbar layout)
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 6),
    sharex=True,
    sharey=True,
    constrained_layout=True   # 🔥 THIS FIXES EVERYTHING
)

# ------------------------------------------------------------
# XGBoost panel
# ------------------------------------------------------------
ax = axes[0]
sc1 = ax.scatter(obs_xgb, sim_xgb, c=z_xgb, s=18, cmap="viridis")

lims = [
    min(obs_xgb.min(), sim_xgb.min(), obs_nn.min(), sim_nn.min()),
    max(obs_xgb.max(), sim_xgb.max(), obs_nn.max(), sim_nn.max())
]

ax.plot(lims, lims, "r--", lw=1)
ax.set_xlim(lims)
ax.set_ylim(lims)

ax.set_xlabel("COSIPY ME (W m$^{-2}$)")
ax.set_ylabel("Predicted ME (W m$^{-2}$)")
ax.set_title("XGB — LOGO")

ax.annotate(
    txt_xgb, (0.65, 0.18), xycoords="axes fraction",
    va="top",
    bbox=dict(boxstyle="round", fc="white")
)

# ------------------------------------------------------------
# Neural Network panel
# ------------------------------------------------------------
ax = axes[1]
sc2 = ax.scatter(obs_nn, sim_nn, c=z_nn, s=18, cmap="viridis")

ax.plot(lims, lims, "r--", lw=1)
ax.set_xlim(lims)
ax.set_ylim(lims)

ax.set_xlabel("COSIPY ME (W m$^{-2}$)")
ax.set_title("ANN — LOGO")

ax.annotate(
    txt_nn, (0.65, 0.18), xycoords="axes fraction",
    va="top",
    bbox=dict(boxstyle="round", fc="white")
)

# ------------------------------------------------------------
# Shared colorbar (correct way)
# ------------------------------------------------------------
cbar = fig.colorbar(
    sc1,
    ax=axes,
    location="right",
    shrink=0.85,
    pad=0.02
)

cbar.set_label("Density")

fig.savefig(dir_figures / f"scatter_LOGO_XGB_vs_NN_{target}.png", dpi=300)
plt.close(fig)

print("\n✔ Fixed layout figure saved")

