# ============================================================
# SCATTER PLOT — G4 USING ALL-TRAINED MODELS
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, gaussian_kde

import torch
import torch.nn as nn

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# PyTorch MLP (needed for unpickling)
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
        X = np.asarray(X)
        X = torch.tensor(X, dtype=torch.float32).to(device)

        self.model_.eval()
        with torch.no_grad():
            return self.model_(X).cpu().numpy()


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
# Load data — G4 only
# ============================================================
glacier = "G4"

data = pd.read_csv(data_dir / "data_train_test_glas.csv")
data = data[data["GLACIER"] == glacier]
data["time"] = pd.to_datetime(data["time"])
data = data.set_index("time").dropna()

features = [
    "HGT", "SLOPE", "ASPECT",
    "u10", "v10", "t2m",
    "slhf", "str", "sshf", "ssrd", "tp"
]

target = "ME"

X = data[features]
y = data[target]

print(f"\nUsing ALL-trained model for glacier: {glacier}")

# ============================================================
# Load ALL-trained models
# ============================================================

nn_model  = joblib.load(dir_models / f"NN_FINAL_all_glaciers_{target}.joblib")
xgb_model = joblib.load(dir_models / f"xgb_FINAL_all_glaciers_{target}.joblib")

pred_nn  = nn_model.predict(X)
pred_xgb = xgb_model.predict(X)

df_nn  = pd.DataFrame({"OBS": y.values, "SIM": pred_nn})
df_xgb = pd.DataFrame({"OBS": y.values, "SIM": pred_xgb})

# ============================================================
# Plot preparation
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


obs_nn, sim_nn, z_nn, txt_nn = prepare_plot_data(df_nn)
obs_xgb, sim_xgb, z_xgb, txt_xgb = prepare_plot_data(df_xgb)

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
# Plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

lims = [
    min(obs_xgb.min(), sim_xgb.min(), obs_nn.min(), sim_nn.min()),
    max(obs_xgb.max(), sim_xgb.max(), obs_nn.max(), sim_nn.max())
]
print(obs_xgb.min())
# XGB
ax = axes[0]
sc1 = ax.scatter(obs_xgb, sim_xgb, c=z_xgb, s=18, cmap="viridis")
ax.plot(lims, lims, "r--", lw=1)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("COSIPY ME (W m$^{-2}$)")
ax.set_ylabel("Predicted ME (W m$^{-2}$)")
ax.set_title(f"XGB — ALL model ({glacier})")
ax.annotate(txt_xgb, (0.56, 0.04), xycoords="axes fraction",
            bbox=dict(boxstyle="round", fc="white"))

# NN
ax = axes[1]
sc2 = ax.scatter(obs_nn, sim_nn, c=z_nn, s=18, cmap="viridis")
ax.plot(lims, lims, "r--", lw=1)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("COSIPY ME (W m$^{-2}$)")
ax.set_title(f"NN — ALL model ({glacier})")
ax.annotate(txt_nn, (0.56, 0.04), xycoords="axes fraction",
            bbox=dict(boxstyle="round", fc="white"))

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

#fig.tight_layout()
fig.savefig(dir_figures / f"scatter_G4_using_ALL_models_{target}.png",
            dpi=300, bbox_inches="tight")

plt.close(fig)

print("\n✔ Scatter for G4 using ALL model saved.")
