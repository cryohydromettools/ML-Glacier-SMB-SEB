# ============================================================
# SMB ML EMULATOR — FULL LOGO PIPELINE (PYTORCH NN)
# ============================================================

# ---------------------------
# Imports
# ---------------------------
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.stats import pearsonr, gaussian_kde

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# Paths
# ---------------------------
base_dir = Path("/home/lacrio/DATA/CB")
data_dir = base_dir / "pros"
out_dir = Path("/home/lacrio/ML-Glacier-SMB-SEB/outputs")

dir_models  = out_dir / "models"
dir_figures = out_dir / "figures"
dir_tables  = out_dir / "tables"

for d in [dir_models, dir_figures, dir_tables]:
    d.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Utility functions
# ---------------------------
def get_series(model, X, y):
    return pd.DataFrame({
        "OBS": y.values,
        "SIM": model.predict(X)
    }).dropna()

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

# ---------------------------
# PyTorch MLP
# ---------------------------
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

# ---------------------------
# sklearn-compatible wrapper
# ---------------------------
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

        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y.values, dtype=torch.float32).to(device)

        self.model_ = MLP(
            self.n_features,
            self.n_hidden,
            self.n_units,
            self.dropout
        ).to(device)

        opt = optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        n = len(X)
        idx = np.arange(n)
        np.random.shuffle(idx)

        split = int(0.8 * n)
        tr_idx, val_idx = idx[:split], idx[split:]

        Xtr, Xval = X[tr_idx], X[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]

        best_loss = np.inf
        patience_count = 0

        for _ in range(self.epochs):
            self.model_.train()

            for i in range(0, len(Xtr), self.batch_size):
                xb = Xtr[i:i+self.batch_size]
                yb = ytr[i:i+self.batch_size]

                opt.zero_grad()
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(Xval), yval).item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = self.model_.state_dict()
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    break

        self.model_.load_state_dict(best_state)
        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(device)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(X).cpu().numpy()

# ---------------------------
# Pipeline
# ---------------------------
def build_pipeline(features):
    return Pipeline([
        ("scaler", ColumnTransformer(
            [("num", StandardScaler(), features)]
        )),
        ("nn", TorchRegressor(
            n_features=len(features)
        ))
    ])

# ---------------------------
# Load data
# ---------------------------
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

# ---------------------------
# LOGO CV
# ---------------------------
logo = LeaveOneGroupOut()

metrics_all = []
df_all = []

for tr_idx, te_idx in logo.split(data, groups=groups):

    train = data.iloc[tr_idx]
    test  = data.iloc[te_idx]
    gtest = test["GLACIER"].iloc[0]

    print(f"\nLOGO — test glacier: {gtest}")

    Xtr, ytr = train[features], train[target]
    Xte, yte = test[features], test[target]

    pipe = build_pipeline(features)

    param_grid = {
        "nn__n_hidden": [2, 3],
        "nn__n_units": [64, 128],
        "nn__dropout": [0.1, 0.2],
        "nn__lr": [1e-3, 5e-4]
    }

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        verbose=0
    )

    grid.fit(Xtr, ytr)
    model = grid.best_estimator_

    joblib.dump(model, dir_models / f"NN_LOGO_{gtest}_{target}.joblib")

    pred = model.predict(Xte)

    metrics_all.append({
        "GLACIER": gtest,
        "R": pearsonr(yte, pred)[0],
        "RMSE": np.sqrt(mean_squared_error(yte, pred)),
        "MAE": mean_absolute_error(yte, pred),
        "Bias": np.mean(pred - yte)
    })

    df_fold = get_series(model, Xte, yte)
    df_fold["GLACIER"] = gtest
    df_all.append(df_fold)

# ---------------------------
# Save tables
# ---------------------------
df_metrics = pd.DataFrame(metrics_all)
df_metrics.to_csv(dir_tables / f"metrics_LOGO_NN_per_glacier_{target}.csv", index=False)

df_metrics.drop(columns="GLACIER").agg(["mean","std"]).to_csv(
    dir_tables / f"metrics_LOGO_NN_summary_{target}.csv"
)

# ---------------------------
# Final model
# ---------------------------
final_model = build_pipeline(features)
final_model.fit(data[features], data[target])
joblib.dump(final_model, dir_models / f"NN_FINAL_all_glaciers_{target}.joblib")

print("\n✔ ALL DONE — PYTORCH NN")
