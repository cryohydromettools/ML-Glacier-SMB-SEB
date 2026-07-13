# ============================================================
# SMB ML EMULATOR — LOGO PIPELINE (FINAL NO-WEIGHT VERSION)
# ============================================================

# ---------------------------
# Imports
# ---------------------------
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

import xgboost as xgb


# ============================================================
# Paths
# ============================================================
base_dir = Path("/home/lacrio/DATA/CB")
data_dir = base_dir / "pros"
out_dir  = Path("/home/lacrio/ML-Glacier-SMB-SEB/outputs")

dir_models = out_dir / "models"
dir_tables = out_dir / "tables"

for d in [dir_models, dir_tables]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Build pipeline
# ============================================================
def build_pipeline(features):

    return Pipeline([
        ("scaler", ColumnTransformer(
            [("num", StandardScaler(), features)]
        )),
        ("xgb", xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,      # no regularization
            reg_lambda=0.0,     # avoid shrinkage
            random_state=42
        ))
    ])


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

target = "MB"
groups = data["GLACIER"].values


# ============================================================
# LOGO Cross-Validation
# ============================================================
logo = LeaveOneGroupOut()
metrics_all = []

for train_idx, test_idx in logo.split(data, groups=groups):

    train = data.iloc[train_idx]
    test  = data.iloc[test_idx]
    gtest = test["GLACIER"].iloc[0]

    print(f"\nLOGO fold — test glacier: {gtest}")

    Xtr, ytr = train[features], train[target]
    Xte, yte = test[features], test[target]

    pipe = build_pipeline(features)

    # --------------------------------------------------------
    # Hyperparameter grid (anti-variance-compression setup)
    # --------------------------------------------------------
    param_grid = {
        "xgb__n_estimators": [1000, 1500],
        "xgb__learning_rate": [0.02, 0.03],
        "xgb__max_depth": [6, 8],
        "xgb__min_child_weight": [1],
        "xgb__gamma": [0],
    }

    cv = TimeSeriesSplit(n_splits=5)

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0
    )

    grid.fit(Xtr, ytr)

    print("Train min:", ytr.min())
    print("Test  min:", yte.min())

    model = grid.best_estimator_

    # Save fold model
    joblib.dump(model, dir_models / f"xgb_LOGO_{gtest}_{target}.joblib")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    pred = model.predict(Xte)

    std_real = np.std(yte)
    std_pred = np.std(pred)

    print("Std real:", std_real)
    print("Std pred:", std_pred)

    if std_pred < std_real:
        print("⚠ Variance compression detected")

    metrics_all.append({
        "GLACIER": gtest,
        "R": pearsonr(yte, pred)[0],
        "RMSE": np.sqrt(mean_squared_error(yte, pred)),
        "MAE": mean_absolute_error(yte, pred),
        "Bias": np.mean(pred - yte),
        "STD_real": std_real,
        "STD_pred": std_pred
    })


# ============================================================
# Save metrics
# ============================================================
df_metrics = pd.DataFrame(metrics_all)

df_metrics.to_csv(
    dir_tables / f"metrics_LOGO_per_glacier_{target}.csv",
    index=False
)

df_metrics.drop(columns="GLACIER").agg(["mean", "std"]).to_csv(
    dir_tables / f"metrics_LOGO_summary_{target}.csv"
)


# ============================================================
# Train FINAL deployment model (ALL glaciers)
# ============================================================
print("\nTraining final model on ALL glaciers...")

final_model = build_pipeline(features)

# Use best hyperparameters from grid logic
final_model.set_params(
    xgb__n_estimators=1500,
    xgb__learning_rate=0.02,
    xgb__max_depth=8,
    xgb__min_child_weight=1,
    xgb__gamma=0
)

final_model.fit(
    data[features],
    data[target]
)

joblib.dump(
    final_model,
    dir_models / f"xgb_FINAL_all_glaciers_{target}.joblib"
)

print("\n✔ ALL DONE")
print("✔ Models saved")
print("✔ Metrics saved")
