# ============================================================
# SMB ML EMULATOR — LOGO PIPELINE (SAVE MODELS ONLY)
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

# ---------------------------
# Paths
# ---------------------------
base_dir = Path("/home/lacrio/DATA/CB")
data_dir = base_dir / "pros"
out_dir = Path("/home/lacrio/ML-Glacier-SMB-SEB/outputs")

dir_models  = out_dir / "models"
dir_tables  = out_dir / "tables"

for d in [dir_models, dir_tables]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Build pipeline
# ---------------------------
def build_pipeline(features):
    return Pipeline([
        ("scaler", ColumnTransformer(
            [("num", StandardScaler(), features)]
        )),
        ("xgb", xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.1,  # <-- correcto
            tree_method="hist",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
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

target = "MB"
groups = data["GLACIER"].values

# ---------------------------
# LOGO Cross-Validation
# ---------------------------
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

    param_grid = {
        "xgb__n_estimators": [300, 600],
        "xgb__learning_rate": [0.03, 0.05],
        "xgb__max_depth": [3, 4],
        "xgb__min_child_weight": [3, 5],
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

    # Evaluate
    pred = model.predict(Xte)

    metrics_all.append({
        "GLACIER": gtest,
        "R": pearsonr(yte, pred)[0],
        "RMSE": np.sqrt(mean_squared_error(yte, pred)),
        "MAE": mean_absolute_error(yte, pred),
        "Bias": np.mean(pred - yte)
    })

# ---------------------------
# Save metrics
# ---------------------------
df_metrics = pd.DataFrame(metrics_all)
df_metrics.to_csv(dir_tables / f"metrics_LOGO_per_glacier_{target}.csv", index=False)

df_metrics.drop(columns="GLACIER").agg(["mean","std"]).to_csv(
    dir_tables / f"metrics_LOGO_summary_{target}.csv"
)

# ---------------------------
# Train FINAL deployment model
# ---------------------------
print("\nTraining final model on ALL glaciers...")

final_model = build_pipeline(features)
final_model.fit(data[features], data[target])

joblib.dump(final_model, dir_models / f"xgb_FINAL_all_glaciers_{target}.joblib")

print("\n✔ ALL DONE")
print("✔ Models saved")
print("✔ Metrics saved")

