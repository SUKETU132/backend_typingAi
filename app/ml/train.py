import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)

from sklearn.model_selection import (
    TimeSeriesSplit,
)

from sklearn.preprocessing import (
    RobustScaler,
)

from xgboost import (
    XGBRegressor,
)

from app.services.feature_engineering import (
    build_time_series_features,
)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = pd.read_csv(
    "uploads/results.csv"
)

print(df.head())
print(df.shape)

# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------

df = build_time_series_features(df)

if len(df) < 100:
    raise Exception(
        "Dataset too small after processing."
    )

# -------------------------------------------------
# FEATURES
# -------------------------------------------------

IGNORE_COLUMNS = [
    "timestamp",
    "future_wpm",
    "future_change",
]

FEATURES = [
    col
    for col in df.columns
    if col not in IGNORE_COLUMNS
]

TARGET = "future_change"

# -------------------------------------------------
# DATA
# -------------------------------------------------

X = df[FEATURES]

y = df[TARGET]

# -------------------------------------------------
# SCALE
# -------------------------------------------------

scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# MODEL
# -------------------------------------------------

model = XGBRegressor(

    objective="reg:squarederror",

    n_estimators=2000,

    learning_rate=0.005,

    max_depth=8,

    min_child_weight=3,

    subsample=0.85,

    colsample_bytree=0.85,

    gamma=0.1,

    reg_alpha=0.5,

    reg_lambda=2,

    random_state=42,
)

# -------------------------------------------------
# VALIDATION
# -------------------------------------------------

tscv = TimeSeriesSplit(
    n_splits=5
)

scores = []

for fold, (
    train_index,
    test_index
) in enumerate(
    tscv.split(X_scaled)
):

    X_train, X_test = (
        X_scaled[train_index],
        X_scaled[test_index],
    )

    y_train, y_test = (
        y.iloc[train_index],
        y.iloc[test_index],
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    preds = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        preds
    )

    r2 = r2_score(
        y_test,
        preds
    )

    scores.append({
        "fold": fold + 1,
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
    })

# -------------------------------------------------
# FINAL TRAIN
# -------------------------------------------------

model.fit(
    X_scaled,
    y
)

# -------------------------------------------------
# SAVE
# -------------------------------------------------

os.makedirs(
    "app/ml/models",
    exist_ok=True
)

joblib.dump(
    model,
    "app/ml/models/xgb_model.pkl"
)

joblib.dump(
    scaler,
    "app/ml/models/scaler.pkl"
)

joblib.dump(
    FEATURES,
    "app/ml/models/features.pkl"
)

# -------------------------------------------------
# RESULTS
# -------------------------------------------------

print("\nTraining Complete\n")

for score in scores:
    print(score)

avg_r2 = np.mean([
    s["R2"]
    for s in scores
])

avg_mae = np.mean([
    s["MAE"]
    for s in scores
])

print("\nAverage MAE:", round(avg_mae, 4))
print("Average R2:", round(avg_r2, 4))

print("\nModel Saved Successfully")