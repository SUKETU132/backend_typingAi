import pandas as pd
import numpy as np


def build_time_series_features(df: pd.DataFrame, is_inference: bool = False):

    # ---------------------------------------------
    # COPY
    # ---------------------------------------------

    df = df.copy()

    # ---------------------------------------------
    # REQUIRED COLUMNS
    # ---------------------------------------------

    required_columns = [
        "wpm",
        "rawWpm",
        "acc",
        "consistency",
        "timestamp",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # ---------------------------------------------
    # CLEAN DATA
    # ---------------------------------------------

    df = df[required_columns]

    df = df.dropna()

    # Remove impossible values
    df = df[
        (df["wpm"] > 20)
        &
        (df["wpm"] < 250)
    ]

    df = df[
        (df["acc"] > 60)
        &
        (df["acc"] <= 100)
    ]

    df = df[
        (df["consistency"] > 0)
        &
        (df["consistency"] <= 100)
    ]

    # ---------------------------------------------
    # SORT BY TIME
    # ---------------------------------------------

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # ---------------------------------------------
    # TIMESTAMP
    # ---------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms"
    )

    # ---------------------------------------------
    # BASIC FEATURES
    # ---------------------------------------------

    df["speed_loss"] = (
        df["rawWpm"] - df["wpm"]
    )

    df["error_rate"] = (
        100 - df["acc"]
    )

    df["consistency_gap"] = (
        100 - df["consistency"]
    )

    df["fatigue_score"] = (
        (
            df["rawWpm"] - df["wpm"]
        )
        *
        (
            100 - df["consistency"]
        )
    ) / 100

    # ---------------------------------------------
    # LAG FEATURES
    # ---------------------------------------------

    for lag in [1, 2, 3, 5, 10]:

        df[f"wpm_lag_{lag}"] = (
            df["wpm"].shift(lag)
        )

        df[f"acc_lag_{lag}"] = (
            df["acc"].shift(lag)
        )

        df[f"consistency_lag_{lag}"] = (
            df["consistency"].shift(lag)
        )

    # ---------------------------------------------
    # ROLLING FEATURES
    # ---------------------------------------------

    windows = [3, 5, 10, 20]

    for window in windows:

        df[f"wpm_roll_mean_{window}"] = (
            df["wpm"]
            .rolling(window)
            .mean()
        )

        df[f"wpm_roll_std_{window}"] = (
            df["wpm"]
            .rolling(window)
            .std()
        )

        df[f"acc_roll_mean_{window}"] = (
            df["acc"]
            .rolling(window)
            .mean()
        )

        df[f"consistency_roll_mean_{window}"] = (
            df["consistency"]
            .rolling(window)
            .mean()
        )

    # ---------------------------------------------
    # EXPONENTIAL MOVING FEATURES
    # ---------------------------------------------

    df["wpm_ewm_10"] = (
        df["wpm"]
        .ewm(span=10)
        .mean()
    )

    df["acc_ewm_10"] = (
        df["acc"]
        .ewm(span=10)
        .mean()
    )

    df["consistency_ewm_10"] = (
        df["consistency"]
        .ewm(span=10)
        .mean()
    )

    # ---------------------------------------------
    # TREND FEATURES
    # ---------------------------------------------

    df["wpm_delta"] = (
        df["wpm"].diff()
    )

    df["acc_delta"] = (
        df["acc"].diff()
    )

    df["consistency_delta"] = (
        df["consistency"].diff()
    )

    df["speed_acceleration"] = (
        df["wpm_delta"].diff()
    )

    # ---------------------------------------------
    # VOLATILITY FEATURES
    # ---------------------------------------------

    df["speed_volatility"] = (
        df["wpm"]
        .rolling(10)
        .std()
    )

    df["accuracy_volatility"] = (
        df["acc"]
        .rolling(10)
        .std()
    )

    # ---------------------------------------------
    # FATIGUE FEATURES
    # ---------------------------------------------

    df["fatigue_accumulation"] = (
        df["wpm"].cummax()
        -
        df["wpm"]
    )

    df["recovery_score"] = (
        df["wpm"]
        -
        df["wpm_roll_mean_10"]
    )

    # ---------------------------------------------
    # TIME FEATURES
    # ---------------------------------------------

    df["hour"] = (
        df["timestamp"].dt.hour
    )

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # ---------------------------------------------
    # FILL NAS INSTEAD OF DROPPING (FOR INFERENCE)
    # ---------------------------------------------
    
    # Forward fill then backward fill to preserve data for users with < 20 tests
    df = df.ffill().bfill()
    df = df.fillna(0) # Ultimate fallback

    # ---------------------------------------------
    # FUTURE TARGET (TRAINING ONLY)
    # ---------------------------------------------

    if not is_inference:
        forecast_horizon = 5

        df["future_wpm"] = (
            df["wpm"]
            .shift(-forecast_horizon)
        )

        # Predict CHANGE instead of raw value
        df["future_change"] = (
            df["future_wpm"]
            -
            df["wpm"]
        )

        df = df.dropna()

    # ---------------------------------------------
    # CLEANUP
    # ---------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        0 # Safe replacement for math scaling
    )

    print(
        "Processed Shape:",
        df.shape
    )

    return df