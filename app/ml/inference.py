import joblib
import pandas as pd

MODEL_PATH = (
    "app/ml/models/xgb_model.pkl"
)

# Load trained model

model = joblib.load(MODEL_PATH)

FEATURES = [

    "wpm",
    "rawWpm",
    "acc",
    "consistency",

    "speed_loss",
    "error_rate",
    "consistency_gap",
    "fatigue_score",

    "wpm_roll_3",
    "wpm_roll_5",

    "acc_roll_5",
    "consistency_roll_5",

    "rolling_std_5",

    "wpm_lag_1",
    "wpm_lag_2",

    "wpm_delta",

    "hour",
    "day_of_week",
    "is_weekend",
]


def predict_future_wpm(data):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return round(
        float(prediction),
        2
    )