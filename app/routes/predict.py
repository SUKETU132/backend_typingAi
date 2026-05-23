from fastapi import APIRouter

from app.ml.inference import (
    predict_future_wpm
)

router = APIRouter()


@router.post("/predict")
async def predict(data: dict):

    prediction = predict_future_wpm(
        data
    )

    current_wpm = data["wpm"]

    delta = prediction - current_wpm

    # Performance Trend

    if delta > 2:
        trend = "improving"

    elif delta < -2:
        trend = "declining"

    else:
        trend = "stable"

    # Fatigue Risk

    fatigue_score = (
        data["fatigue_score"]
    )

    if fatigue_score < -10:
        fatigue_risk = "high"

    elif fatigue_score < -5:
        fatigue_risk = "medium"

    else:
        fatigue_risk = "low"

    # Confidence Estimation

    consistency = (
        data["consistency"]
    )

    confidence = round(
        consistency / 100,
        2
    )

    return {

        "predicted_wpm":
            prediction,

        "current_wpm":
            current_wpm,

        "delta":
            round(delta, 2),

        "performance_trend":
            trend,

        "fatigue_risk":
            fatigue_risk,

        "confidence":
            confidence,
    }