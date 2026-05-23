from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Depends,
)

import pandas as pd
import tempfile
import joblib
import numpy as np
import json

from app.services.feature_engineering import (
    build_time_series_features,
)
from app.services.analytics_service import (
    analyze_typing_behavior,
)
from app.services.recommendation_engine import (
    generate_recommendations,
)
from app.models import AnalysisResult, User, get_db
from app.core.security import get_current_user
from sqlalchemy.orm import Session

router = APIRouter()

# -------------------------------------------------
# LOAD ASSETS
# -------------------------------------------------

model = joblib.load(
    "app/ml/models/xgb_model.pkl"
)

scaler = joblib.load(
    "app/ml/models/scaler.pkl"
)

FEATURES = joblib.load(
    "app/ml/models/features.pkl"
)

# -------------------------------------------------
# ROUTE
# -------------------------------------------------

@router.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    try:

        # -----------------------------------------
        # TEMP FILE
        # -----------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv"
        ) as temp_file:

            temp_file.write(
                await file.read()
            )

            temp_path = temp_file.name

        # -----------------------------------------
        # LOAD CSV
        # -----------------------------------------

        df = pd.read_csv(temp_path)

        # -----------------------------------------
        # FEATURE ENGINEERING
        # -----------------------------------------

        processed_df = (
            build_time_series_features(df, is_inference=True)
        )

        if len(processed_df) == 0:

            raise HTTPException(
                status_code=400,
                detail="No valid rows after processing."
            )

        # -----------------------------------------
        # FEATURE SELECTION
        # -----------------------------------------

        X = processed_df[FEATURES]

        # -----------------------------------------
        # SCALE
        # -----------------------------------------

        X_scaled = scaler.transform(X)

        # -----------------------------------------
        # PREDICT CHANGE
        # -----------------------------------------

        predicted_change = (
            model.predict(X_scaled)
        )

        latest_change = float(
            predicted_change[-1]
        )

        current_wpm = float(
            processed_df.iloc[-1]["wpm"]
        )

        predicted_future_wpm = (
            current_wpm +
            latest_change
        )

        # -----------------------------------------
        # SAFETY CLAMP
        # -----------------------------------------

        predicted_future_wpm = np.clip(
            predicted_future_wpm,
            20,
            250
        )

        # -----------------------------------------
        # ANALYTICS
        # -----------------------------------------

        try:
            analysis = analyze_typing_behavior(
                processed_df
            )
        except Exception as analysis_error:
            raise HTTPException(
                status_code=500,
                detail=f"Analytics error: {str(analysis_error)}"
            )

        try:
            recommendations = generate_recommendations(
                processed_df,
                analysis
            )
        except Exception as rec_error:
            raise HTTPException(
                status_code=500,
                detail=f"Recommendations error: {str(rec_error)}"
            )

        # -----------------------------------------
        # EXTRACT FEATURES FOR FRONTEND
        # -----------------------------------------

        features = {
            "avg_wpm": float(analysis.get("avg_wpm", 0)),
            "max_wpm": float(
                processed_df["wpm"].max()
            ),
            "avg_accuracy": float(analysis.get("avg_accuracy", 0)),
            "avg_consistency": float(analysis.get("avg_consistency", 0)),
        }

        analysis_response = {
            "speed_loss": float(analysis.get("speed_loss", 0)),
            "accuracy_status": "Good" if float(analysis.get("current_accuracy", 0)) >= 95 else "Needs Improvement",
            "consistency_status": "Good" if float(analysis.get("current_consistency", 0)) >= 80 else "Needs Improvement",
        }

        recommendations_response = [
            item.get("message", "")
            for item in recommendations
        ]

        # -----------------------------------------
        # EXTRACT WPM HISTORY FOR CHART
        # -----------------------------------------

        wpm_history = processed_df[["wpm"]].tail(30).reset_index(drop=True).to_dict("records")

        # -----------------------------------------
        # SAVE TO DATABASE
        # -----------------------------------------

        result = AnalysisResult(
            user_id=current_user.id,
            filename=file.filename,
            avg_wpm=float(features["avg_wpm"]),
            max_wpm=float(features["max_wpm"]),
            avg_accuracy=float(features["avg_accuracy"]),
            avg_consistency=float(features["avg_consistency"]),
            speed_loss=float(analysis.get("speed_loss", 0)),
            current_wpm=float(analysis.get("current_wpm", 0)),
            current_accuracy=float(analysis.get("current_accuracy", 0)),
            current_consistency=float(analysis.get("current_consistency", 0)),
            wpm_variance=float(analysis.get("wpm_variance", 0)),
            fatigue_score=float(analysis.get("fatigue_score", 0)),
            trend=analysis.get("trend", "stable"),
            fatigue_risk=analysis.get("fatigue_risk", "low"),
            predicted_future_wpm=float(predicted_future_wpm),
            predicted_change=float(latest_change),
            recommendations=recommendations,
            wpm_history=wpm_history,
        )
        
        db.add(result)
        db.commit()
        db.refresh(result)

        # -----------------------------------------
        # RESPONSE
        # -----------------------------------------

        return {
            "success": True,
            "data": result.to_dict()
        }

    except Exception as e:

        print(f"ERROR in upload: {str(e)}")
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )