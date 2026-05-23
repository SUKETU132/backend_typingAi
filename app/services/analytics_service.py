def analyze_typing_behavior(df):

    latest = df.iloc[-1]

    analysis = {

        # Core Metrics

        "current_wpm":
            round(float(latest["wpm"]), 2),

        "current_accuracy":
            round(float(latest["acc"]), 2),

        "current_consistency":
            round(float(latest["consistency"]), 2),

        # Trend Metrics

        "avg_wpm":
            round(float(df["wpm"].mean()), 2),

        "avg_accuracy":
            round(float(df["acc"].mean()), 2),

        "avg_consistency":
            round(float(
                df["consistency"].mean()
            ), 2),

        # Stability

        "wpm_variance":
            round(float(
                df["wpm_roll_std_5"].mean()
            ), 2),

        # Fatigue

        "fatigue_score":
            round(float(
                latest["fatigue_score"]
            ), 2),

        # Performance Loss

        "speed_loss":
            round(float(
                latest["speed_loss"]
            ), 2),
    }

    # Performance State

    if latest["wpm_delta"] > 2:
        analysis["trend"] = "improving"

    elif latest["wpm_delta"] < -2:
        analysis["trend"] = "declining"

    else:
        analysis["trend"] = "stable"

    # Fatigue Risk

    if latest["fatigue_score"] < -10:
        analysis["fatigue_risk"] = "high"

    elif latest["fatigue_score"] < -5:
        analysis["fatigue_risk"] = "medium"

    else:
        analysis["fatigue_risk"] = "low"

    return analysis