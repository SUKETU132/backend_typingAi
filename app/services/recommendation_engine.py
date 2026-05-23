def generate_recommendations(df, analysis):

    latest = df.iloc[-1]

    recommendations = []

    # Accuracy Problems

    if latest["acc"] < 95:
        recommendations.append({
            "type": "accuracy",
            "message":
                "Your accuracy is reducing your effective WPM. Focus on slower precision drills."
        })

    # Consistency Problems

    if latest["consistency"] < 80:
        recommendations.append({
            "type": "consistency",
            "message":
                "Your typing rhythm is unstable. Practice 60-second consistency sessions."
        })

    # Fatigue Detection

    if latest["fatigue_score"] < -10:
        recommendations.append({
            "type": "fatigue",
            "message":
                "Performance drop detected during longer sessions. Take short recovery breaks."
        })

    # Speed Plateau

    if abs(latest["wpm_delta"]) < 1:
        recommendations.append({
            "type": "plateau",
            "message":
                "Your speed growth has plateaued. Introduce burst typing sessions at 130-140 WPM."
        })

    # Raw Speed Loss

    if latest["speed_loss"] > 10:
        recommendations.append({
            "type": "speed_loss",
            "message":
                "You are losing speed due to correction overhead. Reduce overcorrection habits."
        })

    # Strong Performance

    if (
        latest["wpm"] > 115
        and latest["acc"] > 97
        and latest["consistency"] > 85
    ):
        recommendations.append({
            "type": "elite",
            "message":
                "Excellent typing profile detected. Focus on endurance training to push beyond current limits."
        })

    return recommendations