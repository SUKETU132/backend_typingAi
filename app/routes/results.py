from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models import AnalysisResult, User, get_db
from app.core.security import get_current_user

router = APIRouter()


@router.get("/latest-result")
async def get_latest_result(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get the most recent analysis result for the current user"""
    result = (
        db.query(AnalysisResult)
        .filter(AnalysisResult.user_id == current_user.id)
        .order_by(desc(AnalysisResult.upload_date))
        .first()
    )

    if not result:
        return {"success": False, "message": "No results found"}

    return {"success": True, "data": result.to_dict()}


@router.get("/results-history")
async def get_results_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = 10,
):
    """Get historical analysis results for the current user"""
    results = (
        db.query(AnalysisResult)
        .filter(AnalysisResult.user_id == current_user.id)
        .order_by(desc(AnalysisResult.upload_date))
        .limit(limit)
        .all()
    )

    return {"success": True, "data": [r.to_dict() for r in results]}


@router.get("/wpm-trend")
async def get_wpm_trend(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    days: int = 7,
):
    """Get WPM trend data for the current user"""
    results = (
        db.query(AnalysisResult)
        .filter(AnalysisResult.user_id == current_user.id)
        .order_by(desc(AnalysisResult.upload_date))
        .limit(days)
        .all()
    )

    trend_data = []
    for i, result in enumerate(reversed(results)):
        trend_data.append({
            "day":  result.upload_date.strftime("%a") if result.upload_date else f"Day {i}",
            "wpm":  result.avg_wpm,
            "date": result.upload_date.isoformat() if result.upload_date else None,
        })

    return {"success": True, "data": trend_data}
