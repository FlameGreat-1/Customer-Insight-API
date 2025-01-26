from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.security import get_current_user
from app.schemas.sentiment import (
    SentimentRequest,
    SentimentResponse,
    BatchSentimentRequest,
    SentimentDistribution,
    PerformanceMetrics,
    SentimentTrends,
    FeedbackImpactAnalysis,
    ModelRetrainingRequest
)
from app.services.sentiment_analysis import SentimentAnalysisService
from app.models.user import User
from app.core.logging import logger
from typing import List, Dict
import asyncio
from datetime import datetime

router = APIRouter()

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyze the sentiment of a single text."""
    logger.info(f"Sentiment analysis requested by user {current_user.id}")
    service = SentimentAnalysisService(db)
    try:
        result = await service.analyze(request.text)
        logger.info(f"Sentiment analysis completed for user {current_user.id}")
        return SentimentResponse(**result.dict())
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing sentiment analysis")

@router.post("/batch-analyze", response_model=List[SentimentResponse])
async def batch_analyze_sentiment(
    request: BatchSentimentRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyze the sentiment of multiple texts in batch."""
    logger.info(f"Batch sentiment analysis requested by user {current_user.id}")
    service = SentimentAnalysisService(db)
    try:
        results = await service.analyze_batch(request.texts)
        background_tasks.add_task(service.store_batch_results, results, current_user.id)
        logger.info(f"Batch sentiment analysis completed for user {current_user.id}")
        return [SentimentResponse(**result.dict()) for result in results]
    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing batch sentiment analysis")

@router.get("/history", response_model=List[SentimentResponse])
async def get_sentiment_history(
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the sentiment analysis history for the current user."""
    logger.info(f"Sentiment analysis history requested by user {current_user.id}")
    service = SentimentAnalysisService(db)
    try:
        history = await service.get_user_history(current_user.id, limit)
        return [SentimentResponse(**item.dict()) for item in history]
    except Exception as e:
        logger.error(f"Error retrieving sentiment analysis history: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving sentiment analysis history")

@router.get("/distribution", response_model=SentimentDistribution)
async def get_sentiment_distribution(
    start_date: datetime = Query(..., description="Start date for distribution calculation"),
    end_date: datetime = Query(..., description="End date for distribution calculation"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the distribution of sentiments within a specified date range."""
    logger.info(f"Sentiment distribution requested by user {current_user.id}")
    service = SentimentAnalysisService(db)
    try:
        distribution = await service.get_sentiment_distribution(start_date, end_date)
        return SentimentDistribution(distribution=distribution)
    except Exception as e:
        logger.error(f"Error retrieving sentiment distribution: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving sentiment distribution")

@router.post("/retrain", status_code=202)
async def retrain_model(
    request: ModelRetrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrain the sentiment analysis model with new data."""
    logger.info(f"Model retraining requested by user {current_user.id}")
    service = SentimentAnalysisService(db)
    try:
        background_tasks.add_task(service.retrain_model, request.training_data)
        return {"message": "Model retraining started in the background"}
    except Exception as e:
        logger.error(f"Error starting model retraining: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting model retraining")

@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get performance metrics for the sentiment analysis model."""
    logger.info(f"Performance metrics requested by user {current_user.id}")
    service = SentimentAnalysisService(db)
    try:
        metrics = await service.get_performance_metrics()
        return PerformanceMetrics(**metrics)
    except Exception as e:
        logger.error(f"Error retrieving performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving performance metrics")

@router.get("/trends", response_model=SentimentTrends)
async def get_sentiment_trends(
    start_date: datetime = Query(..., description="Start date for trend calculation"),
    end_date: datetime = Query(..., description="End date for trend calculation"),
    interval: str = Query("day", description="Interval for trend data points (hour, day, week, month)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get sentiment trends over a specified time period."""
    logger.info(f"Sentiment trends requested by user {current_user.id}")
    service = SentimentAnalysisService(db)
    try:
        trends = await service.get_sentiment_trends(start_date, end_date, interval)
        return SentimentTrends(trends=trends)
    except Exception as e:
        logger.error(f"Error retrieving sentiment trends: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving sentiment trends")

@router.get("/feedback-impact", response_model=FeedbackImpactAnalysis)
async def analyze_feedback_impact(
    start_date: datetime = Query(..., description="Start date for impact analysis"),
    end_date: datetime = Query(..., description="End date for impact analysis"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyze the impact of sentiment on customer behavior."""
    logger.info(f"Feedback impact analysis requested by user {current_user.id}")
    service = SentimentAnalysisService(db)
    try:
        impact = await service.analyze_feedback_impact(start_date, end_date)
        return FeedbackImpactAnalysis(impact=impact)
    except Exception as e:
        logger.error(f"Error analyzing feedback impact: {str(e)}")
        raise HTTPException(status_code=500, detail="Error analyzing feedback impact")
