from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.security import get_current_user
from app.schemas.sentiment import SentimentRequest, SentimentResponse, BatchSentimentRequest
from app.services.sentiment_analysis import SentimentAnalysisService
from app.models.user import User
from app.core.logging import logger
from typing import List
import asyncio

router = APIRouter()

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Sentiment analysis requested by user {current_user.id}")
    service = SentimentAnalysisService(db)
    try:
        result = await service.analyze(request.text)
        logger.info(f"Sentiment analysis completed for user {current_user.id}")
        return SentimentResponse(
            sentiment=result.sentiment,
            confidence=result.confidence,
            text=request.text
        )
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
    logger.info(f"Batch sentiment analysis requested by user {current_user.id}")
    service = SentimentAnalysisService(db)
    try:
        results = await asyncio.gather(*[service.analyze(text) for text in request.texts])
        background_tasks.add_task(service.store_batch_results, results, current_user.id)
        logger.info(f"Batch sentiment analysis completed for user {current_user.id}")
        return [
            SentimentResponse(sentiment=result.sentiment, confidence=result.confidence, text=text)
            for result, text in zip(results, request.texts)
        ]
    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing batch sentiment analysis")

@router.get("/history", response_model=List[SentimentResponse])
async def get_sentiment_history(
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Sentiment analysis history requested by user {current_user.id}")
    service = SentimentAnalysisService(db)
    try:
        history = await service.get_user_history(current_user.id, limit)
        return [
            SentimentResponse(sentiment=item.sentiment, confidence=item.confidence, text=item.text)
            for item in history
        ]
    except Exception as e:
        logger.error(f"Error retrieving sentiment analysis history: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving sentiment analysis history")
