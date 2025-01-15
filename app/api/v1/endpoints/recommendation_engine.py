from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.security import get_current_user
from app.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
    BatchRecommendationRequest,
    RecommendationModel,
    RecommendationFeedback
)
from app.services.recommendation_engine import RecommendationService
from app.models.user import User
from app.core.logging import logger
from typing import List
import asyncio

router = APIRouter()

@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Recommendations requested for customer {request.customer_id} by user {current_user.id}")
    service = RecommendationService(db)
    try:
        result = await service.get_recommendations(request.customer_id, request.num_recommendations)
        logger.info(f"Recommendations generated for customer {request.customer_id}")
        return RecommendationResponse(
            customer_id=request.customer_id,
            recommendations=result.recommendations,
            explanation=result.explanation
        )
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating recommendations")

@router.post("/batch-recommend", response_model=List[RecommendationResponse])
async def batch_get_recommendations(
    request: BatchRecommendationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Batch recommendations requested by user {current_user.id}")
    service = RecommendationService(db)
    try:
        results = await asyncio.gather(*[service.get_recommendations(customer_id, request.num_recommendations) for customer_id in request.customer_ids])
        background_tasks.add_task(service.store_batch_results, results, current_user.id)
        logger.info(f"Batch recommendations completed for user {current_user.id}")
        return [
            RecommendationResponse(
                customer_id=customer_id,
                recommendations=result.recommendations,
                explanation=result.explanation
            )
            for result, customer_id in zip(results, request.customer_ids)
        ]
    except Exception as e:
        logger.error(f"Error in batch recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing batch recommendations")

@router.get("/models", response_model=List[RecommendationModel])
async def get_recommendation_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Recommendation models requested by user {current_user.id}")
    service = RecommendationService(db)
    try:
        models = await service.get_available_models()
        return [RecommendationModel(id=model.id, name=model.name, description=model.description) for model in models]
    except Exception as e:
        logger.error(f"Error retrieving recommendation models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving recommendation models")

@router.post("/feedback", response_model=dict)
async def submit_recommendation_feedback(
    feedback: RecommendationFeedback,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Recommendation feedback submitted by user {current_user.id}")
    service = RecommendationService(db)
    try:
        await service.process_feedback(feedback)
        return {"status": "Feedback processed successfully"}
    except Exception as e:
        logger.error(f"Error processing recommendation feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing recommendation feedback")

@router.get("/performance", response_model=dict)
async def get_recommendation_performance(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Recommendation performance metrics requested by user {current_user.id}")
    service = RecommendationService(db)
    try:
        performance = await service.get_performance_metrics()
        return performance
    except Exception as e:
        logger.error(f"Error retrieving recommendation performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving recommendation performance metrics")
