from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.security import get_current_user
from app.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
    BatchRecommendationRequest,
    RecommendationModel,
    RecommendationFeedbackSchema,
    TrendingProductResponse,
    PersonalizedDealResponse
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
    """Get personalized recommendations for a customer."""
    logger.info(f"Recommendations requested for customer {request.customer_id} by user {current_user.id}")
    service = RecommendationService(db)
    try:
        result = await service.get_recommendations(request.customer_id, request.num_recommendations)
        logger.info(f"Recommendations generated for customer {request.customer_id}")
        return RecommendationResponse(**result.dict())
    except ValueError as ve:
        logger.error(f"Value error in recommendations: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
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
    """Get recommendations for multiple customers in batch."""
    logger.info(f"Batch recommendations requested by user {current_user.id}")
    service = RecommendationService(db)
    try:
        results = await asyncio.gather(*[service.get_recommendations(customer_id, request.num_recommendations) for customer_id in request.customer_ids])
        background_tasks.add_task(service.store_batch_results, results)
        logger.info(f"Batch recommendations completed for user {current_user.id}")
        return [RecommendationResponse(**result.dict()) for result in results]
    except Exception as e:
        logger.error(f"Error in batch recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing batch recommendations")

@router.get("/models", response_model=List[RecommendationModel])
async def get_recommendation_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get available recommendation models."""
    logger.info(f"Recommendation models requested by user {current_user.id}")
    service = RecommendationService(db)
    try:
        models = await service.get_available_models()
        return models
    except Exception as e:
        logger.error(f"Error retrieving recommendation models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving recommendation models")

@router.post("/feedback", response_model=dict)
async def submit_recommendation_feedback(
    feedback: RecommendationFeedbackSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit feedback for a recommendation."""
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
    """Get performance metrics for the recommendation engine."""
    logger.info(f"Recommendation performance metrics requested by user {current_user.id}")
    service = RecommendationService(db)
    try:
        performance = await service.get_performance_metrics()
        return performance
    except Exception as e:
        logger.error(f"Error retrieving recommendation performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving recommendation performance metrics")

@router.post("/retrain", response_model=dict)
async def retrain_recommendation_model(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrain the recommendation model."""
    logger.info(f"Recommendation model retraining requested by user {current_user.id}")
    service = RecommendationService(db)
    try:
        background_tasks.add_task(service.retrain_model)
        return {"status": "Model retraining started in the background"}
    except Exception as e:
        logger.error(f"Error starting model retraining: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting model retraining")

@router.get("/trending", response_model=List[TrendingProductResponse])
async def get_trending_products(
    category: str = Query(None, description="Filter trending products by category"),
    limit: int = Query(10, ge=1, le=100, description="Number of trending products to return"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get trending products, optionally filtered by category."""
    logger.info(f"Trending products requested by user {current_user.id}")
    service = RecommendationService(db)
    try:
        trending_products = await service.get_trending_products(category, limit)
        return [TrendingProductResponse(**product) for product in trending_products]
    except Exception as e:
        logger.error(f"Error retrieving trending products: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving trending products")

@router.get("/personalized-deals/{customer_id}", response_model=List[PersonalizedDealResponse])
async def get_personalized_deals(
    customer_id: int,
    limit: int = Query(5, ge=1, le=20, description="Number of personalized deals to return"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get personalized deals for a specific customer."""
    logger.info(f"Personalized deals requested for customer {customer_id} by user {current_user.id}")
    service = RecommendationService(db)
    try:
        deals = await service.get_personalized_deals(customer_id, limit)
        return [PersonalizedDealResponse(**deal) for deal in deals]
    except ValueError as ve:
        logger.error(f"Value error in personalized deals: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error retrieving personalized deals: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving personalized deals")
