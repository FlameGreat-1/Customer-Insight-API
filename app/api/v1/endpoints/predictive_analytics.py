from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.security import get_current_user
from app.schemas.predictive_analytics import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    ModelPerformance,
    FeatureImportance,
    CustomerSegmentResponse,
    CustomerRecommendationsResponse
)
from app.services.predictive_analytics import PredictiveAnalyticsService
from app.models.user import User
from app.core.logging import logger
from typing import List
import asyncio

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def make_prediction(
    request: PredictionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Make a prediction for a single customer."""
    logger.info(f"Prediction requested for customer {request.customer_id} by user {current_user.id}")
    service = PredictiveAnalyticsService(db)
    try:
        result = await service.make_prediction(request.customer_id, request.prediction_type)
        logger.info(f"Prediction generated for customer {request.customer_id}")
        return PredictionResponse(**result.dict())
    except ValueError as ve:
        logger.error(f"Value error in prediction: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error generating prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating prediction")

@router.post("/batch-predict", response_model=List[PredictionResponse])
async def batch_make_predictions(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Make predictions for multiple customers."""
    logger.info(f"Batch predictions requested by user {current_user.id}")
    service = PredictiveAnalyticsService(db)
    try:
        results = await asyncio.gather(*[service.make_prediction(customer_id, request.prediction_type) for customer_id in request.customer_ids])
        background_tasks.add_task(service.store_batch_results, results, current_user.id)
        logger.info(f"Batch predictions completed for user {current_user.id}")
        return [PredictionResponse(**result.dict()) for result in results]
    except ValueError as ve:
        logger.error(f"Value error in batch predictions: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in batch predictions: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing batch predictions")

@router.get("/model-performance", response_model=ModelPerformance)
async def get_model_performance(
    prediction_type: str = Query(..., description="Type of prediction (churn or ltv)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get performance metrics for a specific prediction model."""
    logger.info(f"Model performance requested for {prediction_type} by user {current_user.id}")
    service = PredictiveAnalyticsService(db)
    try:
        performance = await service.get_model_performance(prediction_type)
        return performance
    except ValueError as ve:
        logger.error(f"Value error in getting model performance: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error retrieving model performance: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving model performance")

@router.get("/feature-importance", response_model=List[FeatureImportance])
async def get_feature_importance(
    prediction_type: str = Query(..., description="Type of prediction (churn or ltv)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get feature importance for a specific prediction model."""
    logger.info(f"Feature importance requested for {prediction_type} by user {current_user.id}")
    service = PredictiveAnalyticsService(db)
    try:
        importance = await service.get_feature_importance(prediction_type)
        return importance
    except ValueError as ve:
        logger.error(f"Value error in getting feature importance: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error retrieving feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving feature importance")

@router.post("/retrain", response_model=dict)
async def retrain_model(
    prediction_type: str = Query(..., description="Type of prediction model to retrain (churn or ltv)"),
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrain a specific prediction model."""
    logger.info(f"Model retraining requested for {prediction_type} by user {current_user.id}")
    service = PredictiveAnalyticsService(db)
    try:
        background_tasks.add_task(service.retrain_model, prediction_type)
        return {"status": f"Retraining of {prediction_type} model started in the background"}
    except ValueError as ve:
        logger.error(f"Value error in model retraining: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error starting model retraining: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting model retraining")

@router.get("/customer-segment/{customer_id}", response_model=CustomerSegmentResponse)
async def get_customer_segment(
    customer_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the segment of a specific customer."""
    logger.info(f"Customer segment requested for customer {customer_id} by user {current_user.id}")
    service = PredictiveAnalyticsService(db)
    try:
        segment = await service.get_customer_segment(customer_id)
        return CustomerSegmentResponse(customer_id=customer_id, segment=segment)
    except ValueError as ve:
        logger.error(f"Value error in getting customer segment: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error retrieving customer segment: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving customer segment")

@router.get("/customer-recommendations/{customer_id}", response_model=CustomerRecommendationsResponse)
async def get_customer_recommendations(
    customer_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get recommendations for a specific customer."""
    logger.info(f"Customer recommendations requested for customer {customer_id} by user {current_user.id}")
    service = PredictiveAnalyticsService(db)
    try:
        recommendations = await service.get_customer_recommendations(customer_id)
        return CustomerRecommendationsResponse(customer_id=customer_id, recommendations=recommendations)
    except ValueError as ve:
        logger.error(f"Value error in getting customer recommendations: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error retrieving customer recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving customer recommendations")
