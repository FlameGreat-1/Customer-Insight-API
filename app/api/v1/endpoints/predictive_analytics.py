from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.security import get_current_user
from app.schemas.predictive_analytics import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    ModelPerformance,
    FeatureImportance
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
    logger.info(f"Prediction requested for customer {request.customer_id} by user {current_user.id}")
    service = PredictiveAnalyticsService(db)
    try:
        result = await service.make_prediction(request.customer_id, request.prediction_type)
        logger.info(f"Prediction generated for customer {request.customer_id}")
        return PredictionResponse(
            customer_id=request.customer_id,
            prediction_type=request.prediction_type,
            prediction=result.prediction,
            confidence=result.confidence,
            explanation=result.explanation
        )
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
    logger.info(f"Batch predictions requested by user {current_user.id}")
    service = PredictiveAnalyticsService(db)
    try:
        results = await asyncio.gather(*[service.make_prediction(customer_id, request.prediction_type) for customer_id in request.customer_ids])
        background_tasks.add_task(service.store_batch_results, results, current_user.id)
        logger.info(f"Batch predictions completed for user {current_user.id}")
        return [
            PredictionResponse(
                customer_id=customer_id,
                prediction_type=request.prediction_type,
                prediction=result.prediction,
                confidence=result.confidence,
                explanation=result.explanation
            )
            for result, customer_id in zip(results, request.customer_ids)
        ]
    except Exception as e:
        logger.error(f"Error in batch predictions: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing batch predictions")

@router.get("/model-performance", response_model=ModelPerformance)
async def get_model_performance(
    prediction_type: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Model performance requested for {prediction_type} by user {current_user.id}")
    service = PredictiveAnalyticsService(db)
    try:
        performance = await service.get_model_performance(prediction_type)
        return ModelPerformance(
            prediction_type=prediction_type,
            accuracy=performance.accuracy,
            precision=performance.precision,
            recall=performance.recall,
            f1_score=performance.f1_score
        )
    except Exception as e:
        logger.error(f"Error retrieving model performance: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving model performance")

@router.get("/feature-importance", response_model=List[FeatureImportance])
async def get_feature_importance(
    prediction_type: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Feature importance requested for {prediction_type} by user {current_user.id}")
    service = PredictiveAnalyticsService(db)
    try:
        importance = await service.get_feature_importance(prediction_type)
        return [FeatureImportance(feature=feat, importance=imp) for feat, imp in importance]
    except Exception as e:
        logger.error(f"Error retrieving feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving feature importance")

@router.post("/retrain", response_model=dict)
async def retrain_model(
    prediction_type: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Model retraining requested for {prediction_type} by user {current_user.id}")
    service = PredictiveAnalyticsService(db)
    try:
        background_tasks.add_task(service.retrain_model, prediction_type)
        return {"status": f"Retraining of {prediction_type} model started in the background"}
    except Exception as e:
        logger.error(f"Error starting model retraining: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting model retraining")
