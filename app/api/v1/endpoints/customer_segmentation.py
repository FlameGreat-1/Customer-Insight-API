from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.security import get_current_user
from app.schemas.customer_segmentation import (
    SegmentationRequest,
    SegmentationResponse,
    BatchSegmentationRequest,
    SegmentationModel,
    SegmentUpdateRequest,
    SegmentCharacteristics,
    PerformanceMetrics,
    SegmentTransitionMatrix,
    CohortAnalysisRequest,
    FeatureImportance
)
from app.services.customer_segmentation import CustomerSegmentationService
from app.models.user import User
from app.core.logging import logger
from typing import List, Dict
import asyncio
from datetime import datetime

router = APIRouter()

@router.post("/segment", response_model=SegmentationResponse)
async def segment_customer(
    request: SegmentationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Segment a single customer."""
    logger.info(f"Customer segmentation requested by user {current_user.id}")
    service = CustomerSegmentationService(db)
    try:
        result = await service.segment_customer(request.customer_id)
        logger.info(f"Customer segmentation completed for user {current_user.id}")
        return SegmentationResponse(**result.dict())
    except Exception as e:
        logger.error(f"Error in customer segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing customer segmentation")

@router.post("/batch-segment", response_model=Dict[int, str])
async def batch_segment_customers(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Segment all customers in the database."""
    logger.info(f"Batch customer segmentation requested by user {current_user.id}")
    service = CustomerSegmentationService(db)
    try:
        background_tasks.add_task(service.segment_all_customers)
        return {"message": "Batch segmentation started in the background"}
    except Exception as e:
        logger.error(f"Error in batch customer segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing batch customer segmentation")

@router.get("/models", response_model=List[SegmentationModel])
async def get_segmentation_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get available segmentation models."""
    logger.info(f"Segmentation models requested by user {current_user.id}")
    service = CustomerSegmentationService(db)
    try:
        models = await service.get_available_models()
        return models
    except Exception as e:
        logger.error(f"Error retrieving segmentation models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving segmentation models")

@router.get("/segment-distribution", response_model=Dict[str, float])
async def get_segment_distribution(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the distribution of customers across segments."""
    logger.info(f"Segment distribution requested by user {current_user.id}")
    service = CustomerSegmentationService(db)
    try:
        distribution = await service.get_segment_distribution()
        return distribution
    except Exception as e:
        logger.error(f"Error retrieving segment distribution: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving segment distribution")

@router.get("/segment-characteristics/{segment}", response_model=SegmentCharacteristics)
async def get_segment_characteristics(
    segment: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get characteristics of a specific segment."""
    logger.info(f"Segment characteristics requested for {segment} by user {current_user.id}")
    service = CustomerSegmentationService(db)
    try:
        characteristics = await service.get_segment_characteristics(segment)
        return SegmentCharacteristics(**characteristics)
    except Exception as e:
        logger.error(f"Error retrieving segment characteristics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving segment characteristics")

@router.post("/retrain", status_code=202)
async def retrain_model(
    n_clusters: int = 5,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrain the segmentation model."""
    logger.info(f"Model retraining requested by user {current_user.id}")
    service = CustomerSegmentationService(db)
    try:
        background_tasks.add_task(service.retrain_model, n_clusters)
        return {"message": "Model retraining started in the background"}
    except Exception as e:
        logger.error(f"Error starting model retraining: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting model retraining")

@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get performance metrics of the current segmentation model."""
    logger.info(f"Performance metrics requested by user {current_user.id}")
    service = CustomerSegmentationService(db)
    try:
        metrics = await service.get_performance_metrics()
        return PerformanceMetrics(**metrics)
    except Exception as e:
        logger.error(f"Error retrieving performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving performance metrics")

@router.post("/optimize", status_code=202)
async def optimize_model(
    max_clusters: int = 10,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Optimize the segmentation model."""
    logger.info(f"Model optimization requested by user {current_user.id}")
    service = CustomerSegmentationService(db)
    try:
        background_tasks.add_task(service.optimize_model, max_clusters)
        return {"message": "Model optimization started in the background"}
    except Exception as e:
        logger.error(f"Error starting model optimization: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting model optimization")

@router.get("/transition-matrix", response_model=SegmentTransitionMatrix)
async def get_segment_transition_matrix(
    start_date: datetime,
    end_date: datetime,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the segment transition matrix for a specific time period."""
    logger.info(f"Segment transition matrix requested by user {current_user.id}")
    service = CustomerSegmentationService(db)
    try:
        matrix = await service.get_segment_transition_matrix(start_date, end_date)
        return SegmentTransitionMatrix(matrix=matrix)
    except Exception as e:
        logger.error(f"Error retrieving segment transition matrix: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving segment transition matrix")

@router.post("/cohort-analysis")
async def get_segment_cohort_analysis(
    request: CohortAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Perform cohort analysis on customer segments."""
    logger.info(f"Segment cohort analysis requested by user {current_user.id}")
    service = CustomerSegmentationService(db)
    try:
        analysis = await service.get_segment_cohort_analysis(request.cohort_period)
        return analysis.to_dict()  # Convert DataFrame to dict for JSON serialization
    except Exception as e:
        logger.error(f"Error performing cohort analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error performing cohort analysis")

@router.get("/feature-importance", response_model=FeatureImportance)
async def get_segment_feature_importance(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the importance of features in the segmentation model."""
    logger.info(f"Feature importance requested by user {current_user.id}")
    service = CustomerSegmentationService(db)
    try:
        importance = await service.get_segment_feature_importance()
        return FeatureImportance(feature_importance=importance)
    except Exception as e:
        logger.error(f"Error retrieving feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving feature importance")
