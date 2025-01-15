from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.security import get_current_user
from app.schemas.customer_journey import (
    CustomerJourneyRequest,
    CustomerJourneyResponse,
    TouchpointAnalysisRequest,
    TouchpointAnalysisResponse,
    JourneyOptimizationRequest,
    JourneyOptimizationResponse,
    JourneySegmentRequest,
    JourneySegmentResponse
)
from app.services.customer_journey import CustomerJourneyService
from app.models.user import User
from app.core.logging import logger
from typing import List
import asyncio

router = APIRouter()

@router.post("/analyze", response_model=CustomerJourneyResponse)
async def analyze_customer_journey(
    request: CustomerJourneyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Customer journey analysis requested for customer {request.customer_id} by user {current_user.id}")
    service = CustomerJourneyService(db)
    try:
        result = await service.analyze_journey(request.customer_id)
        logger.info(f"Customer journey analysis completed for customer {request.customer_id}")
        return CustomerJourneyResponse(
            customer_id=request.customer_id,
            touchpoints=result.touchpoints,
            journey_duration=result.journey_duration,
            conversion_rate=result.conversion_rate,
            pain_points=result.pain_points,
            recommendations=result.recommendations
        )
    except Exception as e:
        logger.error(f"Error in customer journey analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing customer journey analysis")

@router.post("/touchpoint-analysis", response_model=TouchpointAnalysisResponse)
async def analyze_touchpoint(
    request: TouchpointAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Touchpoint analysis requested for touchpoint {request.touchpoint_id} by user {current_user.id}")
    service = CustomerJourneyService(db)
    try:
        result = await service.analyze_touchpoint(request.touchpoint_id)
        logger.info(f"Touchpoint analysis completed for touchpoint {request.touchpoint_id}")
        return TouchpointAnalysisResponse(
            touchpoint_id=request.touchpoint_id,
            engagement_rate=result.engagement_rate,
            conversion_impact=result.conversion_impact,
            average_time_spent=result.average_time_spent,
            customer_feedback=result.customer_feedback,
            improvement_suggestions=result.improvement_suggestions
        )
    except Exception as e:
        logger.error(f"Error in touchpoint analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing touchpoint analysis")

@router.post("/optimize-journey", response_model=JourneyOptimizationResponse)
async def optimize_customer_journey(
    request: JourneyOptimizationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Journey optimization requested for segment {request.segment_id} by user {current_user.id}")
    service = CustomerJourneyService(db)
    try:
        result = await service.optimize_journey(request.segment_id)
        background_tasks.add_task(service.implement_journey_changes, result.optimization_id)
        logger.info(f"Journey optimization completed for segment {request.segment_id}")
        return JourneyOptimizationResponse(
            segment_id=request.segment_id,
            optimization_id=result.optimization_id,
            suggested_changes=result.suggested_changes,
            expected_impact=result.expected_impact,
            implementation_timeline=result.implementation_timeline
        )
    except Exception as e:
        logger.error(f"Error in journey optimization: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing journey optimization")

@router.post("/segment-journey", response_model=List[JourneySegmentResponse])
async def segment_customer_journeys(
    request: JourneySegmentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Journey segmentation requested by user {current_user.id}")
    service = CustomerJourneyService(db)
    try:
        results = await service.segment_journeys(request.num_segments)
        logger.info(f"Journey segmentation completed")
        return [
            JourneySegmentResponse(
                segment_id=result.segment_id,
                segment_name=result.segment_name,
                segment_size=result.segment_size,
                average_journey_duration=result.average_journey_duration,
                common_touchpoints=result.common_touchpoints,
                conversion_rate=result.conversion_rate
            ) for result in results
        ]
    except Exception as e:
        logger.error(f"Error in journey segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing journey segmentation")

@router.get("/journey-metrics", response_model=dict)
async def get_journey_metrics(
    start_date: str,
    end_date: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Journey metrics requested by user {current_user.id}")
    service = CustomerJourneyService(db)
    try:
        metrics = await service.get_journey_metrics(start_date, end_date)
        return metrics
    except Exception as e:
        logger.error(f"Error retrieving journey metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving journey metrics")

@router.post("/predict-churn", response_model=dict)
async def predict_customer_churn(
    customer_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Churn prediction requested for customer {customer_id} by user {current_user.id}")
    service = CustomerJourneyService(db)
    try:
        prediction = await service.predict_churn(customer_id)
        return prediction
    except Exception as e:
        logger.error(f"Error predicting customer churn: {str(e)}")
        raise HTTPException(status_code=500, detail="Error predicting customer churn")
