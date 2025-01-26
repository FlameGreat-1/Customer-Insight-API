from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse
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
    JourneySegmentResponse,
    JourneyMetricsResponse,
    ChurnPredictionResponse,
    JourneyReportResponse,
    SentimentAnalysisResponse
)
from app.services.customer_journey import CustomerJourneyService
from app.models.user import User
from app.core.logging import logger
from typing import List, Dict
from datetime import datetime
import asyncio

router = APIRouter()

@router.post("/analyze", response_model=CustomerJourneyResponse)
async def analyze_customer_journey(
    request: CustomerJourneyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze the journey of a specific customer.
    """
    logger.info(f"Customer journey analysis requested for customer {request.customer_id} by user {current_user.id}")
    service = CustomerJourneyService(db)
    try:
        result = await service.analyze_journey(request.customer_id)
        logger.info(f"Customer journey analysis completed for customer {request.customer_id}")
        return CustomerJourneyResponse(**result.dict())
    except ValueError as ve:
        logger.error(f"Value error in customer journey analysis: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in customer journey analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing customer journey analysis")

@router.post("/touchpoint-analysis", response_model=TouchpointAnalysisResponse)
async def analyze_touchpoint(
    request: TouchpointAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze a specific touchpoint in the customer journey.
    """
    logger.info(f"Touchpoint analysis requested for touchpoint {request.touchpoint_id} by user {current_user.id}")
    service = CustomerJourneyService(db)
    try:
        result = await service.analyze_touchpoint(request.touchpoint_id)
        logger.info(f"Touchpoint analysis completed for touchpoint {request.touchpoint_id}")
        return TouchpointAnalysisResponse(**result.dict())
    except ValueError as ve:
        logger.error(f"Value error in touchpoint analysis: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
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
    """
    Optimize the customer journey for a specific segment.
    """
    logger.info(f"Journey optimization requested for segment {request.segment_id} by user {current_user.id}")
    service = CustomerJourneyService(db)
    try:
        result = await service.optimize_journey(request.segment_id)
        background_tasks.add_task(service.implement_journey_changes, result.optimization_id)
        logger.info(f"Journey optimization completed for segment {request.segment_id}")
        return JourneyOptimizationResponse(**result.dict())
    except ValueError as ve:
        logger.error(f"Value error in journey optimization: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in journey optimization: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing journey optimization")

@router.post("/segment-journey", response_model=List[JourneySegmentResponse])
async def segment_customer_journeys(
    request: JourneySegmentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Segment customer journeys into distinct groups.
    """
    logger.info(f"Journey segmentation requested by user {current_user.id}")
    service = CustomerJourneyService(db)
    try:
        results = await service.segment_journeys(request.num_segments)
        logger.info(f"Journey segmentation completed")
        return [JourneySegmentResponse(**result.dict()) for result in results]
    except ValueError as ve:
        logger.error(f"Value error in journey segmentation: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in journey segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing journey segmentation")

@router.get("/journey-metrics", response_model=JourneyMetricsResponse)
async def get_journey_metrics(
    start_date: datetime = Query(..., description="Start date for metrics calculation"),
    end_date: datetime = Query(..., description="End date for metrics calculation"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get aggregated metrics for customer journeys within a specified date range.
    """
    logger.info(f"Journey metrics requested by user {current_user.id}")
    service = CustomerJourneyService(db)
    try:
        metrics = await service.get_journey_metrics(start_date, end_date)
        return JourneyMetricsResponse(**metrics)
    except ValueError as ve:
        logger.error(f"Value error in retrieving journey metrics: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error retrieving journey metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving journey metrics")

@router.post("/predict-churn", response_model=ChurnPredictionResponse)
async def predict_customer_churn(
    customer_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Predict the likelihood of churn for a specific customer.
    """
    logger.info(f"Churn prediction requested for customer {customer_id} by user {current_user.id}")
    service = CustomerJourneyService(db)
    try:
        prediction = await service.predict_churn(customer_id)
        return ChurnPredictionResponse(**prediction)
    except ValueError as ve:
        logger.error(f"Value error in predicting customer churn: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error predicting customer churn: {str(e)}")
        raise HTTPException(status_code=500, detail="Error predicting customer churn")

@router.get("/visualize-journey/{customer_id}", response_class=HTMLResponse)
async def visualize_customer_journey(
    customer_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate a visual representation of a customer's journey.
    """
    logger.info(f"Journey visualization requested for customer {customer_id} by user {current_user.id}")
    service = CustomerJourneyService(db)
    try:
        visualization = await service.visualize_journey(customer_id)
        return visualization
    except ValueError as ve:
        logger.error(f"Value error in visualizing customer journey: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error visualizing customer journey: {str(e)}")
        raise HTTPException(status_code=500, detail="Error visualizing customer journey")

@router.post("/sentiment-analysis", response_model=SentimentAnalysisResponse)
async def analyze_journey_sentiment(
    customer_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze the sentiment throughout a customer's journey.
    """
    logger.info(f"Sentiment analysis requested for customer {customer_id} by user {current_user.id}")
    service = CustomerJourneyService(db)
    try:
        journey = await service.analyze_journey(customer_id)
        sentiment_analysis = await service.analyze_sentiment(journey.touchpoints)
        return SentimentAnalysisResponse(**sentiment_analysis)
    except ValueError as ve:
        logger.error(f"Value error in journey sentiment analysis: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in journey sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing journey sentiment analysis")

@router.get("/journey-report/{customer_id}", response_model=JourneyReportResponse)
async def generate_journey_report(
    customer_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate a comprehensive report of a customer's journey.
    """
    logger.info(f"Journey report requested for customer {customer_id} by user {current_user.id}")
    service = CustomerJourneyService(db)
    try:
        report = await service.generate_journey_report(customer_id)
        return JourneyReportResponse(**report)
    except ValueError as ve:
        logger.error(f"Value error in generating journey report: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error generating journey report: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating journey report")

@router.post("/train-churn-model")
async def train_churn_prediction_model(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Train the churn prediction model using the latest customer data.
    """
    logger.info(f"Churn prediction model training requested by user {current_user.id}")
    service = CustomerJourneyService(db)
    try:
        background_tasks.add_task(service.train_churn_prediction_model)
        return {"message": "Churn prediction model training started in the background"}
    except Exception as e:
        logger.error(f"Error starting churn prediction model training: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting churn prediction model training")
