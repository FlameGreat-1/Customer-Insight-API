from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile, Query
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.security import get_current_user
from app.schemas.emotion_recognition import (
    EmotionRecognitionRequest,
    EmotionRecognitionResponse,
    BatchEmotionRecognitionRequest,
    EmotionModel,
    EmotionDistribution,
    PerformanceMetrics,
    EmotionTrend,
    CustomerEmotionProfile,
    EmotionImpactAnalysis,
    AudioEmotionRecognitionRequest,
    ModelTrainingStatus,
    EmotionRecognitionSettings,
    EmotionAlert
)
from app.services.emotion_recognition import EmotionRecognitionService
from app.models.user import User
from app.core.logging import logger
from typing import List, Dict
import asyncio
import aiofiles
import os
from datetime import datetime

router = APIRouter()

@router.post("/recognize", response_model=EmotionRecognitionResponse)
async def recognize_emotion(
    request: EmotionRecognitionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Emotion recognition requested by user {current_user.id}")
    service = EmotionRecognitionService(db)
    try:
        result = await service.recognize_emotion(request.text)
        logger.info(f"Emotion recognition completed for user {current_user.id}")
        return EmotionRecognitionResponse(**result.dict())
    except Exception as e:
        logger.error(f"Error in emotion recognition: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing emotion recognition")

@router.post("/recognize-batch", response_model=List[EmotionRecognitionResponse])
async def recognize_emotions_batch(
    request: BatchEmotionRecognitionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Batch emotion recognition requested by user {current_user.id}")
    service = EmotionRecognitionService(db)
    try:
        results = await asyncio.gather(*[service.recognize_emotion(text) for text in request.texts])
        background_tasks.add_task(service.store_batch_results, results, current_user.id)
        logger.info(f"Batch emotion recognition completed for user {current_user.id}")
        return [EmotionRecognitionResponse(**result.dict()) for result in results]
    except Exception as e:
        logger.error(f"Error in batch emotion recognition: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing batch emotion recognition")

@router.post("/recognize-audio", response_model=EmotionRecognitionResponse)
async def recognize_emotion_from_audio(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Audio emotion recognition requested by user {current_user.id}")
    service = EmotionRecognitionService(db)
    try:
        file_location = f"temp_{file.filename}"
        async with aiofiles.open(file_location, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        result = await service.recognize_emotion_from_audio(file_location)
        os.remove(file_location)
        
        logger.info(f"Audio emotion recognition completed for user {current_user.id}")
        return EmotionRecognitionResponse(**result.dict())
    except Exception as e:
        logger.error(f"Error in audio emotion recognition: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing audio emotion recognition")

@router.get("/models", response_model=List[EmotionModel])
async def get_emotion_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Emotion models requested by user {current_user.id}")
    service = EmotionRecognitionService(db)
    try:
        models = await service.get_available_models()
        return models
    except Exception as e:
        logger.error(f"Error retrieving emotion models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving emotion models")

@router.post("/train", response_model=ModelTrainingStatus)
async def train_emotion_model(
    model_type: str = Query(..., description="Type of model to train (text or audio)"),
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Emotion model training requested by user {current_user.id}")
    service = EmotionRecognitionService(db)
    try:
        background_tasks.add_task(service.train_model, model_type)
        return ModelTrainingStatus(
            status="started",
            start_time=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error starting emotion model training: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting emotion model training")

@router.get("/performance", response_model=PerformanceMetrics)
async def get_emotion_recognition_performance(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Emotion recognition performance metrics requested by user {current_user.id}")
    service = EmotionRecognitionService(db)
    try:
        metrics = await service.get_performance_metrics()
        return PerformanceMetrics(**metrics)
    except Exception as e:
        logger.error(f"Error retrieving performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving performance metrics")

@router.get("/distribution", response_model=EmotionDistribution)
async def get_emotion_distribution(
    start_date: datetime = Query(..., description="Start date for distribution calculation"),
    end_date: datetime = Query(..., description="End date for distribution calculation"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Emotion distribution requested by user {current_user.id}")
    service = EmotionRecognitionService(db)
    try:
        distribution = await service.get_emotion_distribution(start_date, end_date)
        return distribution
    except Exception as e:
        logger.error(f"Error retrieving emotion distribution: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving emotion distribution")

@router.get("/trends", response_model=Dict[str, List[Dict[str, Any]]])
async def get_emotion_trends(
    start_date: datetime = Query(..., description="Start date for trend calculation"),
    end_date: datetime = Query(..., description="End date for trend calculation"),
    interval: str = Query("day", description="Interval for trend data points (hour, day, week, month)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Emotion trends requested by user {current_user.id}")
    service = EmotionRecognitionService(db)
    try:
        trends = await service.get_emotion_trends(start_date, end_date, interval)
        return trends
    except Exception as e:
        logger.error(f"Error retrieving emotion trends: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving emotion trends")

@router.get("/customer-profile/{customer_id}", response_model=CustomerEmotionProfile)
async def get_customer_emotion_profile(
    customer_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Customer emotion profile requested for customer {customer_id} by user {current_user.id}")
    service = EmotionRecognitionService(db)
    try:
        profile = await service.get_customer_emotion_profile(customer_id)
        return CustomerEmotionProfile(customer_id=customer_id, **profile)
    except Exception as e:
        logger.error(f"Error retrieving customer emotion profile: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving customer emotion profile")

@router.get("/impact-analysis", response_model=Dict[str, EmotionImpactAnalysis])
async def analyze_emotion_impact(
    start_date: datetime = Query(..., description="Start date for impact analysis"),
    end_date: datetime = Query(..., description="End date for impact analysis"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Emotion impact analysis requested by user {current_user.id}")
    service = EmotionRecognitionService(db)
    try:
        impact_analysis = await service.analyze_emotion_impact(start_date, end_date)
        return {emotion: EmotionImpactAnalysis(emotion=emotion, **data) for emotion, data in impact_analysis.items()}
    except Exception as e:
        logger.error(f"Error analyzing emotion impact: {str(e)}")
        raise HTTPException(status_code=500, detail="Error analyzing emotion impact")

@router.post("/settings", response_model=EmotionRecognitionSettings)
async def update_emotion_recognition_settings(
    settings: EmotionRecognitionSettings,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Emotion recognition settings update requested by user {current_user.id}")
    service = EmotionRecognitionService(db)
    try:
        updated_settings = await service.update_settings(settings)
        return updated_settings
    except Exception as e:
        logger.error(f"Error updating emotion recognition settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Error updating emotion recognition settings")

@router.get("/alerts", response_model=List[EmotionAlert])
async def get_emotion_alerts(
    start_date: datetime = Query(..., description="Start date for alerts"),
    end_date: datetime = Query(..., description="End date for alerts"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Emotion alerts requested by user {current_user.id}")
    service = EmotionRecognitionService(db)
    try:
        alerts = await service.get_emotion_alerts(start_date, end_date)
        return alerts
    except Exception as e:
        logger.error(f"Error retrieving emotion alerts: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving emotion alerts")
