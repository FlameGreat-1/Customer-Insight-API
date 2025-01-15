from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.security import get_current_user
from app.schemas.emotion_recognition import (
    EmotionRecognitionRequest,
    EmotionRecognitionResponse,
    BatchEmotionRecognitionRequest,
    EmotionModel,
    EmotionDistribution
)
from app.services.emotion_recognition import EmotionRecognitionService
from app.models.user import User
from app.core.logging import logger
from typing import List
import asyncio
import aiofiles
import os

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
        return EmotionRecognitionResponse(
            text=request.text,
            primary_emotion=result.primary_emotion,
            emotion_distribution=result.emotion_distribution,
            confidence=result.confidence
        )
    except Exception as e:
        logger.error(f"Error in emotion recognition: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing emotion recognition")

@router.post("/batch-recognize", response_model=List[EmotionRecognitionResponse])
async def batch_recognize_emotion(
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
        return [
            EmotionRecognitionResponse(
                text=text,
                primary_emotion=result.primary_emotion,
                emotion_distribution=result.emotion_distribution,
                confidence=result.confidence
            )
            for result, text in zip(results, request.texts)
        ]
    except Exception as e:
        logger.error(f"Error in batch emotion recognition: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing batch emotion recognition")

@router.post("/recognize-audio", response_model=EmotionRecognitionResponse)
async def recognize_emotion_from_audio(
    audio: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Audio emotion recognition requested by user {current_user.id}")
    service = EmotionRecognitionService(db)
    try:
        file_location = f"/tmp/{audio.filename}"
        async with aiofiles.open(file_location, 'wb') as out_file:
            content = await audio.read()
            await out_file.write(content)

        result = await service.recognize_emotion_from_audio(file_location)
        os.remove(file_location)

        logger.info(f"Audio emotion recognition completed for user {current_user.id}")
        return EmotionRecognitionResponse(
            text="Audio file",
            primary_emotion=result.primary_emotion,
            emotion_distribution=result.emotion_distribution,
            confidence=result.confidence
        )
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
        return [EmotionModel(id=model.id, name=model.name, description=model.description) for model in models]
    except Exception as e:
        logger.error(f"Error retrieving emotion models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving emotion models")

@router.post("/train", response_model=dict)
async def train_emotion_model(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Emotion model training requested by user {current_user.id}")
    service = EmotionRecognitionService(db)
    try:
        background_tasks.add_task(service.train_model)
        return {"status": "Emotion model training started in the background"}
    except Exception as e:
        logger.error(f"Error starting emotion model training: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting emotion model training")

@router.get("/performance", response_model=dict)
async def get_emotion_model_performance(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Emotion model performance metrics requested by user {current_user.id}")
    service = EmotionRecognitionService(db)
    try:
        performance = await service.get_performance_metrics()
        return performance
    except Exception as e:
        logger.error(f"Error retrieving emotion model performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving emotion model performance metrics")

@router.get("/emotion-distribution", response_model=EmotionDistribution)
async def get_emotion_distribution(
    start_date: str,
    end_date: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Emotion distribution requested by user {current_user.id}")
    service = EmotionRecognitionService(db)
    try:
        distribution = await service.get_emotion_distribution(start_date, end_date)
        return EmotionDistribution(distribution=distribution)
    except Exception as e:
        logger.error(f"Error retrieving emotion distribution: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving emotion distribution")
