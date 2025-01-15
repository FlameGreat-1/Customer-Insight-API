from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.security import get_current_user
from app.schemas.conversational_ai import (
    ConversationRequest,
    ConversationResponse,
    ConversationHistory,
    IntentAnalysisRequest,
    IntentAnalysisResponse
)
from app.services.conversational_ai import ConversationalAIService
from app.models.user import User
from app.core.logging import logger
from typing import List
import asyncio

router = APIRouter()

@router.post("/chat", response_model=ConversationResponse)
async def chat(
    request: ConversationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Chat request received from user {current_user.id}")
    service = ConversationalAIService(db)
    try:
        response = await service.generate_response(request.message, request.conversation_id)
        logger.info(f"Chat response generated for user {current_user.id}")
        return ConversationResponse(
            message=response.message,
            conversation_id=response.conversation_id,
            confidence=response.confidence
        )
    except Exception as e:
        logger.error(f"Error in chat generation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating chat response")

@router.get("/history/{conversation_id}", response_model=ConversationHistory)
async def get_conversation_history(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Conversation history requested for conversation {conversation_id} by user {current_user.id}")
    service = ConversationalAIService(db)
    try:
        history = await service.get_conversation_history(conversation_id)
        return ConversationHistory(conversation_id=conversation_id, messages=history)
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving conversation history")

@router.post("/analyze-intent", response_model=IntentAnalysisResponse)
async def analyze_intent(
    request: IntentAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Intent analysis requested by user {current_user.id}")
    service = ConversationalAIService(db)
    try:
        intent = await service.analyze_intent(request.message)
        return IntentAnalysisResponse(intent=intent.intent, confidence=intent.confidence)
    except Exception as e:
        logger.error(f"Error in intent analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error analyzing intent")

@router.post("/train", response_model=dict)
async def train_model(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Model training requested by user {current_user.id}")
    service = ConversationalAIService(db)
    try:
        background_tasks.add_task(service.train_model)
        return {"status": "Model training started in the background"}
    except Exception as e:
        logger.error(f"Error starting model training: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting model training")

@router.get("/performance", response_model=dict)
async def get_model_performance(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Model performance metrics requested by user {current_user.id}")
    service = ConversationalAIService(db)
    try:
        performance = await service.get_performance_metrics()
        return performance
    except Exception as e:
        logger.error(f"Error retrieving model performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving model performance metrics")
