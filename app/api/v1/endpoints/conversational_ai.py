from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.core.security import get_current_user
from app.schemas.conversational_ai import (
    ConversationRequest,
    ConversationResponse,
    ConversationHistory,
    IntentAnalysisRequest,
    IntentAnalysisResponse,
    PerformanceMetrics,
    PopularIntent,
    SentimentDistribution,
    ConversationSummary
)
from app.services.conversational_ai import ConversationalAIService
from app.models.user import User
from app.core.logging import logger
from typing import List
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/chat", response_model=ConversationResponse)
async def chat(
    request: ConversationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate a chat response based on the user's input message.
    """
    logger.info(f"Chat request received from user {current_user.id}")
    service = ConversationalAIService(db)
    try:
        response = await service.generate_response(
            message=request.message,
            conversation_id=request.conversation_id,
            customer_id=current_user.id
        )
        logger.info(f"Chat response generated for user {current_user.id}")
        return response
    except Exception as e:
        logger.error(f"Error in chat generation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating chat response")

@router.get("/history/{conversation_id}", response_model=ConversationHistory)
async def get_conversation_history(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve the conversation history for a specific conversation ID.
    """
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
    """
    Analyze the intent of a given message.
    """
    logger.info(f"Intent analysis requested by user {current_user.id}")
    service = ConversationalAIService(db)
    try:
        intent = await service.analyze_intent(request.message)
        return IntentAnalysisResponse(intent=intent.intent, confidence=intent.confidence)
    except Exception as e:
        logger.error(f"Error in intent analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error analyzing intent")

@router.post("/train")
async def train_model(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Start the model training process in the background.
    """
    logger.info(f"Model training requested by user {current_user.id}")
    service = ConversationalAIService(db)
    try:
        background_tasks.add_task(service.train_model)
        return JSONResponse(content={"status": "Model training started in the background"}, status_code=202)
    except Exception as e:
        logger.error(f"Error starting model training: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting model training")

@router.get("/performance", response_model=PerformanceMetrics)
async def get_model_performance(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve the performance metrics of the conversational AI model.
    """
    logger.info(f"Model performance metrics requested by user {current_user.id}")
    service = ConversationalAIService(db)
    try:
        performance = await service.get_performance_metrics()
        return PerformanceMetrics(**performance)
    except Exception as e:
        logger.error(f"Error retrieving model performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving model performance metrics")

@router.get("/popular-intents", response_model=List[PopularIntent])
async def get_popular_intents(
    limit: int = 5,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve the most popular intents, limited by the specified number.
    """
    logger.info(f"Popular intents requested by user {current_user.id}")
    service = ConversationalAIService(db)
    try:
        popular_intents = await service.get_popular_intents(limit)
        return [PopularIntent(**intent) for intent in popular_intents]
    except Exception as e:
        logger.error(f"Error retrieving popular intents: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving popular intents")

@router.get("/sentiment-distribution", response_model=SentimentDistribution)
async def get_sentiment_distribution(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve the distribution of sentiments across all conversations.
    """
    logger.info(f"Sentiment distribution requested by user {current_user.id}")
    service = ConversationalAIService(db)
    try:
        distribution = await service.get_sentiment_distribution()
        return SentimentDistribution(**distribution)
    except Exception as e:
        logger.error(f"Error retrieving sentiment distribution: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving sentiment distribution")

@router.get("/conversation-summary/{conversation_id}", response_model=ConversationSummary)
async def get_conversation_summary(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve a summary of a specific conversation.
    """
    logger.info(f"Conversation summary requested for conversation {conversation_id} by user {current_user.id}")
    service = ConversationalAIService(db)
    try:
        summary = await service.get_conversation_summary(conversation_id)
        return ConversationSummary(**summary)
    except Exception as e:
        logger.error(f"Error retrieving conversation summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving conversation summary")
