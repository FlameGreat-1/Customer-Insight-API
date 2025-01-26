from fastapi import APIRouter, Depends
from app.api.v1.endpoints import (
    sentiment_analysis,
    customer_segmentation,
    recommendation_engine,
    conversational_ai,
    predictive_analytics,
    emotion_recognition,
    customer_journey,
    privacy_compliance
)
from app.core.config import settings
from app.core.security import get_current_user
from app.models.user import User

router = APIRouter()

router.include_router(sentiment_analysis.router, prefix="/sentiment", tags=["Sentiment Analysis"])
router.include_router(customer_segmentation.router, prefix="/segmentation", tags=["Customer Segmentation"])
router.include_router(recommendation_engine.router, prefix="/recommendation", tags=["Recommendation Engine"])
router.include_router(conversational_ai.router, prefix="/conversational", tags=["Conversational AI"])
router.include_router(predictive_analytics.router, prefix="/predictive", tags=["Predictive Analytics"])
router.include_router(emotion_recognition.router, prefix="/emotion", tags=["Emotion Recognition"])
router.include_router(customer_journey.router, prefix="/journey", tags=["Customer Journey"])
router.include_router(privacy_compliance.router, prefix="/privacy", tags=["Privacy Compliance"])

@router.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "healthy", "version": settings.VERSION}

@router.get("/info", tags=["API Information"])
async def api_info():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": settings.PROJECT_DESCRIPTION,
        "contact": {
            "name": settings.CONTACT_NAME,
            "email": settings.CONTACT_EMAIL,
            "url": settings.CONTACT_URL
        },
        "license": {
            "name": settings.LICENSE_NAME,
            "url": settings.LICENSE_URL
        }
    }

@router.get("/rate-limit-info", tags=["API Information"])
async def rate_limit_info():
    return {
        "rate_limit": settings.RATE_LIMIT_PER_MINUTE,
        "rate_limit_period": "1 minute"
    }

@router.get("/user-info", tags=["User Information"])
async def user_info(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "is_active": current_user.is_active,
        "is_superuser": current_user.is_superuser
    }

@router.get("/endpoints", tags=["API Information"])
async def list_endpoints():
    return {
        "endpoints": [
            {"path": "/sentiment", "description": "Sentiment Analysis endpoints"},
            {"path": "/segmentation", "description": "Customer Segmentation endpoints"},
            {"path": "/recommendation", "description": "Recommendation Engine endpoints"},
            {"path": "/conversational", "description": "Conversational AI endpoints"},
            {"path": "/predictive", "description": "Predictive Analytics endpoints"},
            {"path": "/emotion", "description": "Emotion Recognition endpoints"},
            {"path": "/journey", "description": "Customer Journey endpoints"},
            {"path": "/privacy", "description": "Privacy Compliance endpoints"}
        ]
    }
