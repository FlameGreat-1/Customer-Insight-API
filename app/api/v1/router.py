from fastapi import APIRouter
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
