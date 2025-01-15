from .sentiment_analysis import router as sentiment_router
from .customer_segmentation import router as segmentation_router
from .recommendation_engine import router as recommendation_router
from .conversational_ai import router as conversational_router
from .predictive_analytics import router as predictive_router
from .emotion_recognition import router as emotion_router
from .customer_journey import router as journey_router
from .privacy_compliance import router as privacy_router

__all__ = [
    "sentiment_router",
    "segmentation_router",
    "recommendation_router",
    "conversational_router",
    "predictive_router",
    "emotion_router",
    "journey_router",
    "privacy_router",
]
