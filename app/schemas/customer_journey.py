from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class TouchpointType(str, Enum):
    INTERACTION = "interaction"
    ORDER = "order"

class ChannelType(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    WEBSITE = "website"
    MOBILE_APP = "mobile_app"
    SOCIAL_MEDIA = "social_media"
    IN_STORE = "in_store"

class Touchpoint(BaseModel):
    type: TouchpointType
    timestamp: datetime
    channel: ChannelType
    details: Dict[str, Any]

class PainPoint(BaseModel):
    type: str
    description: str
    timestamp: Optional[datetime]

class CustomerJourneyRequest(BaseModel):
    customer_id: int = Field(..., description="The ID of the customer")

class CustomerJourneyResponse(BaseModel):
    customer_id: int
    touchpoints: List[Touchpoint]
    journey_duration: int = Field(..., description="Journey duration in days")
    conversion_rate: float
    pain_points: List[PainPoint]
    recommendations: List[str]
    sentiment_analysis: Dict[str, float]
    customer_value: float

class TouchpointAnalysisRequest(BaseModel):
    touchpoint_id: str = Field(..., description="The ID of the touchpoint to analyze")

class TouchpointAnalysisResponse(BaseModel):
    touchpoint_id: str
    engagement_rate: float
    conversion_impact: float
    average_time_spent: float
    customer_feedback: List[Dict[str, Any]]
    improvement_suggestions: List[str]

class JourneyOptimizationRequest(BaseModel):
    segment_id: str = Field(..., description="The ID of the customer segment to optimize")

class OptimizationSuggestion(BaseModel):
    type: str
    description: str
    priority: str

class ImplementationStep(BaseModel):
    change: str
    start_date: datetime
    end_date: datetime
    status: str
    priority: str

class JourneyOptimizationResponse(BaseModel):
    segment_id: str
    optimization_id: str
    suggested_changes: List[OptimizationSuggestion]
    expected_impact: Dict[str, float]
    implementation_timeline: List[ImplementationStep]

class JourneySegmentRequest(BaseModel):
    num_segments: int = Field(5, ge=2, le=10, description="Number of segments to create")

class JourneySegmentResponse(BaseModel):
    segment_id: str
    segment_name: str
    segment_size: int
    average_journey_duration: float
    common_touchpoints: List[str]
    conversion_rate: float
    average_customer_value: float

class JourneyMetricsResponse(BaseModel):
    total_interactions: int
    total_orders: int
    unique_customers: int
    conversion_rate: float
    channel_distribution: Dict[str, int]
    average_journey_duration: float
    total_revenue: float
    average_order_value: float
    repeat_purchase_rate: float

class ChurnPredictionResponse(BaseModel):
    customer_id: int
    churn_probability: float
    risk_factors: List[str]
    retention_suggestions: List[str]

class SentimentAnalysisResponse(BaseModel):
    overall_sentiment: float
    sentiment_by_channel: Dict[str, float]
    sentiment_trend: List[Dict[str, Any]]

class JourneyReportResponse(BaseModel):
    customer_id: int
    journey_summary: Dict[str, Any]
    key_metrics: Dict[str, float]
    touchpoint_analysis: List[Dict[str, Any]]
    sentiment_analysis: Dict[str, Any]
    recommendations: List[str]

class ModelTrainingStatus(BaseModel):
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    accuracy: Optional[float]
    error_message: Optional[str]
