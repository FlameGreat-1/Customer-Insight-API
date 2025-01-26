from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

class SegmentEnum(str, Enum):
    SEGMENT_0 = "Segment_0"
    SEGMENT_1 = "Segment_1"
    SEGMENT_2 = "Segment_2"
    SEGMENT_3 = "Segment_3"
    SEGMENT_4 = "Segment_4"

class FeatureEnum(str, Enum):
    TOTAL_SPEND = "total_spend"
    LOYALTY_POINTS = "loyalty_points"
    RECENCY = "recency"
    FREQUENCY = "frequency"
    AVG_ORDER_VALUE = "avg_order_value"

class SegmentationRequest(BaseModel):
    customer_id: int = Field(..., description="The ID of the customer to segment")

class SegmentationResponse(BaseModel):
    customer_id: int = Field(..., description="The ID of the segmented customer")
    segment: SegmentEnum = Field(..., description="The assigned segment")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score of the segmentation")
    features: Dict[FeatureEnum, float] = Field(..., description="Feature values used for segmentation")

class BatchSegmentationRequest(BaseModel):
    customer_ids: List[int] = Field(..., description="List of customer IDs to segment")

class SegmentationModel(BaseModel):
    id: str = Field(..., description="Unique identifier for the segmentation model")
    name: str = Field(..., description="Name of the segmentation model")
    description: str = Field(..., description="Description of the segmentation model")

class SegmentUpdateRequest(BaseModel):
    customer_id: int = Field(..., description="The ID of the customer to update")
    new_segment: SegmentEnum = Field(..., description="The new segment to assign to the customer")

class SegmentCharacteristics(BaseModel):
    segment: SegmentEnum = Field(..., description="The segment being described")
    characteristics: Dict[FeatureEnum, Dict[str, float]] = Field(..., description="Statistical characteristics of the segment")

class PerformanceMetrics(BaseModel):
    inertia: float = Field(..., description="Sum of squared distances of samples to their closest cluster center")
    silhouette_score: float = Field(..., ge=-1, le=1, description="Measure of how similar an object is to its own cluster compared to other clusters")
    num_clusters: int = Field(..., gt=0, description="Number of clusters in the model")

class SegmentTransitionMatrix(BaseModel):
    matrix: Dict[SegmentEnum, Dict[SegmentEnum, int]] = Field(..., description="Matrix showing transitions between segments")

class CohortAnalysisRequest(BaseModel):
    cohort_period: str = Field(..., description="Period for cohort analysis (e.g., 'monthly', 'quarterly')")

class FeatureImportance(BaseModel):
    feature_importance: Dict[FeatureEnum, float] = Field(..., description="Importance score of each feature in the segmentation model")

class SegmentDistribution(BaseModel):
    distribution: Dict[SegmentEnum, float] = Field(..., description="Distribution of customers across segments")

class ModelTrainingStatus(BaseModel):
    status: str = Field(..., description="Current status of model training")
    progress: Optional[float] = Field(None, ge=0, le=1, description="Progress of model training")
    error: Optional[str] = Field(None, description="Error message if training failed")

class OptimizationRequest(BaseModel):
    max_clusters: int = Field(10, ge=2, le=20, description="Maximum number of clusters to consider during optimization")

class CohortAnalysisResult(BaseModel):
    cohort_data: Dict[str, Dict[str, Dict[SegmentEnum, int]]] = Field(..., description="Cohort analysis data")

class CustomerSegmentHistory(BaseModel):
    customer_id: int = Field(..., description="The ID of the customer")
    segment_history: List[Dict[str, Union[datetime, SegmentEnum]]] = Field(..., description="History of segment changes")

class SegmentationInsights(BaseModel):
    most_valuable_segment: SegmentEnum = Field(..., description="Segment with the highest average total spend")
    fastest_growing_segment: SegmentEnum = Field(..., description="Segment with the highest growth rate")
    churn_risk_segment: SegmentEnum = Field(..., description="Segment with the highest churn risk")
    recommendations: Dict[SegmentEnum, List[str]] = Field(..., description="Recommended actions for each segment")

class RealTimeSegmentationEvent(BaseModel):
    customer_id: int = Field(..., description="The ID of the customer")
    old_segment: Optional[SegmentEnum] = Field(None, description="Previous segment of the customer")
    new_segment: SegmentEnum = Field(..., description="New segment of the customer")
    timestamp: datetime = Field(..., description="Timestamp of the segmentation event")

    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.now()

class SegmentationAPIStatus(BaseModel):
    status: str = Field(..., description="Current status of the segmentation API")
    model_version: str = Field(..., description="Version of the current segmentation model")
    last_training_date: datetime = Field(..., description="Date and time of the last model training")
    total_customers_segmented: int = Field(..., ge=0, description="Total number of customers segmented")
    average_confidence_score: float = Field(..., ge=0, le=1, description="Average confidence score of segmentations")

class SegmentationExplanation(BaseModel):
    customer_id: int = Field(..., description="The ID of the customer")
    segment: SegmentEnum = Field(..., description="Assigned segment")
    feature_contributions: Dict[FeatureEnum, float] = Field(..., description="Contribution of each feature to the segmentation decision")
    similar_customers: List[int] = Field(..., max_items=5, description="IDs of the 5 most similar customers in the same segment")
