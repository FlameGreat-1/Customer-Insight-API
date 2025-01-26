from pydantic import BaseModel, Field
from typing import List, Optional, Union
from enum import Enum

class PredictionType(str, Enum):
    CHURN = "churn"
    LTV = "ltv"

class PredictionRequest(BaseModel):
    customer_id: int = Field(..., description="The ID of the customer to make a prediction for")
    prediction_type: PredictionType = Field(..., description="The type of prediction to make")

class PredictionResult(BaseModel):
    customer_id: int
    prediction_type: PredictionType
    prediction: Union[bool, float]
    confidence: float
    explanation: str

class PredictionResponse(PredictionResult):
    pass

class BatchPredictionRequest(BaseModel):
    customer_ids: List[int] = Field(..., description="List of customer IDs to make predictions for")
    prediction_type: PredictionType = Field(..., description="The type of prediction to make for all customers")

class ModelPerformance(BaseModel):
    prediction_type: PredictionType
    accuracy: Optional[float] = Field(None, description="Accuracy score for classification models")
    precision: Optional[float] = Field(None, description="Precision score for classification models")
    recall: Optional[float] = Field(None, description="Recall score for classification models")
    f1_score: Optional[float] = Field(None, description="F1 score for classification models")
    mse: Optional[float] = Field(None, description="Mean Squared Error for regression models")
    r2: Optional[float] = Field(None, description="R-squared score for regression models")
    cross_validation_score: float = Field(..., description="Cross-validation score")

class FeatureImportance(BaseModel):
    feature: str = Field(..., description="Name of the feature")
    importance: float = Field(..., description="Importance score of the feature")

class CustomerSegment(str, Enum):
    HIGH_RISK = "High Risk"
    MEDIUM_RISK = "Medium Risk"
    HIGH_VALUE = "High Value"
    MEDIUM_VALUE = "Medium Value"
    LOW_VALUE = "Low Value"

class CustomerSegmentResponse(BaseModel):
    customer_id: int
    segment: CustomerSegment

class CustomerRecommendationsResponse(BaseModel):
    customer_id: int
    recommendations: List[str]

class ModelTrainingStatus(BaseModel):
    prediction_type: PredictionType
    status: str = Field(..., description="Current status of the model training")
    progress: Optional[float] = Field(None, ge=0, le=1, description="Training progress (0-1)")
    error_message: Optional[str] = Field(None, description="Error message if training failed")

class FeatureEngineering(BaseModel):
    feature_name: str = Field(..., description="Name of the engineered feature")
    description: str = Field(..., description="Description of how the feature is calculated")
    importance: Optional[float] = Field(None, description="Importance score of the feature if available")

class ModelComparison(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    performance_metrics: ModelPerformance
    training_time: float = Field(..., description="Time taken to train the model in seconds")
    prediction_time: float = Field(..., description="Average time to make a prediction in milliseconds")

class PredictionExplanation(BaseModel):
    feature: str = Field(..., description="Name of the feature")
    value: Union[float, str] = Field(..., description="Value of the feature for this prediction")
    impact: float = Field(..., description="Impact of this feature on the prediction")
    direction: str = Field(..., description="Whether this feature increased or decreased the prediction")

class DetailedPredictionResponse(PredictionResponse):
    feature_explanations: List[PredictionExplanation] = Field(..., description="Detailed explanations for each feature's impact on the prediction")

class AggregatedPredictions(BaseModel):
    prediction_type: PredictionType
    total_predictions: int = Field(..., description="Total number of predictions made")
    average_confidence: float = Field(..., description="Average confidence score across all predictions")
    distribution: Dict[Union[bool, str], int] = Field(..., description="Distribution of prediction outcomes")

class PredictionThresholds(BaseModel):
    prediction_type: PredictionType
    current_threshold: float = Field(..., description="Current threshold used for classification")
    recommended_threshold: float = Field(..., description="Recommended threshold based on recent data")
    precision_at_threshold: float = Field(..., description="Precision score at the recommended threshold")
    recall_at_threshold: float = Field(..., description="Recall score at the recommended threshold")

