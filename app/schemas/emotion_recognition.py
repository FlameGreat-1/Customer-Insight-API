from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class EmotionEnum(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"

class EmotionRecognitionRequest(BaseModel):
    text: str = Field(..., description="The text to analyze for emotion")

class EmotionRecognitionResponse(BaseModel):
    text: str = Field(..., description="The input text that was analyzed")
    primary_emotion: EmotionEnum = Field(..., description="The primary detected emotion")
    emotion_distribution: Dict[EmotionEnum, float] = Field(..., description="Distribution of emotion probabilities")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score of the emotion recognition")

class BatchEmotionRecognitionRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze for emotion")

class EmotionModel(BaseModel):
    id: str = Field(..., description="Unique identifier for the emotion recognition model")
    name: str = Field(..., description="Name of the emotion recognition model")
    description: str = Field(..., description="Description of the emotion recognition model")

class EmotionDistribution(BaseModel):
    distribution: Dict[EmotionEnum, float] = Field(..., description="Distribution of emotions over a period")

class PerformanceMetrics(BaseModel):
    total_recognitions: int = Field(..., ge=0, description="Total number of emotion recognitions performed")
    average_confidence: float = Field(..., ge=0, le=1, description="Average confidence score of recognitions")
    accuracy: float = Field(..., ge=0, le=1, description="Accuracy of the emotion recognition model")
    precision: float = Field(..., ge=0, le=1, description="Precision of the emotion recognition model")
    recall: float = Field(..., ge=0, le=1, description="Recall of the emotion recognition model")
    f1_score: float = Field(..., ge=0, le=1, description="F1 score of the emotion recognition model")

class EmotionTrend(BaseModel):
    emotion: EmotionEnum = Field(..., description="The emotion being tracked")
    trend: List[Dict[str, Any]] = Field(..., description="List of data points showing the trend")

class CustomerEmotionProfile(BaseModel):
    customer_id: int = Field(..., description="The ID of the customer")
    emotions: Dict[EmotionEnum, Dict[str, float]] = Field(..., description="Breakdown of emotions for the customer")
    dominant_emotion: EmotionEnum = Field(..., description="The most frequent emotion for the customer")
    average_confidence: float = Field(..., ge=0, le=1, description="Average confidence of emotion recognitions")

class EmotionImpactAnalysis(BaseModel):
    emotion: EmotionEnum = Field(..., description="The emotion being analyzed")
    interaction_count: int = Field(..., ge=0, description="Number of interactions associated with this emotion")
    average_interaction_duration: float = Field(..., ge=0, description="Average duration of interactions in seconds")
    average_order_value: float = Field(..., ge=0, description="Average order value associated with this emotion")

class AudioEmotionRecognitionRequest(BaseModel):
    file_path: str = Field(..., description="Path to the audio file for emotion recognition")

class ModelTrainingStatus(BaseModel):
    status: str = Field(..., description="Current status of model training")
    start_time: datetime = Field(..., description="Start time of the training process")
    end_time: Optional[datetime] = Field(None, description="End time of the training process")
    progress: Optional[float] = Field(None, ge=0, le=1, description="Training progress (0-1)")
    error_message: Optional[str] = Field(None, description="Error message if training failed")

class EmotionRecognitionSettings(BaseModel):
    model_id: str = Field(..., description="ID of the emotion recognition model to use")
    confidence_threshold: float = Field(0.5, ge=0, le=1, description="Minimum confidence threshold for recognition")
    include_distribution: bool = Field(True, description="Whether to include emotion distribution in results")

class EmotionAlert(BaseModel):
    customer_id: int = Field(..., description="The ID of the customer")
    emotion: EmotionEnum = Field(..., description="The detected emotion that triggered the alert")
    confidence: float = Field(..., ge=0, le=1, description="Confidence of the emotion recognition")
    timestamp: datetime = Field(..., description="Timestamp of the emotion detection")
    context: Optional[str] = Field(None, description="Context or source of the emotion (e.g., 'customer_support_call')")

