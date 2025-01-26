from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

class SentimentType(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

class SentimentRequest(BaseModel):
    text: str = Field(..., description="The text to analyze for sentiment")

class SentimentResult(BaseModel):
    sentiment: SentimentType = Field(..., description="The detected sentiment")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score of the sentiment analysis")
    text: str = Field(..., description="The input text that was analyzed")

class SentimentResponse(SentimentResult):
    pass

class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze for sentiment")

class SentimentDistribution(BaseModel):
    distribution: Dict[SentimentType, float] = Field(..., description="Distribution of sentiments")

class PerformanceMetrics(BaseModel):
    accuracy: float = Field(..., ge=0, le=1, description="Accuracy of the sentiment analysis model")
    precision: float = Field(..., ge=0, le=1, description="Precision of the sentiment analysis model")
    recall: float = Field(..., ge=0, le=1, description="Recall of the sentiment analysis model")
    f1_score: float = Field(..., ge=0, le=1, description="F1 score of the sentiment analysis model")

class SentimentTrendPoint(BaseModel):
    timestamp: datetime = Field(..., description="Timestamp of the trend point")
    count: int = Field(..., ge=0, description="Count of sentiments at this timestamp")

class SentimentTrends(BaseModel):
    trends: Dict[SentimentType, List[SentimentTrendPoint]] = Field(..., description="Trends of sentiments over time")

class FeedbackImpact(BaseModel):
    average_order_value: float = Field(..., ge=0, description="Average order value for this sentiment")
    order_count: int = Field(..., ge=0, description="Number of orders associated with this sentiment")

class FeedbackImpactAnalysis(BaseModel):
    impact: Dict[SentimentType, FeedbackImpact] = Field(..., description="Impact of sentiment on customer behavior")

class ModelRetrainingRequest(BaseModel):
    training_data: List[Dict[str, str]] = Field(..., description="List of training data items with 'text' and 'sentiment' keys")

class ModelTrainingStatus(BaseModel):
    status: str = Field(..., description="Current status of model training")
    start_time: datetime = Field(..., description="Start time of the training process")
    end_time: Optional[datetime] = Field(None, description="End time of the training process")
    progress: Optional[float] = Field(None, ge=0, le=1, description="Training progress (0-1)")
    error_message: Optional[str] = Field(None, description="Error message if training failed")

class SentimentAnalysisSettings(BaseModel):
    model_name: str = Field(..., description="Name of the sentiment analysis model to use")
    confidence_threshold: float = Field(0.6, ge=0, le=1, description="Confidence threshold for sentiment classification")
    use_gpu: bool = Field(False, description="Whether to use GPU for sentiment analysis")

class DetailedSentimentResponse(SentimentResponse):
    keyword_sentiments: Dict[str, SentimentType] = Field(..., description="Sentiment analysis for key phrases in the text")
    entity_sentiments: Dict[str, SentimentType] = Field(..., description="Sentiment analysis for named entities in the text")

class SentimentComparison(BaseModel):
    text1: str = Field(..., description="First text for comparison")
    text2: str = Field(..., description="Second text for comparison")
    sentiment1: SentimentType = Field(..., description="Sentiment of the first text")
    sentiment2: SentimentType = Field(..., description="Sentiment of the second text")
    similarity_score: float = Field(..., ge=0, le=1, description="Similarity score between the two texts")

class SentimentTimeseriesPoint(BaseModel):
    timestamp: datetime = Field(..., description="Timestamp of the sentiment measurement")
    sentiment: SentimentType = Field(..., description="Sentiment at this timestamp")
    confidence: float = Field(..., ge=0, le=1, description="Confidence of the sentiment analysis")

class SentimentTimeseries(BaseModel):
    customer_id: int = Field(..., description="ID of the customer")
    timeseries: List[SentimentTimeseriesPoint] = Field(..., description="Timeseries of sentiment measurements")

class AggregatedSentimentReport(BaseModel):
    total_analyzed: int = Field(..., ge=0, description="Total number of texts analyzed")
    sentiment_distribution: Dict[SentimentType, float] = Field(..., description="Distribution of sentiments")
    average_confidence: float = Field(..., ge=0, le=1, description="Average confidence score across all analyses")
    most_positive_text: str = Field(..., description="Text with the highest positive sentiment score")
    most_negative_text: str = Field(..., description="Text with the highest negative sentiment score")
    common_positive_keywords: List[str] = Field(..., description="Most common keywords in positive sentiments")
    common_negative_keywords: List[str] = Field(..., description="Most common keywords in negative sentiments")

