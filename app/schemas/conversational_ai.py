from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class IntentEnum(str, Enum):
    GREETING = "greeting"
    FAREWELL = "farewell"
    PRODUCT_INQUIRY = "product_inquiry"
    COMPLAINT = "complaint"
    GENERAL_INQUIRY = "general_inquiry"

class SentimentEnum(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

class ConversationRequest(BaseModel):
    message: str = Field(..., description="The user's input message")
    conversation_id: Optional[str] = Field(None, description="The ID of an existing conversation, if any")

class ConversationResponse(BaseModel):
    message: str = Field(..., description="The AI-generated response")
    conversation_id: str = Field(..., description="The ID of the conversation")
    intent: IntentEnum = Field(..., description="The detected intent of the user's message")
    confidence: float = Field(..., ge=0, le=1, description="The confidence score of the intent classification")
    sentiment: SentimentEnum = Field(..., description="The detected sentiment of the user's message")

class MessageInHistory(BaseModel):
    content: str = Field(..., description="The content of the message")
    is_user: bool = Field(..., description="Whether the message is from the user (True) or AI (False)")
    timestamp: datetime = Field(..., description="The timestamp of the message")
    intent: Optional[IntentEnum] = Field(None, description="The intent of the message, if applicable")
    intent_confidence: Optional[float] = Field(None, ge=0, le=1, description="The confidence score of the intent classification")
    sentiment: Optional[SentimentEnum] = Field(None, description="The sentiment of the message")

class ConversationHistory(BaseModel):
    conversation_id: str = Field(..., description="The ID of the conversation")
    messages: List[MessageInHistory] = Field(..., description="The list of messages in the conversation")

class IntentAnalysisRequest(BaseModel):
    message: str = Field(..., description="The message to analyze for intent")

class IntentAnalysisResponse(BaseModel):
    intent: IntentEnum = Field(..., description="The detected intent of the message")
    confidence: float = Field(..., ge=0, le=1, description="The confidence score of the intent classification")

class PerformanceMetrics(BaseModel):
    total_conversations: int = Field(..., ge=0, description="The total number of conversations")
    avg_messages_per_conversation: float = Field(..., ge=0, description="The average number of messages per conversation")
    positive_sentiment_ratio: float = Field(..., ge=0, le=1, description="The ratio of positive sentiments")
    intent_classification_accuracy: float = Field(..., ge=0, le=1, description="The accuracy of intent classification")
    avg_response_time_seconds: float = Field(..., ge=0, description="The average response time in seconds")

class PopularIntent(BaseModel):
    intent: IntentEnum = Field(..., description="The intent")
    count: int = Field(..., ge=0, description="The number of occurrences of this intent")

class SentimentDistribution(BaseModel):
    POSITIVE: float = Field(..., ge=0, le=1, description="The ratio of positive sentiments")
    NEGATIVE: float = Field(..., ge=0, le=1, description="The ratio of negative sentiments")
    NEUTRAL: float = Field(..., ge=0, le=1, description="The ratio of neutral sentiments")

class ConversationSummary(BaseModel):
    conversation_id: str = Field(..., description="The ID of the conversation")
    start_time: datetime = Field(..., description="The start time of the conversation")
    end_time: datetime = Field(..., description="The end time of the conversation")
    duration: float = Field(..., ge=0, description="The duration of the conversation in seconds")
    total_messages: int = Field(..., ge=0, description="The total number of messages in the conversation")
    user_messages: int = Field(..., ge=0, description="The number of messages from the user")
    ai_messages: int = Field(..., ge=0, description="The number of messages from the AI")
    intents: List[IntentEnum] = Field(..., description="The list of intents detected in the conversation")
    sentiments: List[SentimentEnum] = Field(..., description="The list of sentiments detected in the conversation")
    average_intent_confidence: float = Field(..., ge=0, le=1, description="The average confidence score of intent classifications")

class TrainingStatus(BaseModel):
    status: str = Field(..., description="The status of the model training process")
    start_time: Optional[datetime] = Field(None, description="The start time of the training process")
    end_time: Optional[datetime] = Field(None, description="The end time of the training process")
    progress: Optional[float] = Field(None, ge=0, le=1, description="The progress of the training process")
    error: Optional[str] = Field(None, description="Any error message if the training process failed")
