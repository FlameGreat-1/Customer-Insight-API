from pydantic import BaseModel, validator
from typing import Optional, Dict
from datetime import datetime
from app.models.feedback import FeedbackType, SentimentType

class FeedbackBase(BaseModel):
    customer_id: int
    product_id: Optional[int]
    feedback_type: FeedbackType
    content: str
    rating: Optional[float]
    sentiment: Optional[SentimentType]
    source: str
    is_public: bool = False
    metadata: Optional[Dict]

    @validator('rating')
    def rating_must_be_between_0_and_5(cls, v):
        if v is not None and (v < 0 or v > 5):
            raise ValueError('Rating must be between 0 and 5')
        return v

class FeedbackCreate(FeedbackBase):
    pass

class FeedbackUpdate(FeedbackBase):
    customer_id: Optional[int]
    feedback_type: Optional[FeedbackType]
    content: Optional[str]
    sentiment: Optional[SentimentType]
    is_public: Optional[bool]
    is_resolved: Optional[bool]

class FeedbackInDB(FeedbackBase):
    id: int
    timestamp: datetime
    is_resolved: bool
    resolved_at: Optional[datetime]
    resolved_by: Optional[int]

    class Config:
        orm_mode = True

class FeedbackOut(FeedbackInDB):
    pass

class FeedbackWithRelations(FeedbackOut):
    customer: 'CustomerOut'
    product: Optional['ProductOut']
    resolver: Optional['UserOut']

    class Config:
        orm_mode = True

# Circular import resolution
from .customer import CustomerOut
from .product import ProductOut
from .user import UserOut

FeedbackWithRelations.update_forward_refs()


