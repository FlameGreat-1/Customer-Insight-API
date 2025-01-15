from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float, Text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from app.db.base_class import Base
import enum

class FeedbackType(enum.Enum):
    PRODUCT_REVIEW = "product_review"
    CUSTOMER_SUPPORT = "customer_support"
    GENERAL = "general"
    SURVEY = "survey"

class SentimentType(enum.Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=True)
    feedback_type = Column(Enum(FeedbackType))
    content = Column(Text)
    rating = Column(Float)
    sentiment = Column(Enum(SentimentType))
    timestamp = Column(DateTime)
    source = Column(String)
    is_public = Column(Boolean, default=False)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    metadata = Column(JSONB)

    # Relationships
    customer = relationship("Customer", back_populates="feedback")
    product = relationship("Product", back_populates="feedback")
    resolver = relationship("User", back_populates="resolved_feedback")

    def __repr__(self):
        return f"<Feedback {self.id} - {self.feedback_type.value}>"

    @property
    def feedback_summary(self):
        return f"{self.feedback_type.value} - {self.sentiment.value}"

    def resolve(self, resolver_id):
        self.is_resolved = True
        self.resolved_at = datetime.utcnow()
        self.resolved_by = resolver_id

    def update_sentiment(self, new_sentiment):
        self.sentiment = new_sentiment

    def update_metadata(self, new_metadata):
        self.metadata.update(new_metadata)

    def make_public(self):
        self.is_public = True

    def make_private(self):
        self.is_public = False
