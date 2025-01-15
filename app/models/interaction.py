from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from app.db.base_class import Base
import enum

class InteractionType(enum.Enum):
    WEBSITE_VISIT = "website_visit"
    CUSTOMER_SUPPORT = "customer_support"
    PURCHASE = "purchase"
    EMAIL = "email"
    SOCIAL_MEDIA = "social_media"
    APP_USAGE = "app_usage"

class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), index=True)
    interaction_type = Column(Enum(InteractionType))
    timestamp = Column(DateTime)
    duration = Column(Float)  # in seconds
    channel = Column(String)
    source = Column(String)
    page_url = Column(String)
    referrer = Column(String)
    device_type = Column(String)
    browser = Column(String)
    os = Column(String)
    ip_address = Column(String)
    location = Column(String)
    session_id = Column(String, index=True)
    user_agent = Column(String)
    metadata = Column(JSONB)

    # Relationships
    customer = relationship("Customer", back_populates="interactions")

    def __repr__(self):
        return f"<Interaction {self.id} - {self.interaction_type.value}>"

    @property
    def interaction_summary(self):
        return f"{self.interaction_type.value} on {self.timestamp}"

    def update_metadata(self, new_metadata):
        self.metadata.update(new_metadata)

    def set_duration(self, end_time):
        self.duration = (end_time - self.timestamp).total_seconds()
