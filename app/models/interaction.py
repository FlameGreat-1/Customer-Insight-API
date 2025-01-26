from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float, Index, Boolean
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import JSONB
from app.db.base_class import Base
import enum
from datetime import datetime
from typing import Optional, Dict, Any
import ipaddress
import user_agents
from geoip2.database import Reader
from app.core.config import settings

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
    timestamp = Column(DateTime, default=datetime.utcnow)
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
    is_converted = Column(Boolean, default=False)
    conversion_value = Column(Float, default=0.0)

    # Relationships
    customer = relationship("Customer", back_populates="interactions")

    # Indexes
    __table_args__ = (
        Index('idx_interaction_customer_timestamp', 'customer_id', 'timestamp'),
        Index('idx_interaction_type_timestamp', 'interaction_type', 'timestamp'),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timestamp = datetime.utcnow()
        self.parse_user_agent()
        self.get_location_from_ip()

    def __repr__(self):
        return f"<Interaction {self.id} - {self.interaction_type.value}>"

    @property
    def interaction_summary(self) -> str:
        return f"{self.interaction_type.value} on {self.timestamp}"

    @validates('ip_address')
    def validate_ip_address(self, key, ip_address):
        try:
            ipaddress.ip_address(ip_address)
            return ip_address
        except ValueError:
            raise ValueError("Invalid IP address")

    def update_metadata(self, new_metadata: Dict[str, Any]) -> None:
        if not self.metadata:
            self.metadata = {}
        self.metadata.update(new_metadata)

    def set_duration(self, end_time: datetime) -> None:
        self.duration = (end_time - self.timestamp).total_seconds()

    def parse_user_agent(self) -> None:
        if self.user_agent:
            user_agent = user_agents.parse(self.user_agent)
            self.browser = f"{user_agent.browser.family} {user_agent.browser.version_string}"
            self.os = f"{user_agent.os.family} {user_agent.os.version_string}"
            self.device_type = user_agent.device.model or "Unknown"

    def get_location_from_ip(self) -> None:
        if self.ip_address:
            try:
                with Reader(settings.GEOIP_DATABASE_PATH) as reader:
                    response = reader.city(self.ip_address)
                    self.location = f"{response.city.name}, {response.country.name}"
            except Exception as e:
                print(f"Error getting location from IP: {str(e)}")

    def mark_as_converted(self, value: float) -> None:
        self.is_converted = True
        self.conversion_value = value

    def get_interaction_value(self) -> float:
        if self.is_converted:
            return self.conversion_value
        
        # Assign values to different interaction types
        value_map = {
            InteractionType.WEBSITE_VISIT: 1,
            InteractionType.CUSTOMER_SUPPORT: 5,
            InteractionType.PURCHASE: 10,
            InteractionType.EMAIL: 2,
            InteractionType.SOCIAL_MEDIA: 3,
            InteractionType.APP_USAGE: 4
        }
        return value_map.get(self.interaction_type, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "interaction_type": self.interaction_type.value,
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration,
            "channel": self.channel,
            "source": self.source,
            "page_url": self.page_url,
            "referrer": self.referrer,
            "device_type": self.device_type,
            "browser": self.browser,
            "os": self.os,
            "location": self.location,
            "session_id": self.session_id,
            "is_converted": self.is_converted,
            "conversion_value": self.conversion_value,
            "interaction_value": self.get_interaction_value()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Interaction':
        if 'interaction_type' in data:
            data['interaction_type'] = InteractionType(data['interaction_type'])
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

    def anonymize(self) -> None:
        self.ip_address = 'xxx.xxx.xxx.xxx'
        self.location = 'Anonymous'
        if self.metadata and 'personal_info' in self.metadata:
            del self.metadata['personal_info']

    def is_recent(self, days: int = 30) -> bool:
        return (datetime.utcnow() - self.timestamp).days <= days

    def get_interaction_score(self) -> float:
        base_score = self.get_interaction_value()
        recency_factor = max(1, 30 - (datetime.utcnow() - self.timestamp).days) / 30
        duration_factor = min(self.duration / 3600, 1) if self.duration else 1  # Cap at 1 hour
        return base_score * recency_factor * duration_factor

