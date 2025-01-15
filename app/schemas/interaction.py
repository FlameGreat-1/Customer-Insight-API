from pydantic import BaseModel, validator
from typing import Optional, Dict
from datetime import datetime
from app.models.interaction import InteractionType

class InteractionBase(BaseModel):
    customer_id: int
    interaction_type: InteractionType
    timestamp: datetime
    duration: Optional[float]
    channel: str
    source: Optional[str]
    page_url: Optional[str]
    referrer: Optional[str]
    device_type: Optional[str]
    browser: Optional[str]
    os: Optional[str]
    ip_address: Optional[str]
    location: Optional[str]
    session_id: str
    user_agent: Optional[str]
    metadata: Optional[Dict]

class InteractionCreate(InteractionBase):
    pass

class InteractionUpdate(InteractionBase):
    customer_id: Optional[int]
    interaction_type: Optional[InteractionType]
    timestamp: Optional[datetime]

class InteractionInDB(InteractionBase):
    id: int

    class Config:
        orm_mode = True

class InteractionOut(InteractionInDB):
    pass

class InteractionWithCustomer(InteractionOut):
    customer: 'CustomerOut'

    class Config:
        orm_mode = True

# Circular import resolution
from .customer import CustomerOut

InteractionWithCustomer.update_forward_refs()
