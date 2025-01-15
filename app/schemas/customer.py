from pydantic import BaseModel, EmailStr, validator, constr
from typing import Optional, Dict, List
from datetime import datetime
from app.models.customer import CustomerStatus

class CustomerBase(BaseModel):
    external_id: Optional[str]
    first_name: constr(min_length=1, max_length=50)
    last_name: constr(min_length=1, max_length=50)
    email: EmailStr
    phone_number: Optional[constr(regex=r'^\+?1?\d{9,15}$')]
    date_of_birth: Optional[datetime]
    gender: Optional[str]
    address: Optional[str]
    city: Optional[str]
    state: Optional[str]
    country: Optional[str]
    postal_code: Optional[str]
    status: CustomerStatus = CustomerStatus.ACTIVE
    segment: Optional[str]
    preferences: Optional[Dict]
    custom_attributes: Optional[Dict]

class CustomerCreate(CustomerBase):
    pass

class CustomerUpdate(CustomerBase):
    first_name: Optional[constr(min_length=1, max_length=50)]
    last_name: Optional[constr(min_length=1, max_length=50)]
    email: Optional[EmailStr]

class CustomerInDB(CustomerBase):
    id: int
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime]
    total_spend: float
    loyalty_points: int

    class Config:
        orm_mode = True

class CustomerOut(CustomerInDB):
    pass

class CustomerWithRelations(CustomerOut):
    interactions: List['InteractionOut']
    feedback: List['FeedbackOut']
    orders: List['OrderOut']

    class Config:
        orm_mode = True

# Circular import resolution
from .interaction import InteractionOut
from .feedback import FeedbackOut
from .order import OrderOut

CustomerWithRelations.update_forward_refs()
