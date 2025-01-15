from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from app.db.base_class import Base
import enum

class CustomerStatus(enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    CHURNED = "churned"

class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String, unique=True, index=True)
    first_name = Column(String, index=True)
    last_name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    phone_number = Column(String)
    date_of_birth = Column(DateTime)
    gender = Column(String)
    address = Column(String)
    city = Column(String)
    state = Column(String)
    country = Column(String)
    postal_code = Column(String)
    status = Column(Enum(CustomerStatus), default=CustomerStatus.ACTIVE)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    last_login = Column(DateTime)
    total_spend = Column(Float, default=0.0)
    loyalty_points = Column(Integer, default=0)
    segment = Column(String)
    preferences = Column(JSONB)
    custom_attributes = Column(JSONB)

    # Relationships
    interactions = relationship("Interaction", back_populates="customer")
    feedback = relationship("Feedback", back_populates="customer")
    orders = relationship("Order", back_populates="customer")

    def __repr__(self):
        return f"<Customer {self.first_name} {self.last_name}>"

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    def update_total_spend(self, amount):
        self.total_spend += amount
        self.updated_at = datetime.utcnow()

    def add_loyalty_points(self, points):
        self.loyalty_points += points
        self.updated_at = datetime.utcnow()

    def update_segment(self, new_segment):
        self.segment = new_segment
        self.updated_at = datetime.utcnow()

    def update_status(self, new_status):
        self.status = new_status
        self.updated_at = datetime.utcnow()

    def update_preferences(self, new_preferences):
        self.preferences.update(new_preferences)
        self.updated_at = datetime.utcnow()
