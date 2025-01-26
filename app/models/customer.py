from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, ForeignKey, Enum, Index
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from app.db.base_class import Base
import enum
from datetime import datetime
import re

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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
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

    # Indexes
    __table_args__ = (
        Index('idx_customer_name_email', 'first_name', 'last_name', 'email'),
    )

    def __repr__(self):
        return f"<Customer {self.full_name}>"

    @hybrid_property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    @validates('email')
    def validate_email(self, key, email):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            raise ValueError('Invalid email address')
        return email

    @validates('phone_number')
    def validate_phone_number(self, key, phone_number):
        if not re.match(r"^\+?1?\d{9,15}$", phone_number):
            raise ValueError('Invalid phone number')
        return phone_number

    def update_total_spend(self, amount: float) -> None:
        self.total_spend += amount

    def add_loyalty_points(self, points: int) -> None:
        self.loyalty_points += points

    def update_segment(self, new_segment: str) -> None:
        self.segment = new_segment

    def update_status(self, new_status: CustomerStatus) -> None:
        self.status = new_status

    def update_preferences(self, new_preferences: dict) -> None:
        if not self.preferences:
            self.preferences = {}
        self.preferences.update(new_preferences)

from datetime import datetime
import numpy as np

class Customer(Base):
    # ... (previous code remains the same)

    def calculate_lifetime_value(self, prediction_period: int = 12) -> float:
        """
        Calculate the customer lifetime value using a more sophisticated approach.
        
        This method uses the following factors:
        1. Recency (R): Time since last purchase
        2. Frequency (F): Number of purchases
        3. Monetary (M): Total amount spent
        4. Average order value
        5. Customer lifespan
        6. Churn probability
        
        Args:
            prediction_period (int): Number of months to predict into the future (default is 12)
        
        Returns:
            float: Predicted customer lifetime value
        """
        if not self.orders:
            return 0.0

        now = datetime.utcnow()
        
        # Calculate basic RFM metrics
        last_order_date = max(order.order_date for order in self.orders)
        recency = (now - last_order_date).days
        frequency = len(self.orders)
        monetary = self.total_spend
        
        # Calculate average order value
        avg_order_value = monetary / frequency if frequency > 0 else 0
        
        # Calculate customer lifespan in days
        customer_lifespan = (now - self.created_at).days
        
        # Calculate purchase frequency (purchases per day)
        purchase_frequency = frequency / customer_lifespan if customer_lifespan > 0 else 0
        
        # Estimate churn probability (simple exponential decay model)
        churn_rate = 1 - np.exp(-recency / 365)  # Assuming higher churn rate for longer recency
        
        # Predict future purchases
        predicted_purchases = purchase_frequency * prediction_period * 30  # Convert months to days
        
        # Calculate predicted future value
        future_value = predicted_purchases * avg_order_value * (1 - churn_rate)
        
        # Calculate total customer lifetime value
        clv = monetary + future_value
        
        # Apply a discount factor for future value (e.g., 10% annual discount rate)
        discount_rate = 0.1
        discount_factor = 1 / (1 + discount_rate) ** (prediction_period / 12)
        discounted_clv = monetary + (future_value * discount_factor)
        
        # Adjust CLV based on customer segment and loyalty points
        segment_multiplier = 1.0
        if self.segment == "VIP":
            segment_multiplier = 1.2
        elif self.segment == "Regular":
            segment_multiplier = 1.1
        
        loyalty_multiplier = 1 + (self.loyalty_points / 10000)  
        
        final_clv = discounted_clv * segment_multiplier * loyalty_multiplier
        
        return round(final_clv, 2)

    def get_clv_components(self) -> dict:
        """
        Get the components used in CLV calculation for transparency.
        """
        now = datetime.utcnow()
        last_order_date = max(order.order_date for order in self.orders) if self.orders else self.created_at
        recency = (now - last_order_date).days
        frequency = len(self.orders)
        monetary = self.total_spend
        customer_lifespan = (now - self.created_at).days
        
        return {
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "avg_order_value": monetary / frequency if frequency > 0 else 0,
            "customer_lifespan": customer_lifespan,
            "purchase_frequency": frequency / customer_lifespan if customer_lifespan > 0 else 0,
            "segment": self.segment,
            "loyalty_points": self.loyalty_points
        }


    def get_recent_orders(self, limit: int = 5):
        return sorted(self.orders, key=lambda x: x.order_date, reverse=True)[:limit]

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
