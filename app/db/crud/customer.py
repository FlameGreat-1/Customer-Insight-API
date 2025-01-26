from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from app.models.customer import Customer
from app.schemas.customer import CustomerCreate, CustomerUpdate
from app.core.logging import logger
from typing import List, Optional
from datetime import datetime

def create_customer(db: Session, customer: CustomerCreate) -> Customer:
    try:
        db_customer = Customer(**customer.dict())
        db.add(db_customer)
        db.commit()
        db.refresh(db_customer)
        logger.info(f"Created customer with ID: {db_customer.id}")
        return db_customer
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error creating customer: {str(e)}")
        raise

def get_customer(db: Session, customer_id: int) -> Optional[Customer]:
    try:
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if customer:
            logger.info(f"Retrieved customer with ID: {customer_id}")
        else:
            logger.info(f"Customer with ID {customer_id} not found")
        return customer
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving customer: {str(e)}")
        raise

def get_customers(db: Session, skip: int = 0, limit: int = 100) -> List[Customer]:
    try:
        customers = db.query(Customer).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(customers)} customers")
        return customers
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving customers: {str(e)}")
        raise

def update_customer(db: Session, customer_id: int, customer: CustomerUpdate) -> Optional[Customer]:
    try:
        db_customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if db_customer:
            update_data = customer.dict(exclude_unset=True)
            for key, value in update_data.items():
                setattr(db_customer, key, value)
            db_customer.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(db_customer)
            logger.info(f"Updated customer with ID: {customer_id}")
            return db_customer
        else:
            logger.info(f"Customer with ID {customer_id} not found for update")
            return None
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error updating customer: {str(e)}")
        raise

def delete_customer(db: Session, customer_id: int) -> bool:
    try:
        db_customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if db_customer:
            db.delete(db_customer)
            db.commit()
            logger.info(f"Deleted customer with ID: {customer_id}")
            return True
        else:
            logger.info(f"Customer with ID {customer_id} not found for deletion")
            return False
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error deleting customer: {str(e)}")
        raise

def get_customers_by_segment(db: Session, segment: str, skip: int = 0, limit: int = 100) -> List[Customer]:
    try:
        customers = db.query(Customer).filter(Customer.segment == segment).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(customers)} customers in segment: {segment}")
        return customers
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving customers by segment: {str(e)}")
        raise

def get_high_value_customers(db: Session, value_threshold: float, skip: int = 0, limit: int = 100) -> List[Customer]:
    try:
        customers = db.query(Customer).filter(Customer.lifetime_value >= value_threshold).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(customers)} high-value customers")
        return customers
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving high-value customers: {str(e)}")
        raise

def update_customer_segment(db: Session, customer_id: int, new_segment: str) -> Optional[Customer]:
    try:
        db_customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if db_customer:
            db_customer.segment = new_segment
            db_customer.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(db_customer)
            logger.info(f"Updated segment for customer with ID: {customer_id}")
            return db_customer
        else:
            logger.info(f"Customer with ID {customer_id} not found for segment update")
            return None
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error updating customer segment: {str(e)}")
        raise
