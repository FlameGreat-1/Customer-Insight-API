from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from app.models.interaction import Interaction
from app.schemas.interaction import InteractionCreate, InteractionUpdate
from app.core.logging import logger
from typing import List, Optional
from datetime import datetime

def create_interaction(db: Session, interaction: InteractionCreate) -> Interaction:
    try:
        db_interaction = Interaction(**interaction.dict())
        db.add(db_interaction)
        db.commit()
        db.refresh(db_interaction)
        logger.info(f"Created interaction with ID: {db_interaction.id}")
        return db_interaction
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error creating interaction: {str(e)}")
        raise

def get_interaction(db: Session, interaction_id: int) -> Optional[Interaction]:
    try:
        interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
        if interaction:
            logger.info(f"Retrieved interaction with ID: {interaction_id}")
        else:
            logger.info(f"Interaction with ID {interaction_id} not found")
        return interaction
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving interaction: {str(e)}")
        raise

def get_interactions(db: Session, skip: int = 0, limit: int = 100) -> List[Interaction]:
    try:
        interactions = db.query(Interaction).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(interactions)} interactions")
        return interactions
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving interactions: {str(e)}")
        raise

def update_interaction(db: Session, interaction_id: int, interaction: InteractionUpdate) -> Optional[Interaction]:
    try:
        db_interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
        if db_interaction:
            update_data = interaction.dict(exclude_unset=True)
            for key, value in update_data.items():
                setattr(db_interaction, key, value)
            db_interaction.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(db_interaction)
            logger.info(f"Updated interaction with ID: {interaction_id}")
            return db_interaction
        else:
            logger.info(f"Interaction with ID {interaction_id} not found for update")
            return None
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error updating interaction: {str(e)}")
        raise

def delete_interaction(db: Session, interaction_id: int) -> bool:
    try:
        db_interaction = db.query(Interaction).filter(Interaction.id == interaction_id).first()
        if db_interaction:
            db.delete(db_interaction)
            db.commit()
            logger.info(f"Deleted interaction with ID: {interaction_id}")
            return True
        else:
            logger.info(f"Interaction with ID {interaction_id} not found for deletion")
            return False
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error deleting interaction: {str(e)}")
        raise

def get_customer_interactions(db: Session, customer_id: int, skip: int = 0, limit: int = 100) -> List[Interaction]:
    try:
        interactions = db.query(Interaction).filter(Interaction.customer_id == customer_id).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(interactions)} interactions for customer ID: {customer_id}")
        return interactions
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving customer interactions: {str(e)}")
        raise

def get_interactions_by_type(db: Session, interaction_type: str, skip: int = 0, limit: int = 100) -> List[Interaction]:
    try:
        interactions = db.query(Interaction).filter(Interaction.type == interaction_type).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(interactions)} interactions of type: {interaction_type}")
        return interactions
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving interactions by type: {str(e)}")
        raise

def get_recent_interactions(db: Session, days: int, skip: int = 0, limit: int = 100) -> List[Interaction]:
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        interactions = db.query(Interaction).filter(Interaction.created_at >= cutoff_date).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(interactions)} recent interactions from the last {days} days")
        return interactions
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving recent interactions: {str(e)}")
        raise
