from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from app.models.feedback import Feedback
from app.schemas.feedback import FeedbackCreate, FeedbackUpdate
from app.core.logging import logger
from typing import List, Optional
from datetime import datetime

def create_feedback(db: Session, feedback: FeedbackCreate) -> Feedback:
    try:
        db_feedback = Feedback(**feedback.dict())
        db.add(db_feedback)
        db.commit()
        db.refresh(db_feedback)
        logger.info(f"Created feedback with ID: {db_feedback.id}")
        return db_feedback
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error creating feedback: {str(e)}")
        raise

def get_feedback(db: Session, feedback_id: int) -> Optional[Feedback]:
    try:
        feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()
        if feedback:
            logger.info(f"Retrieved feedback with ID: {feedback_id}")
        else:
            logger.info(f"Feedback with ID {feedback_id} not found")
        return feedback
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving feedback: {str(e)}")
        raise

def get_feedbacks(db: Session, skip: int = 0, limit: int = 100) -> List[Feedback]:
    try:
        feedbacks = db.query(Feedback).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(feedbacks)} feedbacks")
        return feedbacks
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving feedbacks: {str(e)}")
        raise

def update_feedback(db: Session, feedback_id: int, feedback: FeedbackUpdate) -> Optional[Feedback]:
    try:
        db_feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()
        if db_feedback:
            update_data = feedback.dict(exclude_unset=True)
            for key, value in update_data.items():
                setattr(db_feedback, key, value)
            db_feedback.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(db_feedback)
            logger.info(f"Updated feedback with ID: {feedback_id}")
            return db_feedback
        else:
            logger.info(f"Feedback with ID {feedback_id} not found for update")
            return None
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error updating feedback: {str(e)}")
        raise

def delete_feedback(db: Session, feedback_id: int) -> bool:
    try:
        db_feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()
        if db_feedback:
            db.delete(db_feedback)
            db.commit()
            logger.info(f"Deleted feedback with ID: {feedback_id}")
            return True
        else:
            logger.info(f"Feedback with ID {feedback_id} not found for deletion")
            return False
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error deleting feedback: {str(e)}")
        raise

def get_product_feedbacks(db: Session, product_id: int, skip: int = 0, limit: int = 100) -> List[Feedback]:
    try:
        feedbacks = db.query(Feedback).filter(Feedback.product_id == product_id).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(feedbacks)} feedbacks for product ID: {product_id}")
        return feedbacks
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving product feedbacks: {str(e)}")
        raise

def get_recent_feedbacks(db: Session, days: int, skip: int = 0, limit: int = 100) -> List[Feedback]:
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        feedbacks = db.query(Feedback).filter(Feedback.created_at >= cutoff_date).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(feedbacks)} recent feedbacks from the last {days} days")
        return feedbacks
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving recent feedbacks: {str(e)}")
        raise

def get_feedbacks_by_rating(db: Session, rating: int, skip: int = 0, limit: int = 100) -> List[Feedback]:
    try:
        feedbacks = db.query(Feedback).filter(Feedback.rating == rating).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(feedbacks)} feedbacks with rating: {rating}")
        return feedbacks
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving feedbacks by rating: {str(e)}")
        raise

def get_average_product_rating(db: Session, product_id: int) -> float:
    try:
        result = db.query(func.avg(Feedback.rating)).filter(Feedback.product_id == product_id).scalar()
        average_rating = float(result) if result else 0.0
        logger.info(f"Retrieved average rating {average_rating} for product ID: {product_id}")
        return average_rating
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving average product rating: {str(e)}")
        raise

def get_feedback_summary(db: Session, product_id: int) -> Dict[str, Any]:
    try:
        total_count = db.query(func.count(Feedback.id)).filter(Feedback.product_id == product_id).scalar()
        rating_counts = db.query(
            Feedback.rating, func.count(Feedback.id)
        ).filter(Feedback.product_id == product_id).group_by(Feedback.rating).all()
        
        summary = {
            "total_feedbacks": total_count,
            "average_rating": get_average_product_rating(db, product_id),
            "rating_distribution": {rating: count for rating, count in rating_counts}
        }
        logger.info(f"Generated feedback summary for product ID: {product_id}")
        return summary
    except SQLAlchemyError as e:
        logger.error(f"Error generating feedback summary: {str(e)}")
        raise

def update_feedback_sentiment(db: Session, feedback_id: int, sentiment: str) -> Optional[Feedback]:
    try:
        db_feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()
        if db_feedback:
            db_feedback.sentiment = sentiment
            db_feedback.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(db_feedback)
            logger.info(f"Updated sentiment for feedback with ID: {feedback_id}")
            return db_feedback
        else:
            logger.info(f"Feedback with ID {feedback_id} not found for sentiment update")
            return None
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error updating feedback sentiment: {str(e)}")
        raise

def get_feedbacks_requiring_response(db: Session, skip: int = 0, limit: int = 100) -> List[Feedback]:
    try:
        feedbacks = db.query(Feedback).filter(
            (Feedback.rating <= 3) & (Feedback.response.is_(None))
        ).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(feedbacks)} feedbacks requiring response")
        return feedbacks
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving feedbacks requiring response: {str(e)}")
        raise

def add_feedback_response(db: Session, feedback_id: int, response: str) -> Optional[Feedback]:
    try:
        db_feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()
        if db_feedback:
            db_feedback.response = response
            db_feedback.response_date = datetime.utcnow()
            db_feedback.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(db_feedback)
            logger.info(f"Added response to feedback with ID: {feedback_id}")
            return db_feedback
        else:
            logger.info(f"Feedback with ID {feedback_id} not found for adding response")
            return None
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error adding feedback response: {str(e)}")
        raise

def get_feedback_trends(db: Session, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    try:
        trends = db.query(
            func.date(Feedback.created_at).label('date'),
            func.avg(Feedback.rating).label('average_rating'),
            func.count(Feedback.id).label('feedback_count')
        ).filter(
            Feedback.created_at.between(start_date, end_date)
        ).group_by(func.date(Feedback.created_at)).all()

        result = {
            "date_range": {
                "start": start_date.strftime('%Y-%m-%d'),
                "end": end_date.strftime('%Y-%m-%d')
            },
            "trends": [
                {
                    "date": trend.date.strftime('%Y-%m-%d'),
                    "average_rating": float(trend.average_rating),
                    "feedback_count": trend.feedback_count
                } for trend in trends
            ]
        }
        logger.info(f"Generated feedback trends from {start_date} to {end_date}")
        return result
    except SQLAlchemyError as e:
        logger.error(f"Error generating feedback trends: {str(e)}")
        raise

