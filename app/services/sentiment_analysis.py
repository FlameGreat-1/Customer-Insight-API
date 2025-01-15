import asyncio
from typing import List, Dict
from sqlalchemy.orm import Session
from app.models.feedback import Feedback, SentimentType
from app.schemas.sentiment import SentimentAnalysisResult
from app.core.logging import logger
from transformers import pipeline
import numpy as np

class SentimentAnalysisService:
    def __init__(self, db: Session):
        self.db = db
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    async def analyze(self, text: str) -> SentimentAnalysisResult:
        try:
            result = await asyncio.to_thread(self.sentiment_analyzer, text)
            sentiment = result[0]['label']
            confidence = result[0]['score']

            sentiment_type = SentimentType.POSITIVE if sentiment == "POSITIVE" else SentimentType.NEGATIVE
            if 0.4 <= confidence <= 0.6:
                sentiment_type = SentimentType.NEUTRAL

            return SentimentAnalysisResult(
                sentiment=sentiment_type,
                confidence=confidence,
                text=text
            )
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            raise

    async def analyze_batch(self, texts: List[str]) -> List[SentimentAnalysisResult]:
        try:
            results = await asyncio.to_thread(self.sentiment_analyzer, texts)
            return [
                SentimentAnalysisResult(
                    sentiment=SentimentType.POSITIVE if r['label'] == "POSITIVE" else SentimentType.NEGATIVE,
                    confidence=r['score'],
                    text=text
                ) for r, text in zip(results, texts)
            ]
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {str(e)}")
            raise

    async def store_batch_results(self, results: List[SentimentAnalysisResult], user_id: int):
        try:
            for result in results:
                feedback = Feedback(
                    customer_id=user_id,
                    content=result.text,
                    sentiment=result.sentiment,
                    metadata={"confidence": result.confidence}
                )
                self.db.add(feedback)
            await self.db.commit()
        except Exception as e:
            logger.error(f"Error storing batch results: {str(e)}")
            await self.db.rollback()
            raise

    async def get_user_history(self, user_id: int, limit: int = 10) -> List[SentimentAnalysisResult]:
        try:
            feedbacks = await self.db.query(Feedback).filter(Feedback.customer_id == user_id).order_by(Feedback.timestamp.desc()).limit(limit).all()
            return [
                SentimentAnalysisResult(
                    sentiment=feedback.sentiment,
                    confidence=feedback.metadata.get("confidence", 0.0),
                    text=feedback.content
                ) for feedback in feedbacks
            ]
        except Exception as e:
            logger.error(f"Error retrieving user history: {str(e)}")
            raise

    async def get_sentiment_distribution(self, start_date: str, end_date: str) -> Dict[str, float]:
        try:
            query = self.db.query(Feedback.sentiment, func.count(Feedback.id)).\
                filter(Feedback.timestamp.between(start_date, end_date)).\
                group_by(Feedback.sentiment)
            results = await query.all()
            total = sum(count for _, count in results)
            return {sentiment.value: count / total for sentiment, count in results}
        except Exception as e:
            logger.error(f"Error retrieving sentiment distribution: {str(e)}")
            raise

    async def retrain_model(self, training_data: List[Dict[str, str]]):
        try:
            # This is a placeholder for model retraining logic
            # In a real-world scenario, you would implement transfer learning or fine-tuning here
            logger.info("Starting model retraining...")
            await asyncio.sleep(10)  # Simulating long-running task
            logger.info("Model retraining completed")
        except Exception as e:
            logger.error(f"Error in model retraining: {str(e)}")
            raise

    async def get_performance_metrics(self) -> Dict[str, float]:
        try:
            # This is a placeholder for performance metrics calculation
            # In a real-world scenario, you would calculate these metrics based on a test dataset
            return {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.94,
                "f1_score": 0.91
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise
