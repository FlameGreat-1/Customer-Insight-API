import asyncio
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.feedback import Feedback, SentimentType
from app.models.order import Order
from app.schemas.sentiment import SentimentAnalysisResult
from app.core.logging import logger
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch
from datetime import datetime
import joblib
import os
from datetime import datetime, timedelta


class SentimentAnalysisService:
    def __init__(self, db: Session):
        self.db = db
        self.model_path = "app/models/sentiment_model.pth"
        self.tokenizer_path = "app/models/sentiment_tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.tokenizer_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            logger.info("Loaded existing sentiment analysis model")
        else:
            logger.info("Using default sentiment analysis model")

    async def analyze(self, text: str) -> SentimentAnalysisResult:
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment_score = probabilities[0][1].item()  # Probability of positive sentiment
            
            if sentiment_score > 0.6:
                sentiment_type = SentimentType.POSITIVE
            elif sentiment_score < 0.4:
                sentiment_type = SentimentType.NEGATIVE
            else:
                sentiment_type = SentimentType.NEUTRAL

            return SentimentAnalysisResult(
                sentiment=sentiment_type,
                confidence=max(sentiment_score, 1 - sentiment_score),
                text=text
            )
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            raise

    async def analyze_batch(self, texts: List[str]) -> List[SentimentAnalysisResult]:
        try:
            inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment_scores = probabilities[:, 1].tolist()  # Probabilities of positive sentiment

            results = []
            for score, text in zip(sentiment_scores, texts):
                if score > 0.6:
                    sentiment_type = SentimentType.POSITIVE
                elif score < 0.4:
                    sentiment_type = SentimentType.NEGATIVE
                else:
                    sentiment_type = SentimentType.NEUTRAL

                results.append(SentimentAnalysisResult(
                    sentiment=sentiment_type,
                    confidence=max(score, 1 - score),
                    text=text
                ))

            return results
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {str(e)}")
            raise

    async def store_batch_results(self, results: List[SentimentAnalysisResult], user_id: int):
        try:
            async with self.db.begin():
                for result in results:
                    feedback = Feedback(
                        customer_id=user_id,
                        content=result.text,
                        sentiment=result.sentiment,
                        metadata={"confidence": result.confidence},
                        timestamp=datetime.utcnow()
                    )
                    self.db.add(feedback)
            logger.info(f"Stored {len(results)} sentiment analysis results for user {user_id}")
        except Exception as e:
            logger.error(f"Error storing batch results: {str(e)}")
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

    async def get_sentiment_distribution(self, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        try:
            query = self.db.query(Feedback.sentiment, func.count(Feedback.id).label('count')).\
                filter(Feedback.timestamp.between(start_date, end_date)).\
                group_by(Feedback.sentiment)
            results = await self.db.execute(query)
            
            distribution = {sentiment.value: count for sentiment, count in results}
            total = sum(distribution.values())
            
            return {sentiment: count / total for sentiment, count in distribution.items()}
        except Exception as e:
            logger.error(f"Error retrieving sentiment distribution: {str(e)}")
            raise

    async def retrain_model(self, training_data: List[Dict[str, str]]):
        try:
            logger.info("Starting model retraining...")
            
            # Prepare the dataset
            texts = [item['text'] for item in training_data]
            labels = [1 if item['sentiment'] == 'POSITIVE' else 0 for item in training_data]
            
            # Tokenize the texts
            encodings = self.tokenizer(texts, truncation=True, padding=True)
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(encodings['input_ids']),
                torch.tensor(encodings['attention_mask']),
                torch.tensor(labels)
            )
            
            # Set up training parameters
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
            
            # Training loop
            self.model.train()
            for epoch in range(3):  # Number of epochs can be adjusted
                for batch in train_loader:
                    optimizer.zero_grad()
                    input_ids, attention_mask, labels = batch
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
            
            # Save the retrained model
            torch.save(self.model.state_dict(), self.model_path)
            self.tokenizer.save_pretrained(self.tokenizer_path)
            
            logger.info("Model retraining completed and saved")
        except Exception as e:
            logger.error(f"Error in model retraining: {str(e)}")
            raise

    async def get_performance_metrics(self) -> Dict[str, float]:
        try:
            # Fetch a sample of recent feedback for evaluation
            recent_feedbacks = await self.db.query(Feedback).order_by(Feedback.timestamp.desc()).limit(1000).all()
            
            texts = [feedback.content for feedback in recent_feedbacks]
            true_labels = [1 if feedback.sentiment == SentimentType.POSITIVE else 0 for feedback in recent_feedbacks]
            
            # Predict sentiments
            predictions = await self.analyze_batch(texts)
            pred_labels = [1 if pred.sentiment == SentimentType.POSITIVE else 0 for pred in predictions]
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, pred_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise

    async def get_sentiment_trends(self, start_date: datetime, end_date: datetime, interval: str = 'day') -> Dict[str, List[Dict[str, Any]]]:
        try:
            query = self.db.query(
                func.date_trunc(interval, Feedback.timestamp).label('time_bucket'),
                Feedback.sentiment,
                func.count(Feedback.id).label('count')
            ).filter(
                Feedback.timestamp.between(start_date, end_date)
            ).group_by(
                'time_bucket', Feedback.sentiment
            ).order_by('time_bucket')

            results = await self.db.execute(query)

            trends = {sentiment.value: [] for sentiment in SentimentType}
            for row in results:
                trends[row.sentiment.value].append({
                    'timestamp': row.time_bucket,
                    'count': row.count
                })

            return trends
        except Exception as e:
            logger.error(f"Error retrieving sentiment trends: {str(e)}")
            raise

    async def analyze_feedback_impact(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        try:
            query = self.db.query(
                Feedback.sentiment,
                func.avg(Order.total_amount).label('avg_order_value'),
                func.count(Order.id).label('order_count')
            ).join(
                Order, Feedback.customer_id == Order.customer_id
            ).filter(
                Feedback.timestamp.between(start_date, end_date),
                Order.order_date > Feedback.timestamp,
                Order.order_date <= Feedback.timestamp + timedelta(days=30)
            ).group_by(Feedback.sentiment)

            results = await self.db.execute(query)

            impact_analysis = {}
            for row in results:
                impact_analysis[row.sentiment.value] = {
                    'average_order_value': float(row.avg_order_value),
                    'order_count': row.order_count
                }

            return impact_analysis
        except Exception as e:
            logger.error(f"Error analyzing feedback impact: {str(e)}")
            raise
