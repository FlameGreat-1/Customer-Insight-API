# app/services/recommendation_engine.py

import asyncio
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.customer import Customer
from app.models.product import Product
from app.models.order import Order, OrderItem
from app.schemas.recommendation import RecommendationResult, RecommendationModel, RecommendationFeedback
from app.core.logging import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RecommendationService:
    def __init__(self, db: Session):
        self.db = db
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.product_features = None
        self.product_ids = None

    async def initialize(self):
        try:
            products = await self.db.query(Product).all()
            product_descriptions = [f"{p.name} {p.description} {p.category} {p.subcategory} {p.brand}" for p in products]
            self.product_features = self.tfidf.fit_transform(product_descriptions)
            self.product_ids = [p.id for p in products]
        except Exception as e:
            logger.error(f"Error initializing recommendation engine: {str(e)}")
            raise

    async def get_recommendations(self, customer_id: int, num_recommendations: int = 5) -> RecommendationResult:
        try:
            if self.product_features is None:
                await self.initialize()

            customer = await self.db.query(Customer).filter(Customer.id == customer_id).first()
            if not customer:
                raise ValueError(f"Customer with id {customer_id} not found")

            user_vector = await self._get_user_vector(customer_id)
            similarities = cosine_similarity(user_vector, self.product_features).flatten()
            top_indices = similarities.argsort()[-num_recommendations:][::-1]

            recommended_products = await asyncio.gather(*[self._get_product_details(self.product_ids[i]) for i in top_indices])

            explanation = f"Based on your purchase history and preferences, we think you'll like these products."

            return RecommendationResult(
                customer_id=customer_id,
                recommendations=recommended_products,
                explanation=explanation
            )
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise

    async def _get_user_vector(self, customer_id: int) -> np.ndarray:
        try:
            # Get user's purchase history
            purchases = await self.db.query(OrderItem).join(Order).filter(Order.customer_id == customer_id).all()
            purchased_product_ids = [item.product_id for item in purchases]

            # Create user vector based on purchased products
            user_vector = np.zeros(self.product_features.shape[1])
            for product_id in purchased_product_ids:
                if product_id in self.product_ids:
                    index = self.product_ids.index(product_id)
                    user_vector += self.product_features[index].toarray().flatten()

            # Normalize user vector
            if np.linalg.norm(user_vector) > 0:
                user_vector = user_vector / np.linalg.norm(user_vector)

            return user_vector.reshape(1, -1)
        except Exception as e:
            logger.error(f"Error creating user vector: {str(e)}")
            raise

    async def _get_product_details(self, product_id: int) -> Dict[str, Any]:
        try:
            product = await self.db.query(Product).filter(Product.id == product_id).first()
            return {
                "id": product.id,
                "name": product.name,
                "price": product.price,
                "category": product.category,
                "brand": product.brand
            }
        except Exception as e:
            logger.error(f"Error getting product details: {str(e)}")
            raise

    async def store_batch_results(self, results: List[RecommendationResult], user_id: int):
        try:
            for result in results:
                for recommendation in result.recommendations:
                    recommendation_log = RecommendationLog(
                        customer_id=result.customer_id,
                        product_id=recommendation['id'],
                        timestamp=datetime.utcnow(),
                        recommendation_type="batch"
                    )
                    self.db.add(recommendation_log)
            await self.db.commit()
        except Exception as e:
            logger.error(f"Error storing batch results: {str(e)}")
            await self.db.rollback()
            raise

    async def process_feedback(self, feedback: RecommendationFeedback):
        try:
            feedback_log = RecommendationFeedback(
                customer_id=feedback.customer_id,
                product_id=feedback.product_id,
                rating=feedback.rating,
                timestamp=datetime.utcnow()
            )
            self.db.add(feedback_log)
            await self.db.commit()

            # Update user preferences based on feedback
            await self._update_user_preferences(feedback.customer_id, feedback.product_id, feedback.rating)
        except Exception as e:
            logger.error(f"Error processing recommendation feedback: {str(e)}")
            await self.db.rollback()
            raise

    async def _update_user_preferences(self, customer_id: int, product_id: int, rating: float):
        try:
            customer = await self.db.query(Customer).filter(Customer.id == customer_id).first()
            product = await self.db.query(Product).filter(Product.id == product_id).first()

            if not customer.preferences:
                customer.preferences = {}

            if 'liked_categories' not in customer.preferences:
                customer.preferences['liked_categories'] = {}

            if rating > 3:  # Consider ratings above 3 as positive
                if product.category in customer.preferences['liked_categories']:
                    customer.preferences['liked_categories'][product.category] += 1
                else:
                    customer.preferences['liked_categories'][product.category] = 1

            await self.db.commit()
        except Exception as e:
            logger.error(f"Error updating user preferences: {str(e)}")
            await self.db.rollback()
            raise

    async def get_performance_metrics(self) -> Dict[str, float]:
        try:
            # Calculate click-through rate (CTR)
            total_recommendations = await self.db.query(func.count(RecommendationLog.id)).scalar()
            clicked_recommendations = await self.db.query(func.count(RecommendationFeedback.id)).scalar()
            ctr = clicked_recommendations / total_recommendations if total_recommendations > 0 else 0

            # Calculate average rating of recommended products
            avg_rating = await self.db.query(func.avg(RecommendationFeedback.rating)).scalar() or 0

            # Calculate conversion rate
            conversions = await self.db.query(func.count(Order.id)).join(RecommendationLog, Order.product_id == RecommendationLog.product_id).scalar()
            conversion_rate = conversions / total_recommendations if total_recommendations > 0 else 0

            return {
                "click_through_rate": ctr,
                "average_rating": float(avg_rating),
                "conversion_rate": conversion_rate
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise

    async def retrain_model(self):
        try:
            logger.info("Starting model retraining...")
            
            # Fetch latest product data
            products = await self.db.query(Product).all()
            product_descriptions = [f"{p.name} {p.description} {p.category} {p.subcategory} {p.brand}" for p in products]
            
            # Retrain TF-IDF vectorizer
            self.tfidf = TfidfVectorizer(stop_words='english')
            self.product_features = self.tfidf.fit_transform(product_descriptions)
            self.product_ids = [p.id for p in products]

            logger.info("Model retraining completed")
        except Exception as e:
            logger.error(f"Error in model retraining: {str(e)}")
            raise

    async def get_available_models(self) -> List[RecommendationModel]:
        return [
            RecommendationModel(
                id="tfidf_cosine",
                name="TF-IDF with Cosine Similarity",
                description="Recommendation model using TF-IDF vectorization and cosine similarity"
            )
        ]

