# app/services/recommendation_engine.py

import asyncio
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.customer import Customer
from app.models.product import Product
from app.models.order import Order, OrderItem
from app.models.recommendation_log import RecommendationLog
from app.models.recommendation_feedback import RecommendationFeedback
from app.schemas.recommendation import RecommendationResult, RecommendationModel, RecommendationFeedbackSchema
from app.core.logging import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os

class RecommendationService:
    def __init__(self, db: Session):
        self.db = db
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.svd = TruncatedSVD(n_components=100)
        self.product_features = None
        self.product_ids = None
        self.model_path = "app/models/recommendation_model.joblib"
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.tfidf = model_data['tfidf']
            self.svd = model_data['svd']
            self.product_features = model_data['product_features']
            self.product_ids = model_data['product_ids']
            logger.info("Loaded existing recommendation model")
        else:
            logger.info("No existing model found. Will initialize new model.")

    async def initialize(self):
        try:
            if self.product_features is None:
                products = await self.db.query(Product).all()
                product_descriptions = [f"{p.name} {p.description} {p.category} {p.subcategory} {p.brand}" for p in products]
                tfidf_features = self.tfidf.fit_transform(product_descriptions)
                self.product_features = self.svd.fit_transform(tfidf_features)
                self.product_ids = [p.id for p in products]
                
                model_data = {
                    'tfidf': self.tfidf,
                    'svd': self.svd,
                    'product_features': self.product_features,
                    'product_ids': self.product_ids
                }
                joblib.dump(model_data, self.model_path)
                logger.info("Initialized and saved new recommendation model")
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

            explanation = await self._generate_recommendation_explanation(customer_id, recommended_products)

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
            purchases = await self.db.query(OrderItem).join(Order).filter(Order.customer_id == customer_id).all()
            purchased_product_ids = [item.product_id for item in purchases]

            user_vector = np.zeros(self.product_features.shape[1])
            for product_id in purchased_product_ids:
                if product_id in self.product_ids:
                    index = self.product_ids.index(product_id)
                    user_vector += self.product_features[index]

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
                "price": float(product.price),
                "category": product.category,
                "brand": product.brand,
                "rating": await self._get_product_rating(product_id)
            }
        except Exception as e:
            logger.error(f"Error getting product details: {str(e)}")
            raise

    async def _get_product_rating(self, product_id: int) -> float:
        try:
            avg_rating = await self.db.query(func.avg(RecommendationFeedback.rating)).filter(
                RecommendationFeedback.product_id == product_id
            ).scalar()
            return float(avg_rating) if avg_rating else 0.0
        except Exception as e:
            logger.error(f"Error getting product rating: {str(e)}")
            raise

    async def store_batch_results(self, results: List[RecommendationResult]):
        try:
            async with self.db.begin():
                for result in results:
                    for recommendation in result.recommendations:
                        recommendation_log = RecommendationLog(
                            customer_id=result.customer_id,
                            product_id=recommendation['id'],
                            timestamp=datetime.utcnow(),
                            recommendation_type="batch"
                        )
                        self.db.add(recommendation_log)
            logger.info(f"Stored batch results for {len(results)} customers")
        except Exception as e:
            logger.error(f"Error storing batch results: {str(e)}")
            raise

    async def process_feedback(self, feedback: RecommendationFeedbackSchema):
        try:
            async with self.db.begin():
                feedback_log = RecommendationFeedback(
                    customer_id=feedback.customer_id,
                    product_id=feedback.product_id,
                    rating=feedback.rating,
                    timestamp=datetime.utcnow()
                )
                self.db.add(feedback_log)

            await self._update_user_preferences(feedback.customer_id, feedback.product_id, feedback.rating)
            logger.info(f"Processed feedback for customer {feedback.customer_id} on product {feedback.product_id}")
        except Exception as e:
            logger.error(f"Error processing recommendation feedback: {str(e)}")
            raise

    async def _update_user_preferences(self, customer_id: int, product_id: int, rating: float):
        try:
            async with self.db.begin():
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

            logger.info(f"Updated preferences for customer {customer_id}")
        except Exception as e:
            logger.error(f"Error updating user preferences: {str(e)}")
            raise

    async def get_performance_metrics(self) -> Dict[str, float]:
        try:
            total_recommendations = await self.db.query(func.count(RecommendationLog.id)).scalar()
            clicked_recommendations = await self.db.query(func.count(RecommendationFeedback.id)).scalar()
            ctr = clicked_recommendations / total_recommendations if total_recommendations > 0 else 0

            avg_rating = await self.db.query(func.avg(RecommendationFeedback.rating)).scalar() or 0

            conversions = await self.db.query(func.count(Order.id)).join(
                RecommendationLog, Order.product_id == RecommendationLog.product_id
            ).scalar()
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
            
            products = await self.db.query(Product).all()
            product_descriptions = [f"{p.name} {p.description} {p.category} {p.subcategory} {p.brand}" for p in products]
            
            tfidf_features = self.tfidf.fit_transform(product_descriptions)
            self.product_features = self.svd.fit_transform(tfidf_features)
            self.product_ids = [p.id for p in products]

            model_data = {
                'tfidf': self.tfidf,
                'svd': self.svd,
                'product_features': self.product_features,
                'product_ids': self.product_ids
            }
            joblib.dump(model_data, self.model_path)

            logger.info("Model retraining completed and saved")
        except Exception as e:
            logger.error(f"Error in model retraining: {str(e)}")
            raise

    async def get_available_models(self) -> List[RecommendationModel]:
        return [
            RecommendationModel(
                id="tfidf_svd_cosine",
                name="TF-IDF with SVD and Cosine Similarity",
                description="Recommendation model using TF-IDF vectorization, SVD dimensionality reduction, and cosine similarity"
            )
        ]

    async def _generate_recommendation_explanation(self, customer_id: int, recommended_products: List[Dict[str, Any]]) -> str:
        try:
            customer = await self.db.query(Customer).filter(Customer.id == customer_id).first()
            recent_purchases = await self.db.query(Product).join(OrderItem).join(Order).filter(
                Order.customer_id == customer_id
            ).order_by(Order.order_date.desc()).limit(3).all()

            explanation = f"Based on your recent purchases and preferences, we think you'll like these products:\n\n"
            
            for product in recommended_products:
                reason = ""
                if any(rp.category == product['category'] for rp in recent_purchases):
                    reason = f"You've recently purchased items from the {product['category']} category."
                elif customer.preferences and 'liked_categories' in customer.preferences:
                    if product['category'] in customer.preferences['liked_categories']:
                        reason = f"You've shown interest in the {product['category']} category."
                
                if not reason:
                    reason = "This product is popular among customers with similar preferences."
                
                explanation += f"- {product['name']}: {reason}\n"

            return explanation
        except Exception as e:
            logger.error(f"Error generating recommendation explanation: {str(e)}")
            raise

    async def get_trending_products(self, category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            query = self.db.query(
                Product.id,
                Product.name,
                Product.category,
                Product.brand,
                func.count(OrderItem.id).label('order_count')
            ).join(OrderItem).join(Order)
            
            if category:
                query = query.filter(Product.category == category)
            
            query = query.group_by(Product.id).order_by(func.count(OrderItem.id).desc()).limit(limit)
            
            results = await query.all()
            
            trending_products = [
                {
                    "id": product.id,
                    "name": product.name,
                    "category": product.category,
                    "brand": product.brand,
                    "order_count": product.order_count
                }
                for product in results
            ]
            
            return trending_products
        except Exception as e:
            logger.error(f"Error getting trending products: {str(e)}")
            raise

    async def get_personalized_deals(self, customer_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            customer = await self.db.query(Customer).filter(Customer.id == customer_id).first()
            if not customer:
                raise ValueError(f"Customer with id {customer_id} not found")

            recommendations = await self.get_recommendations(customer_id, num_recommendations=limit)
            
            deals = []
            for product in recommendations.recommendations:
                discount = await self._calculate_personalized_discount(customer, product)
                if discount > 0:
                    deals.append({
                        "product": product,
                        "original_price": product['price'],
                        "discounted_price": product['price'] * (1 - discount),
                        "discount_percentage": discount * 100
                    })
            
            return deals
        except Exception as e:
            logger.error(f"Error getting personalized deals: {str(e)}")
            raise

    async def _calculate_personalized_discount(self, customer: Customer, product: Dict[str, Any]) -> float:
        try:
            base_discount = 0.05  # 5% base discount
            
            # Increase discount for loyal customers
            if customer.loyalty_points > 1000:
                base_discount += 0.03
            elif customer.loyalty_points > 500:
                base_discount += 0.02
            
            # Increase discount for customers who haven't purchased recently
            last_order = await self.db.query(Order).filter(Order.customer_id == customer.id).order_by(Order.order_date.desc()).first()
            if last_order:
                days_since_last_order = (datetime.utcnow() - last_order.order_date).days
                if days_since_last_order > 90:
                    base_discount += 0.03
                elif days_since_last_order > 30:
                    base_discount += 0.01
            
            # Cap the maximum discount at 15%
            return min(base_discount, 0.15)
        except Exception as e:
            logger.error(f"Error calculating personalized discount: {str(e)}")
            raise

