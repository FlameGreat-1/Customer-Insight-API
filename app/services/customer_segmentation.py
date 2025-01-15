# app/services/customer_segmentation.py

import asyncio
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.customer import Customer
from app.schemas.customer_segmentation import SegmentationResult, SegmentationModel
from app.core.logging import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class CustomerSegmentationService:
    def __init__(self, db: Session):
        self.db = db
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=5, random_state=42)
        self.feature_columns = ['total_spend', 'loyalty_points', 'recency', 'frequency']

    async def segment_customer(self, customer_id: int) -> SegmentationResult:
        try:
            customer = await self.db.query(Customer).filter(Customer.id == customer_id).first()
            if not customer:
                raise ValueError(f"Customer with id {customer_id} not found")

            features = await self._extract_features(customer)
            scaled_features = self.scaler.transform([features])
            segment = self.model.predict(scaled_features)[0]
            confidence = self._calculate_confidence(scaled_features, segment)

            return SegmentationResult(
                customer_id=customer_id,
                segment=f"Segment_{segment}",
                confidence=confidence,
                features=dict(zip(self.feature_columns, features))
            )
        except Exception as e:
            logger.error(f"Error in customer segmentation: {str(e)}")
            raise

    async def _extract_features(self, customer: Customer) -> List[float]:
        current_date = datetime.utcnow()
        last_order_date = await self.db.query(func.max(Order.order_date)).filter(Order.customer_id == customer.id).scalar()
        
        recency = (current_date - last_order_date).days if last_order_date else 365
        frequency = await self.db.query(func.count(Order.id)).filter(Order.customer_id == customer.id).scalar()

        return [
            float(customer.total_spend),
            float(customer.loyalty_points),
            float(recency),
            float(frequency)
        ]

    def _calculate_confidence(self, features: np.ndarray, segment: int) -> float:
        distances = self.model.transform(features)
        segment_distances = distances[0]
        confidence = 1 - (segment_distances[segment] / np.sum(segment_distances))
        return float(confidence)

    async def segment_all_customers(self) -> Dict[int, str]:
        try:
            customers = await self.db.query(Customer).all()
            features = await asyncio.gather(*[self._extract_features(customer) for customer in customers])
            
            scaled_features = self.scaler.fit_transform(features)
            segments = self.model.fit_predict(scaled_features)

            segmentation_results = {customer.id: f"Segment_{segment}" for customer, segment in zip(customers, segments)}
            
            await self._update_customer_segments(segmentation_results)
            
            return segmentation_results
        except Exception as e:
            logger.error(f"Error in segmenting all customers: {str(e)}")
            raise

    async def _update_customer_segments(self, segmentation_results: Dict[int, str]):
        try:
            for customer_id, segment in segmentation_results.items():
                await self.db.query(Customer).filter(Customer.id == customer_id).update({"segment": segment})
            await self.db.commit()
        except Exception as e:
            logger.error(f"Error updating customer segments: {str(e)}")
            await self.db.rollback()
            raise

    async def get_segment_distribution(self) -> Dict[str, float]:
        try:
            query = self.db.query(Customer.segment, func.count(Customer.id)).\
                group_by(Customer.segment)
            results = await query.all()
            total = sum(count for _, count in results)
            return {segment: count / total for segment, count in results}
        except Exception as e:
            logger.error(f"Error retrieving segment distribution: {str(e)}")
            raise

    async def get_segment_characteristics(self, segment: str) -> Dict[str, Any]:
        try:
            customers = await self.db.query(Customer).filter(Customer.segment == segment).all()
            features = await asyncio.gather(*[self._extract_features(customer) for customer in customers])
            df = pd.DataFrame(features, columns=self.feature_columns)
            
            characteristics = {
                "total_spend": {
                    "mean": df['total_spend'].mean(),
                    "median": df['total_spend'].median(),
                    "std": df['total_spend'].std()
                },
                "loyalty_points": {
                    "mean": df['loyalty_points'].mean(),
                    "median": df['loyalty_points'].median(),
                    "std": df['loyalty_points'].std()
                },
                "recency": {
                    "mean": df['recency'].mean(),
                    "median": df['recency'].median(),
                    "std": df['recency'].std()
                },
                "frequency": {
                    "mean": df['frequency'].mean(),
                    "median": df['frequency'].median(),
                    "std": df['frequency'].std()
                }
            }
            return characteristics
        except Exception as e:
            logger.error(f"Error getting segment characteristics: {str(e)}")
            raise

    async def retrain_model(self, n_clusters: int = 5):
        try:
            logger.info("Starting model retraining...")
            customers = await self.db.query(Customer).all()
            features = await asyncio.gather(*[self._extract_features(customer) for customer in customers])
            
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(features)
            
            self.model = KMeans(n_clusters=n_clusters, random_state=42)
            self.model.fit(scaled_features)
            
            logger.info("Model retraining completed")
            
            # Update all customer segments with the new model
            await self.segment_all_customers()
        except Exception as e:
            logger.error(f"Error in model retraining: {str(e)}")
            raise

    async def get_available_models(self) -> List[SegmentationModel]:
        # In a real-world scenario, you might have multiple models stored and retrievable
        return [
            SegmentationModel(
                id="kmeans_5",
                name="K-Means (5 clusters)",
                description="Customer segmentation using K-Means algorithm with 5 clusters"
            )
        ]

    async def get_performance_metrics(self) -> Dict[str, float]:
        try:
            customers = await self.db.query(Customer).all()
            features = await asyncio.gather(*[self._extract_features(customer) for customer in customers])
            scaled_features = self.scaler.transform(features)
            
            inertia = self.model.inertia_
            silhouette = silhouette_score(scaled_features, self.model.labels_)
            
            return {
                "inertia": inertia,
                "silhouette_score": silhouette
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise

