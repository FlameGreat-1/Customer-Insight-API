# app/services/customer_segmentation.py

import asyncio
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.customer import Customer
from app.models.order import Order
from app.schemas.customer_segmentation import SegmentationResult, SegmentationModel
from app.core.logging import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os

class CustomerSegmentationService:
    def __init__(self, db: Session):
        self.db = db
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=5, random_state=42)
        self.feature_columns = ['total_spend', 'loyalty_points', 'recency', 'frequency', 'avg_order_value']
        self.model_path = "app/models/customer_segmentation_model.joblib"
        self.scaler_path = "app/models/customer_segmentation_scaler.joblib"
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            logger.info("Loaded existing segmentation model and scaler")
        else:
            logger.info("No existing model found. Using default model.")

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
        orders = await self.db.query(Order).filter(Order.customer_id == customer.id).all()
        
        last_order_date = max([order.order_date for order in orders]) if orders else None
        recency = (current_date - last_order_date).days if last_order_date else 365
        frequency = len(orders)
        avg_order_value = sum([order.total_amount for order in orders]) / frequency if frequency > 0 else 0

        return [
            float(customer.total_spend),
            float(customer.loyalty_points),
            float(recency),
            float(frequency),
            float(avg_order_value)
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
            
            # Save the updated model and scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info("Updated segmentation model and scaler saved")

            return segmentation_results
        except Exception as e:
            logger.error(f"Error in segmenting all customers: {str(e)}")
            raise

    async def _update_customer_segments(self, segmentation_results: Dict[int, str]):
        try:
            async with self.db.begin():
                for customer_id, segment in segmentation_results.items():
                    await self.db.execute(
                        Customer.__table__.update().
                        where(Customer.id == customer_id).
                        values(segment=segment)
                    )
        except Exception as e:
            logger.error(f"Error updating customer segments: {str(e)}")
            raise

    async def get_segment_distribution(self) -> Dict[str, float]:
        try:
            query = self.db.query(Customer.segment, func.count(Customer.id).label('count')).\
                group_by(Customer.segment)
            results = await self.db.execute(query)
            distribution = {row.segment: row.count for row in results}
            total = sum(distribution.values())
            return {segment: count / total for segment, count in distribution.items()}
        except Exception as e:
            logger.error(f"Error retrieving segment distribution: {str(e)}")
            raise

    async def get_segment_characteristics(self, segment: str) -> Dict[str, Any]:
        try:
            customers = await self.db.query(Customer).filter(Customer.segment == segment).all()
            features = await asyncio.gather(*[self._extract_features(customer) for customer in customers])
            df = pd.DataFrame(features, columns=self.feature_columns)
            
            characteristics = {
                column: {
                    "mean": df[column].mean(),
                    "median": df[column].median(),
                    "std": df[column].std(),
                    "min": df[column].min(),
                    "max": df[column].max()
                } for column in self.feature_columns
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
            
            # Save the new model and scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info("New segmentation model and scaler saved")

            # Update all customer segments with the new model
            await self.segment_all_customers()
        except Exception as e:
            logger.error(f"Error in model retraining: {str(e)}")
            raise

    async def get_available_models(self) -> List[SegmentationModel]:
        models = [
            SegmentationModel(
                id="kmeans_5",
                name="K-Means (5 clusters)",
                description="Customer segmentation using K-Means algorithm with 5 clusters"
            )
        ]
        
        # Check if there are any custom models saved
        custom_model_path = "app/models/custom_segmentation_model.joblib"
        if os.path.exists(custom_model_path):
            custom_model = joblib.load(custom_model_path)
            models.append(
                SegmentationModel(
                    id="custom_model",
                    name=f"Custom K-Means ({custom_model.n_clusters} clusters)",
                    description="Custom segmentation model with optimized parameters"
                )
            )
        
        return models

    async def get_performance_metrics(self) -> Dict[str, float]:
        try:
            customers = await self.db.query(Customer).all()
            features = await asyncio.gather(*[self._extract_features(customer) for customer in customers])
            scaled_features = self.scaler.transform(features)
            
            inertia = self.model.inertia_
            silhouette = silhouette_score(scaled_features, self.model.labels_)
            
            return {
                "inertia": float(inertia),
                "silhouette_score": float(silhouette),
                "num_clusters": self.model.n_clusters
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise

    async def optimize_model(self, max_clusters: int = 10):
        try:
            logger.info("Starting model optimization...")
            customers = await self.db.query(Customer).all()
            features = await asyncio.gather(*[self._extract_features(customer) for customer in customers])
            scaled_features = self.scaler.fit_transform(features)

            best_model = None
            best_silhouette = -1

            for n_clusters in range(2, max_clusters + 1):
                model = KMeans(n_clusters=n_clusters, random_state=42)
                model.fit(scaled_features)
                silhouette = silhouette_score(scaled_features, model.labels_)
                
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_model = model

            self.model = best_model
            logger.info(f"Model optimization completed. Best number of clusters: {self.model.n_clusters}")

            # Save the optimized model
            joblib.dump(self.model, "app/models/custom_segmentation_model.joblib")
            logger.info("Optimized segmentation model saved")

            # Update all customer segments with the new model
            await self.segment_all_customers()
        except Exception as e:
            logger.error(f"Error in model optimization: {str(e)}")
            raise

    async def get_segment_transition_matrix(self, start_date: datetime, end_date: datetime) -> Dict[str, Dict[str, int]]:
        try:
            query = self.db.query(
                Customer.id,
                func.lag(Customer.segment).over(
                    partition_by=Customer.id,
                    order_by=Customer.updated_at
                ).label('previous_segment'),
                Customer.segment.label('current_segment')
            ).filter(
                Customer.updated_at.between(start_date, end_date)
            )

            results = await self.db.execute(query)
            
            transition_matrix = {}
            for row in results:
                if row.previous_segment and row.current_segment:
                    if row.previous_segment not in transition_matrix:
                        transition_matrix[row.previous_segment] = {}
                    transition_matrix[row.previous_segment][row.current_segment] = \
                        transition_matrix[row.previous_segment].get(row.current_segment, 0) + 1

            return transition_matrix
        except Exception as e:
            logger.error(f"Error calculating segment transition matrix: {str(e)}")
            raise

    async def get_segment_cohort_analysis(self, cohort_period: str = 'monthly') -> pd.DataFrame:
        try:
            query = self.db.query(
                Customer.id,
                Customer.registration_date,
                Customer.segment,
                Order.order_date,
                Order.total_amount
            ).join(Order, Customer.id == Order.customer_id)

            results = await self.db.execute(query)
            df = pd.DataFrame(results, columns=['customer_id', 'registration_date', 'segment', 'order_date', 'total_amount'])

            df['cohort'] = df['registration_date'].dt.to_period(cohort_period)
            df['order_period'] = df['order_date'].dt.to_period(cohort_period)
            df['periods_since_registration'] = (df['order_period'] - df['cohort']).apply(lambda r: r.n)

            cohort_analysis = df.groupby(['cohort', 'periods_since_registration', 'segment'])['customer_id'].nunique().unstack(level=[1, 2])
            return cohort_analysis
        except Exception as e:
            logger.error(f"Error performing cohort analysis: {str(e)}")
            raise

    async def get_segment_feature_importance(self) -> Dict[str, float]:
        try:
            customers = await self.db.query(Customer).all()
            features = await asyncio.gather(*[self._extract_features(customer) for customer in customers])
            scaled_features = self.scaler.transform(features)

            feature_importance = {}
            for i, feature in enumerate(self.feature_columns):
                feature_values = scaled_features[:, i]
                correlation = np.abs(np.corrcoef(feature_values, self.model.labels_)[0, 1])
                feature_importance[feature] = float(correlation)

            return feature_importance
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise
