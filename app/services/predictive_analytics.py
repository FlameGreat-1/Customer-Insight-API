# app/services/predictive_analytics.py

import asyncio
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.customer import Customer
from app.models.order import Order
from app.schemas.predictive_analytics import PredictionResult, ModelPerformance, FeatureImportance
from app.core.logging import logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PredictiveAnalyticsService:
    def __init__(self, db: Session):
        self.db = db
        self.churn_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ltv_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    async def make_prediction(self, customer_id: int, prediction_type: str) -> PredictionResult:
        try:
            customer = await self.db.query(Customer).filter(Customer.id == customer_id).first()
            if not customer:
                raise ValueError(f"Customer with id {customer_id} not found")

            features = await self._extract_features(customer)
            scaled_features = self.scaler.transform([features])

            if prediction_type == "churn":
                prediction = self.churn_model.predict_proba(scaled_features)[0]
                churn_probability = prediction[1]
                is_churn = churn_probability > 0.5
                confidence = max(prediction)
                return PredictionResult(
                    customer_id=customer_id,
                    prediction_type="churn",
                    prediction=is_churn,
                    confidence=confidence,
                    explanation=f"The customer has a {churn_probability:.2f} probability of churning."
                )
            elif prediction_type == "ltv":
                prediction = self.ltv_model.predict(scaled_features)[0]
                confidence = self.ltv_model.score(scaled_features, [prediction])
                return PredictionResult(
                    customer_id=customer_id,
                    prediction_type="ltv",
                    prediction=prediction,
                    confidence=confidence,
                    explanation=f"The predicted lifetime value of the customer is ${prediction:.2f}."
                )
            else:
                raise ValueError(f"Invalid prediction type: {prediction_type}")
        except Exception as e:
            logger.error(f"Error in making prediction: {str(e)}")
            raise

    async def _extract_features(self, customer: Customer) -> List[float]:
        try:
            current_date = datetime.utcnow()
            last_order_date = await self.db.query(func.max(Order.order_date)).filter(Order.customer_id == customer.id).scalar()
            
            recency = (current_date - last_order_date).days if last_order_date else 365
            frequency = await self.db.query(func.count(Order.id)).filter(Order.customer_id == customer.id).scalar()
            monetary = float(customer.total_spend)

            return [
                float(customer.total_spend),
                float(customer.loyalty_points),
                float(recency),
                float(frequency),
                monetary,
                len(customer.interactions),
                customer.age,
                int(customer.gender == "Male"),
                int(customer.status == "Active")
            ]
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    async def retrain_model(self, prediction_type: str):
        try:
            logger.info(f"Starting {prediction_type} model retraining...")

            customers = await self.db.query(Customer).all()
            features = []
            targets = []

            for customer in customers:
                customer_features = await self._extract_features(customer)
                features.append(customer_features)
                
                if prediction_type == "churn":
                    is_churned = customer.status == "Churned"
                    targets.append(int(is_churned))
                elif prediction_type == "ltv":
                    ltv = await self._calculate_ltv(customer)
                    targets.append(ltv)

            X = np.array(features)
            y = np.array(targets)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            if prediction_type == "churn":
                self.churn_model.fit(X_train_scaled, y_train)
                y_pred = self.churn_model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                logger.info(f"Churn model retrained. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
            elif prediction_type == "ltv":
                self.ltv_model.fit(X_train_scaled, y_train)
                y_pred = self.ltv_model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                logger.info(f"LTV model retrained. MSE: {mse}, R2: {r2}")

            logger.info(f"{prediction_type.upper()} model retraining completed")
        except Exception as e:
            logger.error(f"Error in model retraining: {str(e)}")
            raise

    async def _calculate_ltv(self, customer: Customer) -> float:
        try:
            orders = await self.db.query(Order).filter(Order.customer_id == customer.id).all()
            total_revenue = sum(order.total_amount for order in orders)
            customer_age = (datetime.utcnow() - customer.created_at).days / 365.25
            return total_revenue / customer_age if customer_age > 0 else 0
        except Exception as e:
            logger.error(f"Error calculating LTV: {str(e)}")
            raise

    async def get_model_performance(self, prediction_type: str) -> ModelPerformance:
        try:
            customers = await self.db.query(Customer).all()
            features = []
            targets = []

            for customer in customers:
                customer_features = await self._extract_features(customer)
                features.append(customer_features)
                
                if prediction_type == "churn":
                    is_churned = customer.status == "Churned"
                    targets.append(int(is_churned))
                elif prediction_type == "ltv":
                    ltv = await self._calculate_ltv(customer)
                    targets.append(ltv)

            X = np.array(features)
            y = np.array(targets)

            X_scaled = self.scaler.transform(X)

            if prediction_type == "churn":
                y_pred = self.churn_model.predict(X_scaled)
                accuracy = accuracy_score(y, y_pred)
                precision = precision_score(y, y_pred)
                recall = recall_score(y, y_pred)
                f1 = f1_score(y, y_pred)
                return ModelPerformance(
                    prediction_type="churn",
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1
                )
            elif prediction_type == "ltv":
                y_pred = self.ltv_model.predict(X_scaled)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                return ModelPerformance(
                    prediction_type="ltv",
                    mse=mse,
                    r2=r2
                )
            else:
                raise ValueError(f"Invalid prediction type: {prediction_type}")
        except Exception as e:
            logger.error(f"Error getting model performance: {str(e)}")
            raise

    async def get_feature_importance(self, prediction_type: str) -> List[FeatureImportance]:
        try:
            feature_names = [
                "total_spend", "loyalty_points", "recency", "frequency", "monetary",
                "interaction_count", "age", "is_male", "is_active"
            ]
            
            if prediction_type == "churn":
                importances = self.churn_model.feature_importances_
            elif prediction_type == "ltv":
                importances = self.ltv_model.feature_importances_
            else:
                raise ValueError(f"Invalid prediction type: {prediction_type}")

            feature_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            return [FeatureImportance(feature=feature, importance=importance) for feature, importance in feature_importances]
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise

