# app/services/predictive_analytics.py

import asyncio
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.customer import Customer
from app.models.order import Order
from app.models.interaction import Interaction
from app.schemas.predictive_analytics import PredictionResult, ModelPerformance, FeatureImportance
from app.core.logging import logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os

class PredictiveAnalyticsService:
    def __init__(self, db: Session):
        self.db = db
        self.churn_model_path = "app/models/churn_model.joblib"
        self.ltv_model_path = "app/models/ltv_model.joblib"
        self.scaler_path = "app/models/feature_scaler.joblib"
        self.churn_model = self._load_model(self.churn_model_path, RandomForestClassifier(n_estimators=100, random_state=42))
        self.ltv_model = self._load_model(self.ltv_model_path, RandomForestRegressor(n_estimators=100, random_state=42))
        self.scaler = self._load_model(self.scaler_path, StandardScaler())

    def _load_model(self, path: str, default_model):
        if os.path.exists(path):
            return joblib.load(path)
        return default_model

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
                explanation = await self._generate_churn_explanation(customer, churn_probability)
                return PredictionResult(
                    customer_id=customer_id,
                    prediction_type="churn",
                    prediction=is_churn,
                    confidence=confidence,
                    explanation=explanation
                )
            elif prediction_type == "ltv":
                prediction = self.ltv_model.predict(scaled_features)[0]
                confidence = self.ltv_model.score(scaled_features, [prediction])
                explanation = await self._generate_ltv_explanation(customer, prediction)
                return PredictionResult(
                    customer_id=customer_id,
                    prediction_type="ltv",
                    prediction=prediction,
                    confidence=confidence,
                    explanation=explanation
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
            interaction_count = await self.db.query(func.count(Interaction.id)).filter(Interaction.customer_id == customer.id).scalar()

            avg_order_value = monetary / frequency if frequency > 0 else 0
            customer_lifetime = (current_date - customer.created_at).days
            purchase_frequency = frequency / (customer_lifetime / 365) if customer_lifetime > 0 else 0

            return [
                float(customer.total_spend),
                float(customer.loyalty_points),
                float(recency),
                float(frequency),
                monetary,
                float(interaction_count),
                float(customer.age),
                int(customer.gender == "Male"),
                int(customer.status == "Active"),
                avg_order_value,
                float(customer_lifetime),
                purchase_frequency
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
                joblib.dump(self.churn_model, self.churn_model_path)
            elif prediction_type == "ltv":
                self.ltv_model.fit(X_train_scaled, y_train)
                y_pred = self.ltv_model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                logger.info(f"LTV model retrained. MSE: {mse}, R2: {r2}")
                joblib.dump(self.ltv_model, self.ltv_model_path)

            joblib.dump(self.scaler, self.scaler_path)
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
                cv_scores = cross_val_score(self.churn_model, X_scaled, y, cv=5)
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
                    f1_score=f1,
                    cross_validation_score=np.mean(cv_scores)
                )
            elif prediction_type == "ltv":
                cv_scores = cross_val_score(self.ltv_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
                y_pred = self.ltv_model.predict(X_scaled)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                return ModelPerformance(
                    prediction_type="ltv",
                    mse=mse,
                    r2=r2,
                    cross_validation_score=-np.mean(cv_scores)
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
                "interaction_count", "age", "is_male", "is_active", "avg_order_value",
                "customer_lifetime", "purchase_frequency"
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

    async def _generate_churn_explanation(self, customer: Customer, churn_probability: float) -> str:
        try:
            feature_importances = await self.get_feature_importance("churn")
            top_features = feature_importances[:3]
            
            explanation = f"The customer has a {churn_probability:.2f} probability of churning. "
            explanation += "The top factors contributing to this prediction are:\n"
            
            for feature in top_features:
                if feature.feature == "recency":
                    last_order = await self.db.query(Order).filter(Order.customer_id == customer.id).order_by(Order.order_date.desc()).first()
                    if last_order:
                        days_since_last_order = (datetime.utcnow() - last_order.order_date).days
                        explanation += f"- Days since last order: {days_since_last_order}\n"
                elif feature.feature == "frequency":
                    order_count = await self.db.query(func.count(Order.id)).filter(Order.customer_id == customer.id).scalar()
                    explanation += f"- Total number of orders: {order_count}\n"
                elif feature.feature == "monetary":
                    explanation += f"- Total spend: ${customer.total_spend:.2f}\n"
                else:
                    explanation += f"- {feature.feature.replace('_', ' ').title()}\n"
            
            return explanation
        except Exception as e:
            logger.error(f"Error generating churn explanation: {str(e)}")
            raise

    async def _generate_ltv_explanation(self, customer: Customer, predicted_ltv: float) -> str:
        try:
            feature_importances = await self.get_feature_importance("ltv")
            top_features = feature_importances[:3]
            
            explanation = f"The predicted lifetime value of the customer is ${predicted_ltv:.2f}. "
            explanation += "The top factors contributing to this prediction are:\n"
            
            for feature in top_features:
                if feature.feature == "frequency":
                    order_count = await self.db.query(func.count(Order.id)).filter(Order.customer_id == customer.id).scalar()
                    explanation += f"- Total number of orders: {order_count}\n"
                elif feature.feature == "monetary":
                    explanation += f"- Total spend: ${customer.total_spend:.2f}\n"
                elif feature.feature == "customer_lifetime":
                    customer_age = (datetime.utcnow() - customer.created_at).days
                    explanation += f"- Customer age (days): {customer_age}\n"
                else:
                    explanation += f"- {feature.feature.replace('_', ' ').title()}\n"
            
            return explanation
        except Exception as e:
            logger.error(f"Error generating LTV explanation: {str(e)}")
            raise

    async def get_customer_segment(self, customer_id: int) -> str:
        try:
            customer = await self.db.query(Customer).filter(Customer.id == customer_id).first()
            if not customer:
                raise ValueError(f"Customer with id {customer_id} not found")

            features = await self._extract_features(customer)
            scaled_features = self.scaler.transform([features])

            churn_probability = self.churn_model.predict_proba(scaled_features)[0][1]
            predicted_ltv = self.ltv_model.predict(scaled_features)[0]

            if churn_probability > 0.7:
                return "High Risk"
            elif churn_probability > 0.4:
                return "Medium Risk"
            elif predicted_ltv > 1000:
                return "High Value"
            elif predicted_ltv > 500:
                return "Medium Value"
            else:
                return "Low Value"
        except Exception as e:
            logger.error(f"Error getting customer segment: {str(e)}")
            raise

    async def get_customer_recommendations(self, customer_id: int) -> List[str]:
        try:
            customer = await self.db.query(Customer).filter(Customer.id == customer_id).first()
            if not customer:
                raise ValueError(f"Customer with id {customer_id} not found")

            segment = await self.get_customer_segment(customer_id)
            churn_prediction = await self.make_prediction(customer_id, "churn")
            ltv_prediction = await self.make_prediction(customer_id, "ltv")

            recommendations = []

            if segment == "High Risk":
                recommendations.append("Implement immediate retention strategies")
                recommendations.append("Offer personalized discounts or promotions")
                recommendations.append("Conduct a customer satisfaction survey")
            elif segment == "Medium Risk":
                recommendations.append("Increase engagement through targeted email campaigns")
                recommendations.append("Provide special offers for loyalty program enrollment")
                recommendations.append("Highlight new products or services that may interest the customer")
            elif segment == "High Value":
                recommendations.append("Implement VIP customer program")
                recommendations.append("Offer exclusive early access to new products or services")
                recommendations.append("Provide personalized customer support")
            elif segment == "Medium Value":
                recommendations.append("Encourage increased purchase frequency through targeted promotions")
                recommendations.append("Cross-sell related products based on purchase history")
                recommendations.append("Invite to special events or webinars")
            else:  # Low Value
                recommendations.append("Encourage higher-value purchases through product recommendations")
                recommendations.append("Offer incentives for referrals")
                recommendations.append("Provide educational content to increase engagement")

            return recommendations
        except Exception as e:
            logger.error(f"Error getting customer recommendations: {str(e)}")
            raise

