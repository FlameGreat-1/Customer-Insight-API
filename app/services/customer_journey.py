# app/services/customer_journey.py

import asyncio
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.customer import Customer
from app.models.interaction import Interaction
from app.models.order import Order
from app.models.feedback import Feedback
from app.schemas.customer_journey import (
    CustomerJourneyResult,
    TouchpointAnalysisResult,
    JourneyOptimizationResult,
    JourneySegmentResult
)
from app.core.logging import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
from fastapi.responses import HTMLResponse

class CustomerJourneyService:
    def __init__(self, db: Session):
        self.db = db
        self.scaler = StandardScaler()
        self.journey_segmentation_model = KMeans(n_clusters=5, random_state=42)
        self.churn_prediction_model = RandomForestClassifier(n_estimators=100, random_state=42)

    async def analyze_journey(self, customer_id: int) -> CustomerJourneyResult:
        try:
            customer = await self.db.query(Customer).filter(Customer.id == customer_id).first()
            if not customer:
                raise ValueError(f"Customer with id {customer_id} not found")

            interactions = await self.db.query(Interaction).filter(Interaction.customer_id == customer_id).order_by(Interaction.timestamp).all()
            orders = await self.db.query(Order).filter(Order.customer_id == customer_id).order_by(Order.order_date).all()

            touchpoints = await self._extract_touchpoints(interactions, orders)
            journey_duration = (touchpoints[-1]['timestamp'] - touchpoints[0]['timestamp']).days if touchpoints else 0
            conversion_rate = len(orders) / len(interactions) if interactions else 0

            pain_points = await self._identify_pain_points(touchpoints)
            recommendations = await self._generate_recommendations(touchpoints, pain_points)
            sentiment_analysis = await self._analyze_sentiment(touchpoints)
            customer_value = await self._calculate_customer_value(customer, orders)

            return CustomerJourneyResult(
                customer_id=customer_id,
                touchpoints=touchpoints,
                journey_duration=journey_duration,
                conversion_rate=conversion_rate,
                pain_points=pain_points,
                recommendations=recommendations,
                sentiment_analysis=sentiment_analysis,
                customer_value=customer_value
            )
        except Exception as e:
            logger.error(f"Error in analyzing customer journey: {str(e)}")
            raise

    async def _extract_touchpoints(self, interactions: List[Interaction], orders: List[Order]) -> List[Dict[str, Any]]:
        touchpoints = []
        for interaction in interactions:
            touchpoints.append({
                'type': 'interaction',
                'timestamp': interaction.timestamp,
                'channel': interaction.channel,
                'details': interaction.metadata
            })
        for order in orders:
            touchpoints.append({
                'type': 'order',
                'timestamp': order.order_date,
                'channel': 'e-commerce',
                'details': {'order_id': order.id, 'total_amount': order.total_amount}
            })
        return sorted(touchpoints, key=lambda x: x['timestamp'])

    async def _identify_pain_points(self, touchpoints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pain_points = []
        for i in range(len(touchpoints) - 1):
            time_diff = (touchpoints[i+1]['timestamp'] - touchpoints[i]['timestamp']).total_seconds()
            if time_diff > 86400:  # More than 24 hours between touchpoints
                pain_points.append({
                    'type': 'long_gap',
                    'start': touchpoints[i]['timestamp'],
                    'end': touchpoints[i+1]['timestamp'],
                    'duration': time_diff
                })
            if touchpoints[i]['type'] == 'interaction' and touchpoints[i+1]['type'] == 'interaction':
                if touchpoints[i]['channel'] != touchpoints[i+1]['channel']:
                    pain_points.append({
                        'type': 'channel_switch',
                        'from_channel': touchpoints[i]['channel'],
                        'to_channel': touchpoints[i+1]['channel'],
                        'timestamp': touchpoints[i+1]['timestamp']
                    })
        return pain_points

    async def _generate_recommendations(self, touchpoints: List[Dict[str, Any]], pain_points: List[Dict[str, Any]]) -> List[str]:
        recommendations = []
        if pain_points:
            recommendations.append("Address identified pain points in the customer journey.")
        
        channel_counts = {}
        for touchpoint in touchpoints:
            channel_counts[touchpoint['channel']] = channel_counts.get(touchpoint['channel'], 0) + 1
        preferred_channel = max(channel_counts, key=channel_counts.get)
        recommendations.append(f"Focus on {preferred_channel} as the customer's preferred channel.")

        if not any(order['type'] == 'order' for order in touchpoints):
            recommendations.append("Implement targeted conversion strategies to encourage first purchase.")

        # Add personalized recommendations based on touchpoint analysis
        if len(touchpoints) > 10 and channel_counts.get('email', 0) < 2:
            recommendations.append("Increase email engagement to provide personalized offers.")

        if 'social_media' in channel_counts and channel_counts['social_media'] > 5:
            recommendations.append("Leverage social media for brand advocacy and user-generated content.")

        return recommendations

    async def analyze_touchpoint(self, touchpoint_id: str) -> TouchpointAnalysisResult:
        try:
            touchpoint = await self.db.query(Interaction).filter(Interaction.id == touchpoint_id).first()
            if not touchpoint:
                raise ValueError(f"Touchpoint with id {touchpoint_id} not found")

            total_interactions = await self.db.query(func.count(Interaction.id)).filter(Interaction.channel == touchpoint.channel).scalar()
            conversions = await self.db.query(func.count(Order.id)).join(Interaction, Order.customer_id == Interaction.customer_id).filter(Interaction.channel == touchpoint.channel).scalar()

            engagement_rate = await self._calculate_engagement_rate(touchpoint)
            conversion_impact = conversions / total_interactions if total_interactions > 0 else 0
            average_time_spent = await self._calculate_average_time_spent(touchpoint)
            customer_feedback = await self._get_customer_feedback(touchpoint)
            improvement_suggestions = await self._generate_improvement_suggestions(touchpoint, engagement_rate, conversion_impact)

            return TouchpointAnalysisResult(
                touchpoint_id=touchpoint_id,
                engagement_rate=engagement_rate,
                conversion_impact=conversion_impact,
                average_time_spent=average_time_spent,
                customer_feedback=customer_feedback,
                improvement_suggestions=improvement_suggestions
            )
        except Exception as e:
            logger.error(f"Error in analyzing touchpoint: {str(e)}")
            raise

    async def _calculate_engagement_rate(self, touchpoint: Interaction) -> float:
        total_interactions = await self.db.query(func.count(Interaction.id)).filter(Interaction.channel == touchpoint.channel).scalar()
        engaged_interactions = await self.db.query(func.count(Interaction.id)).filter(
            Interaction.channel == touchpoint.channel,
            Interaction.duration > timedelta(minutes=2)
        ).scalar()
        return engaged_interactions / total_interactions if total_interactions > 0 else 0

    async def _calculate_average_time_spent(self, touchpoint: Interaction) -> float:
        avg_duration = await self.db.query(func.avg(Interaction.duration)).filter(Interaction.channel == touchpoint.channel).scalar()
        return avg_duration.total_seconds() if avg_duration else 0

    async def _get_customer_feedback(self, touchpoint: Interaction) -> List[Dict[str, Any]]:
        feedbacks = await self.db.query(Feedback).filter(
            Feedback.customer_id == touchpoint.customer_id,
            Feedback.timestamp > touchpoint.timestamp,
            Feedback.timestamp <= touchpoint.timestamp + timedelta(days=1)
        ).all()
        return [{'content': feedback.content, 'sentiment': feedback.sentiment} for feedback in feedbacks]

    async def _generate_improvement_suggestions(self, touchpoint: Interaction, engagement_rate: float, conversion_impact: float) -> List[str]:
        suggestions = []
        if engagement_rate < 0.5:
            suggestions.append(f"Improve content relevance for {touchpoint.channel} to increase engagement.")
        if conversion_impact < 0.1:
            suggestions.append(f"Optimize conversion funnel for {touchpoint.channel}.")
        if touchpoint.duration.total_seconds() < 60:
            suggestions.append(f"Increase interactivity in {touchpoint.channel} to extend user engagement time.")
        return suggestions

    async def optimize_journey(self, segment_id: str) -> JourneyOptimizationResult:
        try:
            customers = await self.db.query(Customer).filter(Customer.segment == segment_id).all()
            journeys = await asyncio.gather(*[self.analyze_journey(customer.id) for customer in customers])

            optimization_id = f"opt_{segment_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            suggested_changes = await self._generate_optimization_suggestions(journeys)
            expected_impact = await self._estimate_optimization_impact(journeys, suggested_changes)
            implementation_timeline = await self._create_implementation_timeline(suggested_changes)

            return JourneyOptimizationResult(
                segment_id=segment_id,
                optimization_id=optimization_id,
                suggested_changes=suggested_changes,
                expected_impact=expected_impact,
                implementation_timeline=implementation_timeline
            )
        except Exception as e:
            logger.error(f"Error in optimizing journey: {str(e)}")
            raise

    async def _generate_optimization_suggestions(self, journeys: List[CustomerJourneyResult]) -> List[Dict[str, Any]]:
        all_pain_points = [point for journey in journeys for point in journey.pain_points]
        pain_point_counts = {}
        for point in all_pain_points:
            key = (point['type'], point.get('from_channel'), point.get('to_channel'))
            pain_point_counts[key] = pain_point_counts.get(key, 0) + 1

        suggestions = []
        for (point_type, from_channel, to_channel), count in pain_point_counts.items():
            if count > len(journeys) * 0.2:  # If more than 20% of journeys have this pain point
                if point_type == 'long_gap':
                    suggestions.append({
                        'type': 'reduce_gap',
                        'description': f"Implement re-engagement strategies to reduce long gaps in customer interactions.",
                        'priority': 'high'
                    })
                elif point_type == 'channel_switch':
                    suggestions.append({
                        'type': 'improve_channel_transition',
                        'description': f"Improve transition from {from_channel} to {to_channel} to reduce friction.",
                        'priority': 'medium'
                    })

        conversion_rates = [journey.conversion_rate for journey in journeys]
        avg_conversion_rate = np.mean(conversion_rates)
        if avg_conversion_rate < 0.1:
            suggestions.append({
                'type': 'increase_conversion',
                'description': "Implement personalized product recommendations to increase conversion rate.",
                'priority': 'high'
            })

        avg_journey_duration = np.mean([journey.journey_duration for journey in journeys])
        if avg_journey_duration > 30:
            suggestions.append({
                'type': 'shorten_journey',
                'description': "Streamline the customer journey to reduce time to conversion.",
                'priority': 'medium'
            })

        return suggestions

    async def _estimate_optimization_impact(self, journeys: List[CustomerJourneyResult], suggested_changes: List[Dict[str, Any]]) -> Dict[str, float]:
        current_conversion_rate = np.mean([journey.conversion_rate for journey in journeys])
        current_journey_duration = np.mean([journey.journey_duration for journey in journeys])

        # Estimate impact based on the number and type of suggested changes
        estimated_conversion_increase = sum(0.05 if change['priority'] == 'high' else 0.03 for change in suggested_changes)
        estimated_duration_decrease = sum(3 if change['priority'] == 'high' else 1 for change in suggested_changes)

        return {
            'estimated_conversion_rate': min(current_conversion_rate * (1 + estimated_conversion_increase), 1.0),
            'estimated_journey_duration': max(current_journey_duration - estimated_duration_decrease, 1),
            'potential_revenue_increase': estimated_conversion_increase * len(journeys) * 100  # Assuming $100 average order value
        }

    async def _create_implementation_timeline(self, suggested_changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        timeline = []
        start_date = datetime.utcnow()
        for i, change in enumerate(suggested_changes):
            duration = 2 if change['priority'] == 'high' else 3  # weeks
            timeline.append({
                'change': change['description'],
                'start_date': start_date + timedelta(weeks=i*duration),
                'end_date': start_date + timedelta(weeks=(i+1)*duration),
                'status': 'planned',
                'priority': change['priority']
            })
        return timeline

    async def segment_journeys(self, num_segments: int = 5) -> List[JourneySegmentResult]:
        try:
            customers = await self.db.query(Customer).all()
            journey_features = await asyncio.gather(*[self._extract_journey_features(customer.id) for customer in customers])

            scaled_features = self.scaler.fit_transform(journey_features)
            self.journey_segmentation_model.n_clusters = num_segments
            segments = self.journey_segmentation_model.fit_predict(scaled_features)

            results = []
            for segment_id in range(num_segments):
                segment_customers = [customer for customer, seg in zip(customers, segments) if seg == segment_id]
                segment_size = len(segment_customers)
                avg_journey_duration = np.mean([f[0] for f, seg in zip(journey_features, segments) if seg == segment_id])
                common_touchpoints = await self._get_common_touchpoints(segment_customers)
                conversion_rate = np.mean([f[1] for f, seg in zip(journey_features, segments) if seg == segment_id])
                avg_customer_value = np.mean([f[4] for f, seg in zip(journey_features, segments) if seg == segment_id])

                results.append(JourneySegmentResult(
                    segment_id=f"Segment_{segment_id}",
                    segment_name=f"Journey Segment {segment_id}",
                    segment_size=segment_size,
                    average_journey_duration=avg_journey_duration,
                    common_touchpoints=common_touchpoints,
                    conversion_rate=conversion_rate,
                    average_customer_value=avg_customer_value
                ))

            return results
        except Exception as e:
            logger.error(f"Error in segmenting journeys: {str(e)}")
            raise

    async def _extract_journey_features(self, customer_id: int) -> List[float]:
        journey = await self.analyze_journey(customer_id)
        return [
            journey.journey_duration,
            journey.conversion_rate,
            len(journey.touchpoints),
            len(set(t['channel'] for t in journey.touchpoints)),
            len(journey.pain_points),
            journey.customer_value
        ]

    async def _get_common_touchpoints(self, customers: List[Customer]) -> List[str]:
        all_touchpoints = []
        for customer in customers:
            interactions = await self.db.query(Interaction).filter(Interaction.customer_id == customer.id).all()
            all_touchpoints.extend([interaction.channel for interaction in interactions])
        
        touchpoint_counts = pd.Series(all_touchpoints).value_counts()
        common_touchpoints = touchpoint_counts[touchpoint_counts > len(customers) * 0.5].index.tolist()
        return common_touchpoints

    async def get_journey_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        try:
            interactions = await self.db.query(Interaction).filter(Interaction.timestamp.between(start_date, end_date)).all()
            orders = await self.db.query(Order).filter(Order.order_date.between(start_date, end_date)).all()

            total_interactions = len(interactions)
            total_orders = len(orders)
            unique_customers = len(set([i.customer_id for i in interactions] + [o.customer_id for o in orders]))
            
            channel_distribution = pd.Series([i.channel for i in interactions]).value_counts().to_dict()
            avg_journey_duration = np.mean([(max(i.timestamp for i in customer_interactions) - min(i.timestamp for i in customer_interactions)).days 
                                            for customer_id, customer_interactions in pd.DataFrame(interactions).groupby('customer_id')])

            # Calculate additional metrics
            total_revenue = sum(o.total_amount for o in orders)
            avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
            repeat_purchase_rate = len([c for c, count in pd.Series([o.customer_id for o in orders]).value_counts().items() if count > 1]) / unique_customers if unique_customers > 0 else 0

            return {
                'total_interactions': total_interactions,
                'total_orders': total_orders,
                'unique_customers': unique_customers,
                'conversion_rate': total_orders / total_interactions if total_interactions > 0 else 0,
                'channel_distribution': channel_distribution,
                'average_journey_duration': avg_journey_duration,
                'total_revenue': total_revenue,
                'average_order_value': avg_order_value,
                'repeat_purchase_rate': repeat_purchase_rate
            }
        except Exception as e:
            logger.error(f"Error retrieving journey metrics: {str(e)}")
            raise

    async def predict_churn(self, customer_id: int) -> Dict[str, Any]:
        try:
            customer = await self.db.query(Customer).filter(Customer.id == customer_id).first()
            if not customer:
                raise ValueError(f"Customer with id {customer_id} not found")

            journey = await self.analyze_journey(customer_id)
            recent_interactions = [t for t in journey.touchpoints if (datetime.utcnow() - t['timestamp']).days <= 30]

            features = [
                journey.journey_duration,
                journey.conversion_rate,
                len(recent_interactions),
                len(journey.pain_points),
                customer.total_spend,
                customer.loyalty_points
            ]

            # Use the trained churn prediction model
            churn_probability = self.churn_prediction_model.predict_proba([features])[0][1]

            risk_factors = []
            if len(recent_interactions) < 3:
                risk_factors.append('Low recent activity')
            if len(journey.pain_points) > 2:
                risk_factors.append('Multiple pain points')
            if journey.conversion_rate < 0.1:
                risk_factors.append('Low conversion rate')
            if customer.total_spend < 100:
                risk_factors.append('Low total spend')

            recommended_actions = [
                'Send personalized re-engagement email',
                'Offer special promotion based on past purchases',
                'Conduct customer satisfaction survey',
                'Provide personalized product recommendations'
            ]

            if churn_probability > 0.7:
                recommended_actions.append('Initiate high-priority retention campaign')
            elif churn_probability > 0.5:
                recommended_actions.append('Enroll in loyalty program')

            return {
                'customer_id': customer_id,
                'churn_probability': churn_probability,
                'risk_factors': risk_factors,
                'recommended_actions': recommended_actions
            }
        except Exception as e:
            logger.error(f"Error predicting customer churn: {str(e)}")
            raise

    async def visualize_journey(self, customer_id: int) -> HTMLResponse:
        try:
            journey = await self.analyze_journey(customer_id)
            
            G = nx.DiGraph()
            for i, touchpoint in enumerate(journey.touchpoints):
                G.add_node(i, title=f"{touchpoint['type']} - {touchpoint['channel']}", 
                           label=f"{touchpoint['type']}\n{touchpoint['channel']}")
                if i > 0:
                    G.add_edge(i-1, i)

            net = Network(notebook=True, directed=True, height="500px", width="100%")
            net.from_nx(G)
            net.set_options("""
            var options = {
                "nodes": {
                    "shape": "dot",
                    "size": 30,
                    "font": {
                        "size": 12,
                        "face": "Tahoma"
                    }
                },
                "edges": {
                    "arrows": {
                        "to": {
                            "enabled": true
                        }
                    },
                    "smooth": {
                        "type": "curvedCW",
                        "forceDirection": "none"
                    }
                },
                "physics": {
                    "barnesHut": {
                        "gravitationalConstant": -15000,
                        "springLength": 250,
                        "springConstant": 0.04
                    },
                    "minVelocity": 0.75
                }
            }
            """)
            return HTMLResponse(content=net.generate_html(), status_code=200)
        except Exception as e:
            logger.error(f"Error visualizing customer journey: {str(e)}")
            raise

    async def analyze_sentiment(self, touchpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            sentiments = [t.get('sentiment', 'neutral') for t in touchpoints if 'sentiment' in t]
            sentiment_counts = pd.Series(sentiments).value_counts()
            overall_sentiment = sentiment_counts.index[0] if not sentiment_counts.empty else 'neutral'
            
            sentiment_trend = []
            window_size = 5
            for i in range(0, len(sentiments), window_size):
                window = sentiments[i:i+window_size]
                window_sentiment = pd.Series(window).value_counts().index[0]
                sentiment_trend.append(window_sentiment)

            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_distribution': sentiment_counts.to_dict(),
                'sentiment_trend': sentiment_trend
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            raise

    async def calculate_customer_value(self, customer: Customer, orders: List[Order]) -> float:
        try:
            total_spend = sum(order.total_amount for order in orders)
            frequency = len(orders) / ((datetime.utcnow() - customer.registration_date).days / 365) if customer.registration_date else 1
            recency = (datetime.utcnow() - max(order.order_date for order in orders)).days if orders else 365
            
            # Simple customer value calculation
            customer_value = (total_spend * frequency) / (recency + 1)
            return customer_value
        except Exception as e:
            logger.error(f"Error calculating customer value: {str(e)}")
            raise

    async def generate_journey_report(self, customer_id: int) -> Dict[str, Any]:
        try:
            journey = await self.analyze_journey(customer_id)
            churn_prediction = await self.predict_churn(customer_id)
            sentiment_analysis = await self.analyze_sentiment(journey.touchpoints)

            report = {
                'customer_id': customer_id,
                'journey_summary': {
                    'duration': journey.journey_duration,
                    'touchpoints': len(journey.touchpoints),
                    'conversion_rate': journey.conversion_rate,
                    'customer_value': journey.customer_value
                },
                'pain_points': journey.pain_points,
                'sentiment_analysis': sentiment_analysis,
                'churn_prediction': churn_prediction,
                'recommendations': journey.recommendations
            }

            return report
        except Exception as e:
            logger.error(f"Error generating journey report: {str(e)}")
            raise

    async def train_churn_prediction_model(self):
        try:
            customers = await self.db.query(Customer).all()
            features = []
            labels = []

            for customer in customers:
                journey = await self.analyze_journey(customer.id)
                recent_interactions = [t for t in journey.touchpoints if (datetime.utcnow() - t['timestamp']).days <= 30]
                
                features.append([
                    journey.journey_duration,
                    journey.conversion_rate,
                    len(recent_interactions),
                    len(journey.pain_points),
                    customer.total_spend,
                    customer.loyalty_points
                ])
                
                # Assuming a customer is churned if they haven't interacted in the last 90 days
                is_churned = 1 if not recent_interactions and journey.touchpoints else 0
                labels.append(is_churned)

            X = np.array(features)
            y = np.array(labels)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            self.churn_prediction_model.fit(X_train, y_train)

            # Evaluate the model
            accuracy = self.churn_prediction_model.score(X_test, y_test)
            logger.info(f"Churn prediction model trained with accuracy: {accuracy}")

        except Exception as e:
            logger.error(f"Error training churn prediction model: {str(e)}")
            raise
