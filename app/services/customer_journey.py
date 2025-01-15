# app/services/customer_journey.py

import asyncio
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.customer import Customer
from app.models.interaction import Interaction
from app.models.order import Order
from app.schemas.customer_journey import (
    CustomerJourneyResult,
    TouchpointAnalysisResult,
    JourneyOptimizationResult,
    JourneySegmentResult
)
from app.core.logging import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
from pyvis.network import Network

class CustomerJourneyService:
    def __init__(self, db: Session):
        self.db = db
        self.scaler = StandardScaler()
        self.journey_segmentation_model = KMeans(n_clusters=5, random_state=42)

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

            return CustomerJourneyResult(
                customer_id=customer_id,
                touchpoints=touchpoints,
                journey_duration=journey_duration,
                conversion_rate=conversion_rate,
                pain_points=pain_points,
                recommendations=recommendations
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

        return suggestions

    async def _estimate_optimization_impact(self, journeys: List[CustomerJourneyResult], suggested_changes: List[Dict[str, Any]]) -> Dict[str, float]:
        current_conversion_rate = np.mean([journey.conversion_rate for journey in journeys])
        current_journey_duration = np.mean([journey.journey_duration for journey in journeys])

        # Estimate impact based on the number and type of suggested changes
        estimated_conversion_increase = len(suggested_changes) * 0.05  # Assume each change increases conversion by 5%
        estimated_duration_decrease = len(suggested_changes) * 2  # Assume each change decreases duration by 2 days

        return {
            'estimated_conversion_rate': min(current_conversion_rate * (1 + estimated_conversion_increase), 1.0),
            'estimated_journey_duration': max(current_journey_duration - estimated_duration_decrease, 1)
        }

    async def _create_implementation_timeline(self, suggested_changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        timeline = []
        start_date = datetime.utcnow()
        for i, change in enumerate(suggested_changes):
            timeline.append({
                'change': change['description'],
                'start_date': start_date + timedelta(weeks=i*2),
                'end_date': start_date + timedelta(weeks=i*2+2),
                'status': 'planned'
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

                results.append(JourneySegmentResult(
                    segment_id=f"Segment_{segment_id}",
                    segment_name=f"Journey Segment {segment_id}",
                    segment_size=segment_size,
                    average_journey_duration=avg_journey_duration,
                    common_touchpoints=common_touchpoints,
                    conversion_rate=conversion_rate
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
            len(journey.pain_points)
        ]

    async def _get_common_touchpoints(self, customers: List[Customer]) -> List[str]:
        all_touchpoints = []
        for customer in customers:
            interactions = await self.db.query(Interaction).filter(Interaction.customer_id == customer.id).all()
            all_touchpoints.extend([interaction.channel for interaction in interactions])
        
        touchpoint_counts = pd.Series(all_touchpoints).value_counts()
        common_touchpoints = touchpoint_counts[touchpoint_counts > len(customers) * 0.5].index.tolist()
        return common_touchpoints

    async def get_journey_metrics(self, start_date: str, end_date: str) -> Dict[str, Any]:
        try:
            interactions = await self.db.query(Interaction).filter(Interaction.timestamp.between(start_date, end_date)).all()
            orders = await self.db.query(Order).filter(Order.order_date.between(start_date, end_date)).all()

            total_interactions = len(interactions)
            total_orders = len(orders)
            unique_customers = len(set([i.customer_id for i in interactions] + [o.customer_id for o in orders]))
            
            channel_distribution = pd.Series([i.channel for i in interactions]).value_counts().to_dict()
            avg_journey_duration = np.mean([(max(i.timestamp for i in customer_interactions) - min(i.timestamp for i in customer_interactions)).days 
                                            for customer_id, customer_interactions in pd.DataFrame(interactions).groupby('customer_id')])

            return {
                'total_interactions': total_interactions,
                'total_orders': total_orders,
                'unique_customers': unique_customers,
                'conversion_rate': total_orders / total_interactions if total_interactions > 0 else 0,
                'channel_distribution': channel_distribution,
                'average_journey_duration': avg_journey_duration
            }
        except Exception as e:
            logger.error(f"Error retrieving journey metrics: {str(e)}")
            raise

    async def predict_churn(self, customer_id: str) -> Dict[str, Any]:
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

            # This is a placeholder for a more sophisticated churn prediction model
            churn_probability = 1 / (1 + np.exp(-sum(features)))  # Simple logistic function

            return {
                'customer_id': customer_id,
                'churn_probability': churn_probability,
                'risk_factors': [
                    'Low recent activity' if len(recent_interactions) < 3 else None,
                    'Multiple pain points' if len(journey.pain_points) > 2 else None,
                    'Low conversion rate' if journey.conversion_rate < 0.1 else None
                ],
                'recommended_actions': [
                    'Send personalized re-engagement email',
                    'Offer special promotion',
                    'Conduct customer satisfaction survey'
                ]
            }
        except Exception as e:
            logger.error(f"Error predicting customer churn: {str(e)}")
            raise

    async def visualize_journey(self, customer_id: str) -> str:
        try:
            journey = await self.analyze_journey(customer_id)
            
            G = nx.DiGraph()
            for i, touchpoint in enumerate(journey.touchpoints):
                G.add_node(i, title=f"{touchpoint['type']} - {touchpoint['channel']}", 
                           label=f"{touchpoint['type']}\n{touchpoint['channel']}")
                if i > 0:
                    G.add_edge(i-1, i)

            net = Network(notebook=True, directed=True)
            net.from_nx(G)
            net.show_buttons(filter_=['physics'])
            return net.generate_html()
        except Exception as e:
            logger.error(f"Error visualizing customer journey: {str(e)}")
            raise

