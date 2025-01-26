from typing import List, Dict, Any, Union
import numpy as np
from datetime import datetime
from app.core.logging import logger
from app.core.config import settings
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_customer_features(self, customer_data: List[Dict[str, Any]]) -> pd.DataFrame:
        try:
            features = []
            for customer in customer_data:
                last_purchase = datetime.strptime(customer['last_purchase_date'], '%Y-%m-%d')
                recency = (datetime.now() - last_purchase).days
                frequency = customer['total_purchases']
                monetary = customer['total_spend']

                clv = customer['total_spend'] / (datetime.now().year - customer['first_purchase_year'] + 1)
                aov = customer['total_spend'] / customer['total_purchases'] if customer['total_purchases'] > 0 else 0
                purchase_frequency = customer['total_purchases'] / (datetime.now().year - customer['first_purchase_year'] + 1)
                customer_age = datetime.now().year - customer['birth_year']
                time_since_first_purchase = (datetime.now().year - customer['first_purchase_year'])
                purchase_trend = self.calculate_purchase_trend(customer['purchase_history'])
                segment = self.determine_customer_segment(recency, frequency, monetary)

                features.append({
                    'customer_id': customer['id'],
                    'recency': recency,
                    'frequency': frequency,
                    'monetary': monetary,
                    'clv': clv,
                    'aov': aov,
                    'purchase_frequency': purchase_frequency,
                    'customer_age': customer_age,
                    'time_since_first_purchase': time_since_first_purchase,
                    'purchase_trend': purchase_trend,
                    'segment': segment
                })

            return pd.DataFrame(features)
        except Exception as e:
            logger.error(f"Error in creating customer features: {str(e)}")
            raise

    def create_product_features(self, product_data: List[Dict[str, Any]]) -> pd.DataFrame:
        try:
            features = []
            for product in product_data:
                sales_velocity = product['total_sales'] / product['days_available']
                price_elasticity = self.calculate_price_elasticity(product['price_history'], product['sales_history'])
                product_age = (datetime.now() - datetime.strptime(product['launch_date'], '%Y-%m-%d')).days
                avg_review_score = np.mean(product['review_scores']) if product['review_scores'] else 0
                seasonality_index = self.calculate_seasonality_index(product['monthly_sales'])
                stock_turnover = self.calculate_stock_turnover(product['sales_quantity'], product['average_inventory'])
                profit_margin = (product['price'] - product['cost']) / product['price']

                features.append({
                    'product_id': product['id'],
                    'sales_velocity': sales_velocity,
                    'price_elasticity': price_elasticity,
                    'product_age': product_age,
                    'avg_review_score': avg_review_score,
                    'category': product['category'],
                    'brand': product['brand'],
                    'seasonality_index': seasonality_index,
                    'stock_turnover': stock_turnover,
                    'profit_margin': profit_margin
                })

            return pd.DataFrame(features)
        except Exception as e:
            logger.error(f"Error in creating product features: {str(e)}")
            raise

    def calculate_price_elasticity(self, price_history: List[float], sales_history: List[int]) -> float:
        try:
            if len(price_history) != len(sales_history) or len(price_history) < 2:
                return 0

            price_changes = np.diff(price_history) / price_history[:-1]
            sales_changes = np.diff(sales_history) / sales_history[:-1]

            valid_indices = (price_changes != 0) & (sales_changes != 0)
            if not np.any(valid_indices):
                return 0

            elasticities = sales_changes[valid_indices] / price_changes[valid_indices]
            return np.mean(elasticities)
        except Exception as e:
            logger.error(f"Error in calculating price elasticity: {str(e)}")
            raise

    def calculate_purchase_trend(self, purchase_history: List[Dict[str, Any]]) -> float:
        try:
            if len(purchase_history) < 2:
                return 0

            purchase_amounts = [purchase['amount'] for purchase in purchase_history]
            purchase_dates = [datetime.strptime(purchase['date'], '%Y-%m-%d') for purchase in purchase_history]

            slope, _, _, _, _ = stats.linregress([(date - purchase_dates[0]).days for date in purchase_dates], purchase_amounts)
            return slope
        except Exception as e:
            logger.error(f"Error in calculating purchase trend: {str(e)}")
            raise

    def determine_customer_segment(self, recency: int, frequency: int, monetary: float) -> str:
        try:
            r_score = 1 if recency <= 30 else (2 if recency <= 90 else (3 if recency <= 180 else 4))
            f_score = 4 if frequency >= 10 else (3 if frequency >= 5 else (2 if frequency >= 2 else 1))
            m_score = 4 if monetary >= 1000 else (3 if monetary >= 500 else (2 if monetary >= 100 else 1))

            total_score = r_score + f_score + m_score

            if total_score <= 5:
                return "Lost"
            elif total_score <= 8:
                return "At Risk"
            elif total_score <= 10:
                return "Active"
            else:
                return "VIP"
        except Exception as e:
            logger.error(f"Error in determining customer segment: {str(e)}")
            raise

    def calculate_seasonality_index(self, monthly_sales: List[float]) -> float:
        try:
            if len(monthly_sales) < 12:
                return 0

            moving_avg = np.convolve(monthly_sales, np.ones(12), 'valid') / 12
            seasonal_indices = np.array(monthly_sales[:12]) / moving_avg

            return np.std(seasonal_indices)
        except Exception as e:
            logger.error(f"Error in calculating seasonality index: {str(e)}")
            raise

    def calculate_stock_turnover(self, sales_quantity: int, average_inventory: int) -> float:
        try:
            if average_inventory == 0:
                return 0
            return sales_quantity / average_inventory
        except Exception as e:
            logger.error(f"Error in calculating stock turnover: {str(e)}")
            raise

    def normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            features[numeric_columns] = self.scaler.fit_transform(features[numeric_columns])
            return features
        except Exception as e:
            logger.error(f"Error in normalizing features: {str(e)}")
            raise

    def engineer_features(self, customer_data: List[Dict[str, Any]], product_data: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        try:
            customer_features = self.create_customer_features(customer_data)
            product_features = self.create_product_features(product_data)

            if settings.NORMALIZE_FEATURES:
                customer_features = self.normalize_features(customer_features)
                product_features = self.normalize_features(product_features)

            return {
                'customer_features': customer_features,
                'product_features': product_features
            }
        except Exception as e:
            logger.error(f"Error in feature engineering process: {str(e)}")
            raise

# Initialize the FeatureEngineer
feature_engineer = FeatureEngineer()
