from typing import List, Dict
import numpy as np
from datetime import datetime
from app.core.logging import logger
from sklearn.preprocessing import StandardScaler
from scipy import stats

def create_customer_features(customer_data: List[Dict[str, any]]) -> List[Dict[str, float]]:
    try:
        features = []
        for customer in customer_data:
            # RFM (Recency, Frequency, Monetary) features
            last_purchase = datetime.strptime(customer['last_purchase_date'], '%Y-%m-%d')
            recency = (datetime.now() - last_purchase).days
            frequency = customer['total_purchases']
            monetary = customer['total_spend']

            # Customer lifetime value
            clv = customer['total_spend'] / (datetime.now().year - customer['first_purchase_year'] + 1)

            # Average order value
            aov = customer['total_spend'] / customer['total_purchases'] if customer['total_purchases'] > 0 else 0

            # Purchase frequency (purchases per year)
            purchase_frequency = customer['total_purchases'] / (datetime.now().year - customer['first_purchase_year'] + 1)

            # Customer age
            customer_age = datetime.now().year - customer['birth_year']

            # Time since first purchase
            time_since_first_purchase = (datetime.now().year - customer['first_purchase_year'])

            # Purchase trend (increasing or decreasing)
            purchase_trend = calculate_purchase_trend(customer['purchase_history'])

            # Customer segment
            segment = determine_customer_segment(recency, frequency, monetary)

            features.append({
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

        return features
    except Exception as e:
        logger.error(f"Error in creating customer features: {str(e)}")
        raise

def create_product_features(product_data: List[Dict[str, any]]) -> List[Dict[str, float]]:
    try:
        features = []
        for product in product_data:
            # Sales velocity
            sales_velocity = product['total_sales'] / product['days_available']

            # Price elasticity (if historical price data is available)
            price_elasticity = calculate_price_elasticity(product['price_history'], product['sales_history'])

            # Product age
            product_age = (datetime.now() - datetime.strptime(product['launch_date'], '%Y-%m-%d')).days

            # Review score
            avg_review_score = np.mean(product['review_scores']) if product['review_scores'] else 0

            # Seasonality index
            seasonality_index = calculate_seasonality_index(product['monthly_sales'])

            # Stock turnover ratio
            stock_turnover = calculate_stock_turnover(product['sales_quantity'], product['average_inventory'])

            # Profit margin
            profit_margin = (product['price'] - product['cost']) / product['price']

            features.append({
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

        return features
    except Exception as e:
        logger.error(f"Error in creating product features: {str(e)}")
        raise

def calculate_price_elasticity(price_history: List[float], sales_history: List[int]) -> float:
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

def calculate_purchase_trend(purchase_history: List[Dict[str, any]]) -> float:
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

def determine_customer_segment(recency: int, frequency: int, monetary: float) -> str:
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

def calculate_seasonality_index(monthly_sales: List[float]) -> float:
    try:
        if len(monthly_sales) < 12:
            return 0

        # Calculate moving average
        moving_avg = np.convolve(monthly_sales, np.ones(12), 'valid') / 12

        # Calculate seasonal indices
        seasonal_indices = np.array(monthly_sales[:12]) / moving_avg

        return np.std(seasonal_indices)
    except Exception as e:
        logger.error(f"Error in calculating seasonality index: {str(e)}")
        raise

def calculate_stock_turnover(sales_quantity: int, average_inventory: int) -> float:
    try:
        if average_inventory == 0:
            return 0
        return sales_quantity / average_inventory
    except Exception as e:
        logger.error(f"Error in calculating stock turnover: {str(e)}")
        raise

def normalize_features(features: List[Dict[str, float]]) -> List[Dict[str, float]]:
    try:
        scaler = StandardScaler()
        feature_names = list(features[0].keys())
        feature_values = [list(feature.values()) for feature in features]
        
        normalized_values = scaler.fit_transform(feature_values)
        
        return [dict(zip(feature_names, values)) for values in normalized_values]
    except Exception as e:
        logger.error(f"Error in normalizing features: {str(e)}")
        raise
