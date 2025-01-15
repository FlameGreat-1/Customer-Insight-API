import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
from app.core.config import settings
from app.core.logging import logger
import joblib

class RecommendationModel:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.product_features = None
        self.product_ids = None

    def train(self, products: List[Dict[str, str]]):
        try:
            self.product_ids = [p['id'] for p in products]
            product_descriptions = [f"{p['name']} {p['description']} {p['category']} {p['brand']}" for p in products]
            self.product_features = self.tfidf.fit_transform(product_descriptions)
            logger.info("Recommendation model training completed")
        except Exception as e:
            logger.error(f"Error in recommendation model training: {str(e)}")
            raise

    def get_recommendations(self, user_vector: np.ndarray, num_recommendations: int = 5) -> List[str]:
        try:
            if self.product_features is None:
                raise ValueError("Model not trained. Call train() first.")

            similarities = cosine_similarity(user_vector, self.product_features).flatten()
            top_indices = similarities.argsort()[-num_recommendations:][::-1]
            return [self.product_ids[i] for i in top_indices]
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise

    def get_user_vector(self, user_history: List[str]) -> np.ndarray:
        try:
            if self.product_features is None:
                raise ValueError("Model not trained. Call train() first.")

            user_vector = np.zeros(self.product_features.shape[1])
            for product_id in user_history:
                if product_id in self.product_ids:
                    index = self.product_ids.index(product_id)
                    user_vector += self.product_features[index].toarray().flatten()

            if np.linalg.norm(user_vector) > 0:
                user_vector = user_vector / np.linalg.norm(user_vector)

            return user_vector.reshape(1, -1)
        except Exception as e:
            logger.error(f"Error creating user vector: {str(e)}")
            raise

    def save(self, path: str):
        try:
            joblib.dump({
                'tfidf': self.tfidf,
                'product_features': self.product_features,
                'product_ids': self.product_ids
            }, path)
            logger.info(f"Recommendation model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving recommendation model: {str(e)}")
            raise

    def load(self, path: str):
        try:
            loaded = joblib.load(path)
            self.tfidf = loaded['tfidf']
            self.product_features = loaded['product_features']
            self.product_ids = loaded['product_ids']
            logger.info(f"Recommendation model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading recommendation model: {str(e)}")
            raise
