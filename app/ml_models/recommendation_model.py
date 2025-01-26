import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Union
from app.core.config import settings
from app.core.logging import logger
import joblib
from datetime import datetime

class RecommendationModel:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.product_features = None
        self.product_ids = None
        self.product_data = None
        self.last_trained = None

    def train(self, products: List[Dict[str, Union[str, float]]]):
        try:
            self.product_ids = [p['id'] for p in products]
            product_descriptions = [f"{p['name']} {p['description']} {p['category']} {p['brand']}" for p in products]
            self.product_features = self.tfidf.fit_transform(product_descriptions)
            self.product_data = products
            self.last_trained = datetime.now()
            logger.info(f"Recommendation model training completed. {len(products)} products processed.")
        except Exception as e:
            logger.error(f"Error in recommendation model training: {str(e)}")
            raise

    def get_recommendations(self, user_vector: np.ndarray, num_recommendations: int = 5, 
                            filters: Dict[str, Union[str, List[str]]] = None) -> List[Dict[str, Union[str, float]]]:
        try:
            if self.product_features is None:
                raise ValueError("Model not trained. Call train() first.")

            similarities = cosine_similarity(user_vector, self.product_features).flatten()
            
            if filters:
                mask = self._apply_filters(filters)
                similarities = similarities * mask

            top_indices = similarities.argsort()[-num_recommendations:][::-1]
            recommendations = [self.product_data[i] for i in top_indices]
            
            for rec, sim in zip(recommendations, similarities[top_indices]):
                rec['similarity_score'] = float(sim)

            return recommendations
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise

    def get_user_vector(self, user_history: List[str], weights: List[float] = None) -> np.ndarray:
        try:
            if self.product_features is None:
                raise ValueError("Model not trained. Call train() first.")

            user_vector = np.zeros(self.product_features.shape[1])
            
            if weights is None:
                weights = [1] * len(user_history)
            elif len(weights) != len(user_history):
                raise ValueError("Length of weights must match length of user_history")

            for product_id, weight in zip(user_history, weights):
                if product_id in self.product_ids:
                    index = self.product_ids.index(product_id)
                    user_vector += weight * self.product_features[index].toarray().flatten()

            if np.linalg.norm(user_vector) > 0:
                user_vector = user_vector / np.linalg.norm(user_vector)

            return user_vector.reshape(1, -1)
        except Exception as e:
            logger.error(f"Error creating user vector: {str(e)}")
            raise

    def _apply_filters(self, filters: Dict[str, Union[str, List[str]]]) -> np.ndarray:
        mask = np.ones(len(self.product_ids))
        for key, value in filters.items():
            if isinstance(value, str):
                mask *= np.array([p[key] == value for p in self.product_data])
            elif isinstance(value, list):
                mask *= np.array([p[key] in value for p in self.product_data])
        return mask

    def save(self, path: str):
        try:
            joblib.dump({
                'tfidf': self.tfidf,
                'product_features': self.product_features,
                'product_ids': self.product_ids,
                'product_data': self.product_data,
                'last_trained': self.last_trained
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
            self.product_data = loaded['product_data']
            self.last_trained = loaded['last_trained']
            logger.info(f"Recommendation model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading recommendation model: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Union[int, str]]:
        return {
            "num_products": len(self.product_ids) if self.product_ids else 0,
            "num_features": self.product_features.shape[1] if self.product_features is not None else 0,
            "last_trained": str(self.last_trained) if self.last_trained else "Never"
        }

    def update_product(self, product: Dict[str, Union[str, float]]):
        try:
            if product['id'] in self.product_ids:
                index = self.product_ids.index(product['id'])
                self.product_data[index] = product
                product_description = f"{product['name']} {product['description']} {product['category']} {product['brand']}"
                self.product_features[index] = self.tfidf.transform([product_description])
                logger.info(f"Product {product['id']} updated in recommendation model")
            else:
                self.product_ids.append(product['id'])
                self.product_data.append(product)
                product_description = f"{product['name']} {product['description']} {product['category']} {product['brand']}"
                new_feature = self.tfidf.transform([product_description])
                self.product_features = np.vstack((self.product_features.toarray(), new_feature.toarray()))
                logger.info(f"Product {product['id']} added to recommendation model")
        except Exception as e:
            logger.error(f"Error updating product in recommendation model: {str(e)}")
            raise

    def __str__(self):
        return f"RecommendationModel(num_products={len(self.product_ids) if self.product_ids else 0}, last_trained={self.last_trained})"

    def __repr__(self):
        return self.__str__()
