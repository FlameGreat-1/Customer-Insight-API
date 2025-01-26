import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import List, Dict, Union
from app.core.config import settings
from app.core.logging import logger
import joblib
from datetime import datetime

class SegmentationModel:
    def __init__(self, n_clusters: int = 5):
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.feature_names = ['recency', 'frequency', 'monetary']
        self.last_trained = None
        self.silhouette_score = None

    def preprocess(self, data: List[Dict[str, float]]) -> np.ndarray:
        try:
            features = np.array([[d[feature] for feature in self.feature_names] for d in data])
            return self.scaler.fit_transform(features)
        except KeyError as e:
            logger.error(f"Missing feature in data: {str(e)}")
            raise ValueError(f"Data must contain all features: {self.feature_names}")
        except Exception as e:
            logger.error(f"Error in data preprocessing for segmentation: {str(e)}")
            raise

    def train(self, data: List[Dict[str, float]], optimize_clusters: bool = False):
        try:
            preprocessed_data = self.preprocess(data)
            
            if optimize_clusters:
                self._optimize_clusters(preprocessed_data)
            
            self.model.fit(preprocessed_data)
            self.silhouette_score = silhouette_score(preprocessed_data, self.model.labels_)
            self.last_trained = datetime.now()
            logger.info(f"Segmentation model training completed. Silhouette score: {self.silhouette_score:.4f}")
        except Exception as e:
            logger.error(f"Error in segmentation model training: {str(e)}")
            raise

    def _optimize_clusters(self, data: np.ndarray, max_clusters: int = 10):
        best_score = -1
        best_n_clusters = self.model.n_clusters

        for n_clusters in range(2, max_clusters + 1):
            temp_model = KMeans(n_clusters=n_clusters, random_state=42)
            temp_model.fit(data)
            score = silhouette_score(data, temp_model.labels_)
            
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters

        self.model = KMeans(n_clusters=best_n_clusters, random_state=42)
        logger.info(f"Optimized number of clusters: {best_n_clusters}")

    def predict(self, data: Union[Dict[str, float], List[Dict[str, float]]]) -> Union[int, List[int]]:
        try:
            if isinstance(data, dict):
                data = [data]
            preprocessed_data = self.preprocess(data)
            predictions = self.model.predict(preprocessed_data).tolist()
            return predictions[0] if len(predictions) == 1 else predictions
        except Exception as e:
            logger.error(f"Error in segmentation prediction: {str(e)}")
            raise

    def get_segment_profiles(self) -> List[Dict[str, float]]:
        try:
            segment_centers = self.scaler.inverse_transform(self.model.cluster_centers_)
            return [
                {
                    "segment": i,
                    **{feature: center[j] for j, feature in enumerate(self.feature_names)}
                }
                for i, center in enumerate(segment_centers)
            ]
        except Exception as e:
            logger.error(f"Error getting segment profiles: {str(e)}")
            raise

    def get_segment_distribution(self, data: List[Dict[str, float]]) -> Dict[int, float]:
        try:
            preprocessed_data = self.preprocess(data)
            predictions = self.model.predict(preprocessed_data)
            unique, counts = np.unique(predictions, return_counts=True)
            distribution = dict(zip(unique, counts / len(predictions)))
            return {int(segment): float(proportion) for segment, proportion in distribution.items()}
        except Exception as e:
            logger.error(f"Error getting segment distribution: {str(e)}")
            raise

    def save(self, path: str):
        try:
            joblib.dump({
                'scaler': self.scaler,
                'model': self.model,
                'feature_names': self.feature_names,
                'last_trained': self.last_trained,
                'silhouette_score': self.silhouette_score
            }, path)
            logger.info(f"Segmentation model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving segmentation model: {str(e)}")
            raise

    def load(self, path: str):
        try:
            loaded = joblib.load(path)
            self.scaler = loaded['scaler']
            self.model = loaded['model']
            self.feature_names = loaded['feature_names']
            self.last_trained = loaded['last_trained']
            self.silhouette_score = loaded['silhouette_score']
            logger.info(f"Segmentation model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading segmentation model: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Union[int, str, float]]:
        return {
            "n_clusters": self.model.n_clusters,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "last_trained": str(self.last_trained) if self.last_trained else "Never",
            "silhouette_score": float(self.silhouette_score) if self.silhouette_score is not None else None
        }

    def __str__(self):
        return f"SegmentationModel(n_clusters={self.model.n_clusters}, n_features={len(self.feature_names)}, last_trained={self.last_trained})"

    def __repr__(self):
        return self.__str__()
