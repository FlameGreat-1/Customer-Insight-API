import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict
from app.core.config import settings
from app.core.logging import logger
import joblib

class SegmentationModel:
    def __init__(self, n_clusters: int = 5):
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

    def preprocess(self, data: List[Dict[str, float]]) -> np.ndarray:
        try:
            features = np.array([[d['recency'], d['frequency'], d['monetary']] for d in data])
            return self.scaler.fit_transform(features)
        except Exception as e:
            logger.error(f"Error in data preprocessing for segmentation: {str(e)}")
            raise

    def train(self, data: List[Dict[str, float]]):
        try:
            preprocessed_data = self.preprocess(data)
            self.model.fit(preprocessed_data)
            logger.info("Segmentation model training completed")
        except Exception as e:
            logger.error(f"Error in segmentation model training: {str(e)}")
            raise

    def predict(self, data: List[Dict[str, float]]) -> List[int]:
        try:
            preprocessed_data = self.preprocess(data)
            return self.model.predict(preprocessed_data).tolist()
        except Exception as e:
            logger.error(f"Error in segmentation prediction: {str(e)}")
            raise

    def get_segment_profiles(self) -> List[Dict[str, float]]:
        try:
            segment_centers = self.scaler.inverse_transform(self.model.cluster_centers_)
            return [
                {
                    "segment": i,
                    "recency": center[0],
                    "frequency": center[1],
                    "monetary": center[2]
                }
                for i, center in enumerate(segment_centers)
            ]
        except Exception as e:
            logger.error(f"Error getting segment profiles: {str(e)}")
            raise

    def save(self, path: str):
        try:
            joblib.dump({
                'scaler': self.scaler,
                'model': self.model
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
            logger.info(f"Segmentation model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading segmentation model: {str(e)}")
            raise
