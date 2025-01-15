from transformers import pipeline
from typing import List, Dict
from app.core.config import settings
from app.core.logging import logger

class EmotionDetectionModel:
    def __init__(self):
        self.emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        try:
            results = self.emotion_pipeline(texts)
            return [
                {
                    "emotion": result['label'],
                    "score": result['score']
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            raise

    def save(self, path: str):
        logger.info(f"Emotion detection model saving not implemented. Model is loaded from Hugging Face Hub.")

    def load(self, path: str):
        logger.info(f"Emotion detection model loading not implemented. Model is loaded from Hugging Face Hub.")
