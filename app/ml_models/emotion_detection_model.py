from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union
from app.core.config import settings
from app.core.logging import logger
import torch
import os

class EmotionDetectionModel:
    def __init__(self):
        self.model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
        self.tokenizer = None
        self.model = None
        self.emotion_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load()

    def load(self, path: str = None):
        try:
            if path and os.path.exists(path):
                logger.info(f"Loading emotion detection model from {path}")
                self.tokenizer = AutoTokenizer.from_pretrained(path)
                self.model = AutoModelForSequenceClassification.from_pretrained(path)
            else:
                logger.info(f"Loading emotion detection model from Hugging Face Hub: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self.model.to(self.device)
            self.emotion_pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=self.device)
            logger.info("Emotion detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading emotion detection model: {str(e)}")
            raise

    def predict(self, texts: Union[str, List[str]]) -> List[Dict[str, float]]:
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            results = self.emotion_pipeline(texts, top_k=None)
            
            processed_results = []
            for result in results:
                emotions = {item['label']: item['score'] for item in result}
                dominant_emotion = max(emotions, key=emotions.get)
                processed_results.append({
                    "dominant_emotion": dominant_emotion,
                    "dominant_score": emotions[dominant_emotion],
                    "emotions": emotions
                })
            
            return processed_results
        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            raise

    def save(self, path: str):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            logger.info(f"Saving emotion detection model to {path}")
            self.tokenizer.save_pretrained(path)
            self.model.save_pretrained(path)
            logger.info("Emotion detection model saved successfully")
        except Exception as e:
            logger.error(f"Error saving emotion detection model: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "num_labels": self.model.num_labels if self.model else None,
            "vocab_size": len(self.tokenizer) if self.tokenizer else None
        }

    def __str__(self):
        return f"EmotionDetectionModel(model_name={self.model_name}, device={self.device})"

    def __repr__(self):
        return self.__str__()
