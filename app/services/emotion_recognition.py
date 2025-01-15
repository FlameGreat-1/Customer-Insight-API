# app/services/emotion_recognition.py

import asyncio
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.customer import Customer
from app.models.interaction import Interaction
from app.schemas.emotion_recognition import EmotionRecognitionResult, EmotionModel, EmotionDistribution
from app.core.logging import logger
from transformers import pipeline
import numpy as np
from datetime import datetime, timedelta
import librosa
import soundfile as sf

class EmotionRecognitionService:
    def __init__(self, db: Session):
        self.db = db
        self.text_emotion_recognizer = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
        self.audio_emotion_recognizer = pipeline("audio-classification", model="MIT/ast-finetuned-speech-commands-v2")

    async def recognize_emotion(self, text: str) -> EmotionRecognitionResult:
        try:
            result = await asyncio.to_thread(self.text_emotion_recognizer, text)
            emotion = result[0]['label']
            confidence = result[0]['score']

            emotion_distribution = {r['label']: r['score'] for r in result}

            return EmotionRecognitionResult(
                text=text,
                primary_emotion=emotion,
                emotion_distribution=emotion_distribution,
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"Error in text emotion recognition: {str(e)}")
            raise

    async def recognize_emotion_from_audio(self, audio_file: str) -> EmotionRecognitionResult:
        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_file, sr=16000)
            audio = librosa.resample(audio, sr, 16000)

            result = await asyncio.to_thread(self.audio_emotion_recognizer, audio)
            emotion = result[0]['label']
            confidence = result[0]['score']

            emotion_distribution = {r['label']: r['score'] for r in result}

            return EmotionRecognitionResult(
                text="Audio file",
                primary_emotion=emotion,
                emotion_distribution=emotion_distribution,
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"Error in audio emotion recognition: {str(e)}")
            raise

    async def store_batch_results(self, results: List[EmotionRecognitionResult], user_id: int):
        try:
            for result in results:
                emotion_log = EmotionLog(
                    customer_id=user_id,
                    text=result.text,
                    primary_emotion=result.primary_emotion,
                    emotion_distribution=result.emotion_distribution,
                    confidence=result.confidence,
                    timestamp=datetime.utcnow()
                )
                self.db.add(emotion_log)
            await self.db.commit()
        except Exception as e:
            logger.error(f"Error storing batch results: {str(e)}")
            await self.db.rollback()
            raise

    async def get_available_models(self) -> List[EmotionModel]:
        return [
            EmotionModel(
                id="distilbert-base-uncased-emotion",
                name="DistilBERT Emotion Recognition",
                description="Text-based emotion recognition model"
            ),
            EmotionModel(
                id="ast-finetuned-speech-commands-v2",
                name="AST Audio Emotion Recognition",
                description="Audio-based emotion recognition model"
            )
        ]

    async def train_model(self):
        try:
            logger.info("Starting emotion recognition model training...")

            # Fetch training data from database
            emotion_logs = await self.db.query(EmotionLog).all()
            texts = [log.text for log in emotion_logs]
            labels = [log.primary_emotion for log in emotion_logs]

            # Fine-tune the model (this is a placeholder, actual implementation would depend on the specific model being used)
            # In a real-world scenario, you might use a more sophisticated training process
            # or even outsource the training to a dedicated machine learning pipeline
            self.text_emotion_recognizer.model.train()
            for text, label in zip(texts, labels):
                inputs = self.text_emotion_recognizer.tokenizer(text, return_tensors="pt")
                outputs = self.text_emotion_recognizer.model(**inputs, labels=label)
                loss = outputs.loss
                loss.backward()

            logger.info("Emotion recognition model training completed")
        except Exception as e:
            logger.error(f"Error in emotion recognition model training: {str(e)}")
            raise

    async def get_performance_metrics(self) -> Dict[str, float]:
        try:
            total_recognitions = await self.db.query(func.count(EmotionLog.id)).scalar()
            avg_confidence = await self.db.query(func.avg(EmotionLog.confidence)).scalar() or 0

            # Calculate accuracy (assuming we have some ground truth labels)
            correct_predictions = await self.db.query(func.count(EmotionLog.id)).filter(EmotionLog.primary_emotion == EmotionLog.ground_truth_emotion).scalar()
            accuracy = correct_predictions / total_recognitions if total_recognitions > 0 else 0

            return {
                "total_recognitions": total_recognitions,
                "average_confidence": float(avg_confidence),
                "accuracy": accuracy
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise

    async def get_emotion_distribution(self, start_date: str, end_date: str) -> EmotionDistribution:
        try:
            query = self.db.query(EmotionLog.primary_emotion, func.count(EmotionLog.id)).\
                filter(EmotionLog.timestamp.between(start_date, end_date)).\
                group_by(EmotionLog.primary_emotion)
            results = await query.all()
            total = sum(count for _, count in results)
            distribution = {emotion: count / total for emotion, count in results}
            return EmotionDistribution(distribution=distribution)
        except Exception as e:
            logger.error(f"Error retrieving emotion distribution: {str(e)}")
            raise

