# app/services/emotion_recognition.py

import asyncio
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.customer import Customer
from app.models.interaction import Interaction
from app.models.emotion_log import EmotionLog
from app.models.order import Order
from app.schemas.emotion_recognition import EmotionRecognitionResult, EmotionModel, EmotionDistribution
from app.core.logging import logger
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from datetime import datetime, timedelta
import librosa
import soundfile as sf
import joblib
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class EmotionRecognitionService:
    class AudioEmotionModel(torch.nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
            self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
            self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
            self.pool = torch.nn.MaxPool1d(kernel_size=2)
            self.fc1 = torch.nn.Linear(256 * 2000, 512)
            self.fc2 = torch.nn.Linear(512, num_classes)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = self.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    def __init__(self, db: Session):
        self.db = db
        self.text_model_path = "app/models/text_emotion_model.pth"
        self.audio_model_path = "app/models/audio_emotion_model.pth"
        self.text_tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
        self.text_model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
        self.audio_model = pipeline("audio-classification", model="MIT/ast-finetuned-speech-commands-v2")
        self._load_models()

    def _load_models(self):
        if os.path.exists(self.text_model_path):
            self.text_model.load_state_dict(torch.load(self.text_model_path))
            logger.info("Loaded existing text emotion recognition model")
        if os.path.exists(self.audio_model_path):
            self.audio_model.model.load_state_dict(torch.load(self.audio_model_path))
            logger.info("Loaded existing audio emotion recognition model")

    async def recognize_emotion(self, text: str) -> EmotionRecognitionResult:
        try:
            inputs = self.text_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.text_model(**inputs)
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            emotion_distribution = {
                self.text_model.config.id2label[i]: float(prob)
                for i, prob in enumerate(probabilities[0])
            }
            primary_emotion = max(emotion_distribution, key=emotion_distribution.get)
            confidence = emotion_distribution[primary_emotion]

            return EmotionRecognitionResult(
                text=text,
                primary_emotion=primary_emotion,
                emotion_distribution=emotion_distribution,
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"Error in text emotion recognition: {str(e)}")
            raise

    async def recognize_emotion_from_audio(self, audio_file: str) -> EmotionRecognitionResult:
        try:
            audio, sr = librosa.load(audio_file, sr=16000)
            audio = librosa.resample(audio, sr, 16000)

            result = await asyncio.to_thread(self.audio_model, audio)
            emotion_distribution = {r['label']: r['score'] for r in result}
            primary_emotion = max(emotion_distribution, key=emotion_distribution.get)
            confidence = emotion_distribution[primary_emotion]

            return EmotionRecognitionResult(
                text="Audio file",
                primary_emotion=primary_emotion,
                emotion_distribution=emotion_distribution,
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"Error in audio emotion recognition: {str(e)}")
            raise

    async def store_batch_results(self, results: List[EmotionRecognitionResult], user_id: int):
        try:
            async with self.db.begin():
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
            logger.info(f"Stored {len(results)} emotion recognition results for user {user_id}")
        except Exception as e:
            logger.error(f"Error storing batch results: {str(e)}")
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

    async def train_model(self, model_type: str = "text"):
        try:
            logger.info(f"Starting {model_type} emotion recognition model training...")

            emotion_logs = await self.db.query(EmotionLog).all()
            texts = [log.text for log in emotion_logs]
            labels = [log.primary_emotion for log in emotion_logs]

            if model_type == "text":
                model = self.text_model
                tokenizer = self.text_tokenizer
                optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

                model.train()
                for epoch in range(3):  # Number of epochs can be adjusted
                    for text, label in zip(texts, labels):
                        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                        labels = torch.tensor([model.config.label2id[label]])
                        outputs = model(**inputs, labels=labels)
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                torch.save(model.state_dict(), self.text_model_path)
                logger.info("Text emotion recognition model training completed and saved")

            elif model_type == "audio":
                logger.info("Starting audio emotion recognition model training...")
    
                # Load audio data and labels
                audio_data = []
                audio_labels = []
                for log in emotion_logs:
                    if log.audio_file_path:
                        try:
                            audio, sr = librosa.load(log.audio_file_path, sr=16000)
                            audio = librosa.resample(audio, sr, 16000)
                            audio_data.append(audio)
                            audio_labels.append(log.primary_emotion)
                        except Exception as e:
                            logger.error(f"Error loading audio file {log.audio_file_path}: {str(e)}")

                if not audio_data:
                    raise ValueError("No audio data available for training")

                # Convert labels to numerical format
                label_to_id = {label: i for i, label in enumerate(set(audio_labels))}
                id_to_label = {i: label for label, i in label_to_id.items()}
                numerical_labels = [label_to_id[label] for label in audio_labels]

                # Prepare dataset
                dataset = list(zip(audio_data, numerical_labels))
                train_size = int(0.8 * len(dataset))
                train_dataset = dataset[:train_size]
                val_dataset = dataset[train_size:]

                # Initialize model, loss function, and optimizer
                num_classes = len(label_to_id)
                model = self.AudioEmotionModel(num_classes)
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                # Training loop
                num_epochs = 10
                batch_size = 32

                for epoch in range(num_epochs):
                    model.train()
                    train_loss = 0.0
                    for i in range(0, len(train_dataset), batch_size):
                        batch = train_dataset[i:i+batch_size]
                        audio_batch, label_batch = zip(*batch)
                        
                        audio_tensor = torch.FloatTensor(audio_batch).unsqueeze(1)  # Add channel dimension
                        label_tensor = torch.LongTensor(label_batch)

                        optimizer.zero_grad()
                        outputs = model(audio_tensor)
                        loss = criterion(outputs, label_tensor)
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()

                    # Validation
                    model.eval()
                    val_loss = 0.0
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for i in range(0, len(val_dataset), batch_size):
                            batch = val_dataset[i:i+batch_size]
                            audio_batch, label_batch = zip(*batch)
                            
                            audio_tensor = torch.FloatTensor(audio_batch).unsqueeze(1)
                            label_tensor = torch.LongTensor(label_batch)

                            outputs = model(audio_tensor)
                            loss = criterion(outputs, label_tensor)
                            val_loss += loss.item()

                            _, predicted = torch.max(outputs.data, 1)
                            total += label_tensor.size(0)
                            correct += (predicted == label_tensor).sum().item()

                    train_loss /= len(train_dataset) / batch_size
                    val_loss /= len(val_dataset) / batch_size
                    accuracy = 100 * correct / total

                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

                # Save the trained model
                torch.save(model.state_dict(), self.audio_model_path)
                
                # Update the audio emotion recognizer with the new model
                self.audio_model.model = model
                self.audio_model.config.id2label = id_to_label
                self.audio_model.config.label2id = label_to_id

                logger.info("Audio emotion recognition model training completed and saved")

            else:
                raise ValueError(f"Unknown model type: {model_type}")

        except Exception as e:
            logger.error(f"Error in emotion recognition model training: {str(e)}")
            raise

    async def get_performance_metrics(self) -> Dict[str, float]:
        try:
            total_recognitions = await self.db.query(func.count(EmotionLog.id)).scalar()
            avg_confidence = await self.db.query(func.avg(EmotionLog.confidence)).scalar() or 0

            # Assuming we have ground truth labels for a subset of the data
            evaluation_logs = await self.db.query(EmotionLog).filter(EmotionLog.ground_truth_emotion.isnot(None)).all()
            
            if evaluation_logs:
                y_true = [log.ground_truth_emotion for log in evaluation_logs]
                y_pred = [log.primary_emotion for log in evaluation_logs]
                
                accuracy = accuracy_score(y_true, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            else:
                accuracy = precision = recall = f1 = 0

            return {
                "total_recognitions": total_recognitions,
                "average_confidence": float(avg_confidence),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise

    async def get_emotion_distribution(self, start_date: datetime, end_date: datetime) -> EmotionDistribution:
        try:
            query = self.db.query(EmotionLog.primary_emotion, func.count(EmotionLog.id).label('count')).\
                filter(EmotionLog.timestamp.between(start_date, end_date)).\
                group_by(EmotionLog.primary_emotion)
            results = await self.db.execute(query)
            
            distribution = {}
            total = 0
            for row in results:
                distribution[row.primary_emotion] = row.count
                total += row.count

            normalized_distribution = {emotion: count / total for emotion, count in distribution.items()}
            return EmotionDistribution(distribution=normalized_distribution)
        except Exception as e:
            logger.error(f"Error retrieving emotion distribution: {str(e)}")
            raise

    async def get_emotion_trends(self, start_date: datetime, end_date: datetime, interval: str = 'day') -> Dict[str, List[float]]:
        try:
            if interval not in ['hour', 'day', 'week', 'month']:
                raise ValueError("Invalid interval. Choose from 'hour', 'day', 'week', or 'month'.")

            query = self.db.query(
                func.date_trunc(interval, EmotionLog.timestamp).label('time_bucket'),
                EmotionLog.primary_emotion,
                func.count(EmotionLog.id).label('count')
            ).filter(
                EmotionLog.timestamp.between(start_date, end_date)
            ).group_by(
                'time_bucket', EmotionLog.primary_emotion
            ).order_by('time_bucket')

            results = await self.db.execute(query)

            trends = {}
            for row in results:
                if row.primary_emotion not in trends:
                    trends[row.primary_emotion] = []
                trends[row.primary_emotion].append({
                    'timestamp': row.time_bucket,
                    'count': row.count
                })

            return trends
        except Exception as e:
            logger.error(f"Error retrieving emotion trends: {str(e)}")
            raise

    async def get_customer_emotion_profile(self, customer_id: int) -> Dict[str, Any]:
        try:
            query = self.db.query(
                EmotionLog.primary_emotion,
                func.count(EmotionLog.id).label('count'),
                func.avg(EmotionLog.confidence).label('avg_confidence')
            ).filter(
                EmotionLog.customer_id == customer_id
            ).group_by(EmotionLog.primary_emotion)

            results = await self.db.execute(query)

            profile = {
                'emotions': {},
                'dominant_emotion': None,
                'average_confidence': 0
            }

            total_count = 0
            for row in results:
                profile['emotions'][row.primary_emotion] = {
                    'count': row.count,
                    'average_confidence': float(row.avg_confidence)
                }
                total_count += row.count
                profile['average_confidence'] += row.count * row.avg_confidence

            if total_count > 0:
                profile['average_confidence'] /= total_count
                profile['dominant_emotion'] = max(profile['emotions'], key=lambda x: profile['emotions'][x]['count'])

            return profile
        except Exception as e:
            logger.error(f"Error retrieving customer emotion profile: {str(e)}")
            raise

    async def analyze_emotion_impact(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        try:
            query = self.db.query(
                EmotionLog.primary_emotion,
                func.count(Interaction.id).label('interaction_count'),
                func.avg(Interaction.duration).label('avg_duration'),
                func.avg(Order.total_amount).label('avg_order_value')
            ).join(
                Interaction, EmotionLog.customer_id == Interaction.customer_id
            ).outerjoin(
                Order, EmotionLog.customer_id == Order.customer_id
            ).filter(
                EmotionLog.timestamp.between(start_date, end_date),
                Interaction.timestamp.between(start_date, end_date)
            ).group_by(EmotionLog.primary_emotion)

            results = await self.db.execute(query)

            impact_analysis = {}
            for row in results:
                impact_analysis[row.primary_emotion] = {
                    'interaction_count': row.interaction_count,
                    'average_interaction_duration': float(row.avg_duration.total_seconds()) if row.avg_duration else 0,
                    'average_order_value': float(row.avg_order_value) if row.avg_order_value else 0
                }

            return impact_analysis
        except Exception as e:
            logger.error(f"Error analyzing emotion impact: {str(e)}")
            raise
