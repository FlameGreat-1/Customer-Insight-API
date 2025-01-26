import torch
from transformers import BertForSequenceClassification, BertTokenizer
from typing import List, Dict, Union
from app.core.config import settings
from app.core.logging import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from datetime import datetime

class SentimentModel:
    def __init__(self, model_name: str = 'bert-base-uncased', num_labels: int = 3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)
        self.model.eval()
        self.num_labels = num_labels
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
        self.last_trained = None

    def predict(self, texts: Union[str, List[str]]) -> List[Dict[str, float]]:
        try:
            if isinstance(texts, str):
                texts = [texts]

            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            results = []
            for probs in probabilities:
                sentiment_scores = {self.label_map[i]: prob.item() for i, prob in enumerate(probs)}
                results.append(sentiment_scores)

            return results
        except Exception as e:
            logger.error(f"Error in sentiment prediction: {str(e)}")
            raise

    def train(self, train_data: List[Dict[str, Union[str, int]]], validation_data: List[Dict[str, Union[str, int]]] = None, epochs: int = 3, batch_size: int = 16):
        try:
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

            for epoch in range(epochs):
                total_loss = 0
                for i in range(0, len(train_data), batch_size):
                    batch = train_data[i:i+batch_size]
                    texts = [item['text'] for item in batch]
                    labels = torch.tensor([item['label'] for item in batch]).to(self.device)

                    inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss
                    total_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                avg_loss = total_loss / (len(train_data) / batch_size)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

                if validation_data:
                    val_metrics = self.evaluate(validation_data)
                    logger.info(f"Validation Metrics: {val_metrics}")

            self.model.eval()
            self.last_trained = datetime.now()
        except Exception as e:
            logger.error(f"Error in sentiment model training: {str(e)}")
            raise

    def evaluate(self, eval_data: List[Dict[str, Union[str, int]]]) -> Dict[str, float]:
        try:
            self.model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for item in eval_data:
                    inputs = self.tokenizer(item['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.append(item['label'])

            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        except Exception as e:
            logger.error(f"Error in sentiment model evaluation: {str(e)}")
            raise

    def save(self, path: str):
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'tokenizer': self.tokenizer,
                'num_labels': self.num_labels,
                'label_map': self.label_map,
                'last_trained': self.last_trained
            }, path)
            logger.info(f"Sentiment model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving sentiment model: {str(e)}")
            raise

    def load(self, path: str):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.tokenizer = checkpoint['tokenizer']
            self.num_labels = checkpoint['num_labels']
            self.label_map = checkpoint['label_map']
            self.last_trained = checkpoint['last_trained']
            self.model.eval()
            logger.info(f"Sentiment model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Union[int, str, Dict[int, str]]]:
        return {
            "num_labels": self.num_labels,
            "label_map": self.label_map,
            "device": str(self.device),
            "last_trained": str(self.last_trained) if self.last_trained else "Never"
        }

    def __str__(self):
        return f"SentimentModel(num_labels={self.num_labels}, device={self.device}, last_trained={self.last_trained})"

    def __repr__(self):
        return self.__str__()
