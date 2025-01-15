import torch
from transformers import BertForSequenceClassification, BertTokenizer
from typing import List, Dict
from app.core.config import settings
from app.core.logging import logger

class SentimentModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        try:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            results = []
            for probs in probabilities:
                sentiment_scores = {
                    "negative": probs[0].item(),
                    "neutral": probs[1].item(),
                    "positive": probs[2].item()
                }
                results.append(sentiment_scores)

            return results
        except Exception as e:
            logger.error(f"Error in sentiment prediction: {str(e)}")
            raise

    def train(self, train_data: List[Dict[str, str]], epochs: int = 3):
        try:
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

            for epoch in range(epochs):
                total_loss = 0
                for item in train_data:
                    inputs = self.tokenizer(item['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    labels = torch.tensor([item['label']]).to(self.device)

                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss
                    total_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data)}")

            self.model.eval()
        except Exception as e:
            logger.error(f"Error in sentiment model training: {str(e)}")
            raise

    def save(self, path: str):
        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Sentiment model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving sentiment model: {str(e)}")
            raise

    def load(self, path: str):
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            logger.info(f"Sentiment model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            raise
