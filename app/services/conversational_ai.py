# app/services/conversational_ai.py

import asyncio
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.customer import Customer
from app.models.conversation import Conversation, Message
from app.schemas.conversational_ai import ConversationResponse, IntentAnalysisResult
from app.core.logging import logger
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from datetime import datetime, timedelta

class ConversationalAIService:
    def __init__(self, db: Session):
        self.db = db
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForSequenceClassification.from_pretrained("gpt2")
        self.intent_classifier = pipeline("text-classification", model="facebook/bart-large-mnli")
        self.sentiment_analyzer = pipeline("sentiment-analysis")

    async def generate_response(self, message: str, conversation_id: str = None) -> ConversationResponse:
        try:
            # Analyze intent
            intent = await self.analyze_intent(message)

            # Generate response based on intent
            response = await self._generate_response_for_intent(message, intent)

            # Analyze sentiment of the response
            sentiment = await self._analyze_sentiment(response)

            # Store conversation in database
            conversation = await self._store_conversation(message, response, conversation_id, intent, sentiment)

            return ConversationResponse(
                message=response,
                conversation_id=conversation.id,
                intent=intent.intent,
                confidence=intent.confidence,
                sentiment=sentiment
            )
        except Exception as e:
            logger.error(f"Error in generating response: {str(e)}")
            raise

    async def _generate_response_for_intent(self, message: str, intent: IntentAnalysisResult) -> str:
        try:
            input_ids = self.tokenizer.encode(message, return_tensors="pt")
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
            pad_token_id = self.tokenizer.eos_token_id

            output = self.model.generate(
                input_ids,
                max_length=100,
                num_return_sequences=1,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )

            response = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # Adjust response based on intent
            if intent.intent == "greeting":
                response = f"Hello! {response}"
            elif intent.intent == "farewell":
                response = f"{response} Goodbye!"
            elif intent.intent == "product_inquiry":
                response = f"Regarding your product inquiry: {response}"
            elif intent.intent == "complaint":
                response = f"I'm sorry to hear that. {response}"

            return response
        except Exception as e:
            logger.error(f"Error in generating response for intent: {str(e)}")
            raise

    async def analyze_intent(self, message: str) -> IntentAnalysisResult:
        try:
            intents = ["greeting", "farewell", "product_inquiry", "complaint", "general_inquiry"]
            results = self.intent_classifier(message, candidate_labels=intents, multi_label=False)
            
            return IntentAnalysisResult(
                intent=results['labels'][0],
                confidence=results['scores'][0]
            )
        except Exception as e:
            logger.error(f"Error in intent analysis: {str(e)}")
            raise

    async def _analyze_sentiment(self, message: str) -> str:
        try:
            result = self.sentiment_analyzer(message)[0]
            return result['label']
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            raise

    async def _store_conversation(self, user_message: str, ai_response: str, conversation_id: str, intent: IntentAnalysisResult, sentiment: str) -> Conversation:
        try:
            if conversation_id:
                conversation = await self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
                if not conversation:
                    raise ValueError(f"Conversation with id {conversation_id} not found")
            else:
                conversation = Conversation(started_at=datetime.utcnow())
                self.db.add(conversation)
                await self.db.flush()

            user_message = Message(
                conversation_id=conversation.id,
                content=user_message,
                is_user=True,
                intent=intent.intent,
                intent_confidence=intent.confidence,
                sentiment=sentiment
            )
            ai_message = Message(
                conversation_id=conversation.id,
                content=ai_response,
                is_user=False,
                sentiment=sentiment
            )

            self.db.add(user_message)
            self.db.add(ai_message)
            await self.db.commit()

            return conversation
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")
            await self.db.rollback()
            raise

    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        try:
            messages = await self.db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp).all()
            return [
                {
                    "content": message.content,
                    "is_user": message.is_user,
                    "timestamp": message.timestamp,
                    "intent": message.intent,
                    "intent_confidence": message.intent_confidence,
                    "sentiment": message.sentiment
                }
                for message in messages
            ]
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            raise

    async def train_model(self):
        try:
            logger.info("Starting model training...")

            # Fetch training data from database
            conversations = await self.db.query(Conversation).all()
            training_data = []
            for conversation in conversations:
                messages = await self.db.query(Message).filter(Message.conversation_id == conversation.id).order_by(Message.timestamp).all()
                for i in range(0, len(messages) - 1, 2):
                    if messages[i].is_user and not messages[i+1].is_user:
                        training_data.append({
                            "input": messages[i].content,
                            "output": messages[i+1].content
                        })

            # Train model (this is a placeholder, actual implementation would depend on the specific model being used)
            # In a real-world scenario, you might use a more sophisticated training process
            # or even outsource the training to a dedicated machine learning pipeline
            for data in training_data:
                input_ids = self.tokenizer.encode(data["input"], return_tensors="pt")
                labels = self.tokenizer.encode(data["output"], return_tensors="pt")
                outputs = self.model(input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()

            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    async def get_performance_metrics(self) -> Dict[str, float]:
        try:
            total_conversations = await self.db.query(func.count(Conversation.id)).scalar()
            total_messages = await self.db.query(func.count(Message.id)).scalar()
            avg_messages_per_conversation = total_messages / total_conversations if total_conversations > 0 else 0

            positive_sentiments = await self.db.query(func.count(Message.id)).filter(Message.sentiment == "POSITIVE").scalar()
            sentiment_ratio = positive_sentiments / total_messages if total_messages > 0 else 0

            intent_accuracy = await self.db.query(func.avg(Message.intent_confidence)).filter(Message.is_user == True).scalar() or 0

            return {
                "total_conversations": total_conversations,
                "avg_messages_per_conversation": avg_messages_per_conversation,
                "positive_sentiment_ratio": sentiment_ratio,
                "intent_classification_accuracy": float(intent_accuracy)
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise

