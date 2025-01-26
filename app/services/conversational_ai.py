# app/services/conversational_ai.py

import asyncio
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.customer import Customer
from app.models.conversation import Conversation, Message
from app.schemas.conversational_ai import ConversationResponse, IntentAnalysisResult
from app.core.logging import logger
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from datasets import Dataset
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download('punkt')
nltk.download('stopwords')

class ConversationalAIService:
    def __init__(self, db: Session):
        self.db = db
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForSequenceClassification.from_pretrained("gpt2")
        self.intent_classifier = pipeline("text-classification", model="facebook/bart-large-mnli")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.stop_words = set(stopwords.words('english'))

    async def generate_response(self, message: str, conversation_id: str = None, customer_id: int = None) -> ConversationResponse:
        try:
            # Preprocess the message
            preprocessed_message = await self._preprocess_text(message)

            # Analyze intent
            intent = await self.analyze_intent(preprocessed_message)

            # Generate response based on intent
            response = await self._generate_response_for_intent(preprocessed_message, intent)

            # Analyze sentiment of the response
            sentiment = await self._analyze_sentiment(response)

            # Store conversation in database
            conversation = await self._store_conversation(message, response, conversation_id, intent, sentiment, customer_id)

            # Update customer profile
            if customer_id:
                await self._update_customer_profile(customer_id, intent, sentiment)

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

    async def _preprocess_text(self, text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Join the tokens back into a string
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text

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
                response = f"Hello! Welcome to our service. {response}"
            elif intent.intent == "farewell":
                response = f"Thank you for using our service. {response} Have a great day!"
            elif intent.intent == "product_inquiry":
                response = f"Regarding your product inquiry: {response} Would you like more detailed information?"
            elif intent.intent == "complaint":
                response = f"I'm sorry to hear that you're experiencing issues. {response} Let me help you resolve this."
            elif intent.intent == "general_inquiry":
                response = f"Thank you for your inquiry. {response} Is there anything else I can help you with?"

            return response
        except Exception as e:
            logger.error(f"Error in generating response for intent: {str(e)}")
            raise

    async def analyze_intent(self, message: str) -> IntentAnalysisResult:
        try:
            intents = ["greeting", "farewell", "product_inquiry", "complaint", "general_inquiry"]
            results = self.intent_classifier(message, candidate_labels=intents, multi_label=False)
            
            return IntentAnalysisResult(
                intent=results[0]['label'],
                confidence=float(results[0]['score'])
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

    async def _store_conversation(self, user_message: str, ai_response: str, conversation_id: str, intent: IntentAnalysisResult, sentiment: str, customer_id: int = None) -> Conversation:
        try:
            async with self.db.begin():
                if conversation_id:
                    conversation = await self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
                    if not conversation:
                        raise ValueError(f"Conversation with id {conversation_id} not found")
                else:
                    conversation = Conversation(started_at=datetime.utcnow(), customer_id=customer_id)
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

            return conversation
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")
            raise

    async def _update_customer_profile(self, customer_id: int, intent: IntentAnalysisResult, sentiment: str):
        try:
            async with self.db.begin():
                customer = await self.db.query(Customer).filter(Customer.id == customer_id).first()
                if not customer:
                    raise ValueError(f"Customer with id {customer_id} not found")

                # Update customer profile based on intent and sentiment
                if intent.intent == "product_inquiry":
                    customer.product_interest += 1
                elif intent.intent == "complaint":
                    customer.complaint_count += 1

                if sentiment == "POSITIVE":
                    customer.positive_interaction_count += 1
                elif sentiment == "NEGATIVE":
                    customer.negative_interaction_count += 1

                customer.last_interaction_date = datetime.utcnow()

                self.db.add(customer)
        except Exception as e:
            logger.error(f"Error updating customer profile: {str(e)}")
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

            # Prepare dataset
            dataset = Dataset.from_dict({
                "input": [item["input"] for item in training_data],
                "output": [item["output"] for item in training_data]
            })

            # Split dataset
            train_dataset, eval_dataset = train_test_split(dataset, test_size=0.2)

            # Define training arguments
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=64,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir="./logs",
            )

            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
            )

            # Train the model
            trainer.train()

            # Save the model
            self.model.save_pretrained("./trained_model")
            self.tokenizer.save_pretrained("./trained_model")

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

            # Calculate response time
            response_times = await self.db.query(
                (Message.timestamp - func.lag(Message.timestamp).over(partition_by=Message.conversation_id, order_by=Message.timestamp)).label('response_time')
            ).filter(Message.is_user == False).all()
            avg_response_time = sum([rt.response_time.total_seconds() for rt in response_times if rt.response_time]) / len(response_times) if response_times else 0

            return {
                "total_conversations": total_conversations,
                "avg_messages_per_conversation": avg_messages_per_conversation,
                "positive_sentiment_ratio": sentiment_ratio,
                "intent_classification_accuracy": float(intent_accuracy),
                "avg_response_time_seconds": avg_response_time
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise

    async def get_popular_intents(self, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            popular_intents = await self.db.query(
                Message.intent,
                func.count(Message.id).label('count')
            ).filter(
                Message.is_user == True,
                Message.intent.isnot(None)
            ).group_by(
                Message.intent
            ).order_by(
                func.count(Message.id).desc()
            ).limit(limit).all()

            return [{"intent": intent, "count": count} for intent, count in popular_intents]
        except Exception as e:
            logger.error(f"Error retrieving popular intents: {str(e)}")
            raise

    async def get_sentiment_distribution(self) -> Dict[str, float]:
        try:
            total_messages = await self.db.query(func.count(Message.id)).scalar()
            sentiment_counts = await self.db.query(
                Message.sentiment,
                func.count(Message.id).label('count')
            ).group_by(Message.sentiment).all()

            distribution = {sentiment: count / total_messages for sentiment, count in sentiment_counts}
            return distribution
        except Exception as e:
            logger.error(f"Error calculating sentiment distribution: {str(e)}")
            raise

    async def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        try:
            conversation = await self.db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conversation:
                raise ValueError(f"Conversation with id {conversation_id} not found")

            messages = await self.db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp).all()

            summary = {
                "conversation_id": conversation_id,
                "start_time": conversation.started_at,
                "end_time": messages[-1].timestamp if messages else None,
                "duration": (messages[-1].timestamp - conversation.started_at).total_seconds() if messages else 0,
                "total_messages": len(messages),
                "user_messages": sum(1 for m in messages if m.is_user),
                "ai_messages": sum(1 for m in messages if not m.is_user),
                "intents": list(set(m.intent for m in messages if m.intent)),
                "sentiments": list(set(m.sentiment for m in messages if m.sentiment)),
                "average_intent_confidence": sum(m.intent_confidence for m in messages if m.intent_confidence) / sum(1 for m in messages if m.intent_confidence) if any(m.intent_confidence for m in messages) else 0
            }

            return summary
        except Exception as e:
            logger.error(f"Error generating conversation summary: {str(e)}")
            raise
