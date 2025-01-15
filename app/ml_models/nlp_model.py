from transformers import pipeline
from typing import List, Dict
from app.core.config import settings
from app.core.logging import logger

class NLPModel:
    def __init__(self):
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        self.summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    def named_entity_recognition(self, text: str) -> List[Dict[str, str]]:
        try:
            return self.ner_pipeline(text)
        except Exception as e:
            logger.error(f"Error in named entity recognition: {str(e)}")
            raise

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        try:
            summary = self.summarization_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Error in text summarization: {str(e)}")
            raise

    def question_answering(self, context: str, question: str) -> Dict[str, str]:
        try:
            return self.qa_pipeline(question=question, context=context)
        except Exception as e:
            logger.error(f"Error in question answering: {str(e)}")
            raise

    def save(self, path: str):
        logger.info(f"NLP model saving not implemented. Models are loaded from Hugging Face Hub.")

    def load(self, path: str):
        logger.info(f"NLP model loading not implemented. Models are loaded from Hugging Face Hub.")
