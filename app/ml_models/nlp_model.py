from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
from typing import List, Dict, Union
from app.core.config import settings
from app.core.logging import logger
import torch
import os

class NLPModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {
            "ner": {
                "name": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "tokenizer": None,
                "model": None,
                "pipeline": None
            },
            "summarization": {
                "name": "facebook/bart-large-cnn",
                "tokenizer": None,
                "model": None,
                "pipeline": None
            },
            "qa": {
                "name": "distilbert-base-cased-distilled-squad",
                "tokenizer": None,
                "model": None,
                "pipeline": None
            }
        }
        self.load()

    def load(self, path: str = None):
        try:
            for task, info in self.models.items():
                if path and os.path.exists(os.path.join(path, task)):
                    logger.info(f"Loading {task} model from {path}")
                    info["tokenizer"] = AutoTokenizer.from_pretrained(os.path.join(path, task))
                    if task == "ner":
                        info["model"] = AutoModelForTokenClassification.from_pretrained(os.path.join(path, task))
                    elif task == "summarization":
                        info["model"] = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(path, task))
                    elif task == "qa":
                        info["model"] = AutoModelForQuestionAnswering.from_pretrained(os.path.join(path, task))
                else:
                    logger.info(f"Loading {task} model from Hugging Face Hub: {info['name']}")
                    info["tokenizer"] = AutoTokenizer.from_pretrained(info["name"])
                    if task == "ner":
                        info["model"] = AutoModelForTokenClassification.from_pretrained(info["name"])
                    elif task == "summarization":
                        info["model"] = AutoModelForSeq2SeqLM.from_pretrained(info["name"])
                    elif task == "qa":
                        info["model"] = AutoModelForQuestionAnswering.from_pretrained(info["name"])
                
                info["model"].to(self.device)
                info["pipeline"] = pipeline(task, model=info["model"], tokenizer=info["tokenizer"], device=self.device)
            
            logger.info("All NLP models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading NLP models: {str(e)}")
            raise

    def named_entity_recognition(self, text: Union[str, List[str]]) -> List[Dict[str, str]]:
        try:
            return self.models["ner"]["pipeline"](text)
        except Exception as e:
            logger.error(f"Error in named entity recognition: {str(e)}")
            raise

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        try:
            summary = self.models["summarization"]["pipeline"](text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Error in text summarization: {str(e)}")
            raise

    def question_answering(self, context: str, question: str) -> Dict[str, str]:
        try:
            return self.models["qa"]["pipeline"](question=question, context=context)
        except Exception as e:
            logger.error(f"Error in question answering: {str(e)}")
            raise

    def save(self, path: str):
        try:
            for task, info in self.models.items():
                task_path = os.path.join(path, task)
                if not os.path.exists(task_path):
                    os.makedirs(task_path)
                logger.info(f"Saving {task} model to {task_path}")
                info["tokenizer"].save_pretrained(task_path)
                info["model"].save_pretrained(task_path)
            logger.info("All NLP models saved successfully")
        except Exception as e:
            logger.error(f"Error saving NLP models: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Dict[str, str]]:
        return {
            task: {
                "model_name": info["name"],
                "device": self.device,
                "vocab_size": len(info["tokenizer"]) if info["tokenizer"] else None
            }
            for task, info in self.models.items()
        }

    def __str__(self):
        return f"NLPModel(device={self.device}, models={list(self.models.keys())})"

    def __repr__(self):
        return self.__str__()
