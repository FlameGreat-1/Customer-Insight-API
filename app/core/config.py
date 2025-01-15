from pydantic import BaseSettings, AnyHttpUrl, EmailStr, validator
from typing import List, Union, Optional
from pathlib import Path
import secrets

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI-Powered Customer Insight and Engagement API"
    PROJECT_DESCRIPTION: str = "An advanced API for customer analytics and engagement using AI and ML technologies."
    VERSION: str = "1.0.0"
    
    # SECURITY
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    ALGORITHM: str = "HS256"
    
    # DATABASE
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    SQLALCHEMY_DATABASE_URI: Optional[str] = None

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
        return f"postgresql://{values.get('POSTGRES_USER')}:{values.get('POSTGRES_PASSWORD')}@{values.get('POSTGRES_SERVER')}/{values.get('POSTGRES_DB')}"

    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    ALLOWED_HOSTS: List[str] = ["*"]

    # LOGGING
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = Path("/var/log/customer_insight_api.log")

    # EMAIL
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[EmailStr] = None
    EMAILS_FROM_NAME: Optional[str] = None

    # ML MODEL PATHS
    SENTIMENT_MODEL_PATH: Path = Path("/app/ml_models/sentiment_model.pkl")
    SEGMENTATION_MODEL_PATH: Path = Path("/app/ml_models/segmentation_model.pkl")
    RECOMMENDATION_MODEL_PATH: Path = Path("/app/ml_models/recommendation_model.pkl")
    NLP_MODEL_PATH: Path = Path("/app/ml_models/nlp_model.pkl")
    EMOTION_DETECTION_MODEL_PATH: Path = Path("/app/ml_models/emotion_detection_model.pkl")

    # API RATE LIMITING
    RATE_LIMIT_PER_MINUTE: int = 100

    # CACHING
    REDIS_URL: str = "redis://redis:6379/0"
    CACHE_EXPIRATION_SECONDS: int = 3600  # 1 hour

    # EXTERNAL SERVICES
    OPENAI_API_KEY: Optional[str] = None
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: Optional[str] = None

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
