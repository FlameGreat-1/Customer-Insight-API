from pydantic import BaseSettings, AnyHttpUrl, EmailStr, validator, SecretStr
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
import secrets
from datetime import timedelta

class Settings(BaseSettings):
    # API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI-Powered Customer Insight and Engagement API"
    PROJECT_DESCRIPTION: str = "An advanced API for customer analytics and engagement using AI and ML technologies."
    VERSION: str = "1.0.0"
    
    # SECURITY
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30  # 30 days
    ALGORITHM: str = "HS256"
    SECURITY_BCRYPT_ROUNDS: int = 12
    JWT_TOKEN_PREFIX: str = "Bearer"
    
    # DATABASE
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: SecretStr
    POSTGRES_DB: str
    SQLALCHEMY_DATABASE_URI: Optional[str] = None
    SQLALCHEMY_POOL_SIZE: int = 5
    SQLALCHEMY_MAX_OVERFLOW: int = 10
    SQLALCHEMY_POOL_RECYCLE: int = 3600

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        if isinstance(v, str):
            return v
        return f"postgresql://{values.get('POSTGRES_USER')}:{values.get('POSTGRES_PASSWORD').get_secret_value()}@{values.get('POSTGRES_SERVER')}/{values.get('POSTGRES_DB')}"

    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    ALLOWED_HOSTS: List[str] = ["*"]

    # LOGGING
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = Path("/var/log/customer_insight_api.log")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
    LOG_BACKUP_COUNT: int = 5

    # EMAIL
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[SecretStr] = None
    EMAILS_FROM_EMAIL: Optional[EmailStr] = None
    EMAILS_FROM_NAME: Optional[str] = None
    EMAIL_RESET_TOKEN_EXPIRE_HOURS: int = 48
    EMAIL_TEMPLATES_DIR: str = "/app/email-templates/build"

    # ML MODEL PATHS
    SENTIMENT_MODEL_PATH: Path = Path("/app/ml_models/sentiment_model.pkl")
    SEGMENTATION_MODEL_PATH: Path = Path("/app/ml_models/segmentation_model.pkl")
    RECOMMENDATION_MODEL_PATH: Path = Path("/app/ml_models/recommendation_model.pkl")
    NLP_MODEL_PATH: Path = Path("/app/ml_models/nlp_model.pkl")
    EMOTION_DETECTION_MODEL_PATH: Path = Path("/app/ml_models/emotion_detection_model.pkl")
    CHURN_PREDICTION_MODEL_PATH: Path = Path("/app/ml_models/churn_prediction_model.pkl")
    LTV_PREDICTION_MODEL_PATH: Path = Path("/app/ml_models/ltv_prediction_model.pkl")

    # API RATE LIMITING
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_BURST: int = 20

    # CACHING
    REDIS_URL: str = "redis://redis:6379/0"
    CACHE_EXPIRATION_SECONDS: int = 3600  # 1 hour
    CACHE_BACKEND: str = "redis"

    # EXTERNAL SERVICES
    OPENAI_API_KEY: Optional[SecretStr] = None
    AWS_ACCESS_KEY_ID: Optional[SecretStr] = None
    AWS_SECRET_ACCESS_KEY: Optional[SecretStr] = None
    AWS_REGION: Optional[str] = None
    AWS_S3_BUCKET_NAME: Optional[str] = None
    STRIPE_API_KEY: Optional[SecretStr] = None
    TWILIO_ACCOUNT_SID: Optional[SecretStr] = None
    TWILIO_AUTH_TOKEN: Optional[SecretStr] = None
    SENDGRID_API_KEY: Optional[SecretStr] = None

    # FEATURE FLAGS
    ENABLE_DOCS: bool = True
    ENABLE_REAL_TIME_ANALYTICS: bool = True
    ENABLE_AI_CHATBOT: bool = True

    # PERFORMANCE
    WORKER_CONCURRENCY: int = 4
    ASYNC_TASKS_ENABLED: bool = True
    BACKGROUND_TASK_MAX_ATTEMPTS: int = 3
    BACKGROUND_TASK_RETRY_DELAY: int = 60  # seconds

    # DATA RETENTION
    DATA_RETENTION_DAYS: int = 365  # 1 year
    ANONYMIZE_INACTIVE_USERS_AFTER_DAYS: int = 730  # 2 years

    # MONITORING
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_ENABLED: bool = True

    # INTERNATIONALIZATION
    DEFAULT_LANGUAGE: str = "en"
    SUPPORTED_LANGUAGES: List[str] = ["en", "es", "fr", "de", "zh"]

    # TESTING
    TESTING: bool = False
    TEST_DATABASE_URL: Optional[str] = None

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = 'utf-8'

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    def get_jwt_settings(self) -> Dict[str, Any]:
        return {
            "SECRET_KEY": self.SECRET_KEY,
            "ALGORITHM": self.ALGORITHM,
            "ACCESS_TOKEN_EXPIRE_MINUTES": self.ACCESS_TOKEN_EXPIRE_MINUTES,
            "REFRESH_TOKEN_EXPIRE_MINUTES": self.REFRESH_TOKEN_EXPIRE_MINUTES,
            "JWT_TOKEN_PREFIX": self.JWT_TOKEN_PREFIX,
        }

    def get_database_settings(self) -> Dict[str, Any]:
        return {
            "SQLALCHEMY_DATABASE_URI": self.SQLALCHEMY_DATABASE_URI,
            "SQLALCHEMY_POOL_SIZE": self.SQLALCHEMY_POOL_SIZE,
            "SQLALCHEMY_MAX_OVERFLOW": self.SQLALCHEMY_MAX_OVERFLOW,
            "SQLALCHEMY_POOL_RECYCLE": self.SQLALCHEMY_POOL_RECYCLE,
        }

    def get_email_settings(self) -> Dict[str, Any]:
        return {
            "SMTP_TLS": self.SMTP_TLS,
            "SMTP_PORT": self.SMTP_PORT,
            "SMTP_HOST": self.SMTP_HOST,
            "SMTP_USER": self.SMTP_USER,
            "SMTP_PASSWORD": self.SMTP_PASSWORD.get_secret_value() if self.SMTP_PASSWORD else None,
            "EMAILS_FROM_EMAIL": self.EMAILS_FROM_EMAIL,
            "EMAILS_FROM_NAME": self.EMAILS_FROM_NAME,
            "EMAIL_RESET_TOKEN_EXPIRE_HOURS": self.EMAIL_RESET_TOKEN_EXPIRE_HOURS,
            "EMAIL_TEMPLATES_DIR": self.EMAIL_TEMPLATES_DIR,
        }

settings = Settings()
