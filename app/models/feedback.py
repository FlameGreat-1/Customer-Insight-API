from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum, Float, Text, Boolean, Index
from typing import List, Tuple
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import JSONB
from app.db.base_class import Base
import enum
from datetime import datetime
import re
import spacy
import phonenumbers
from faker import Faker
from cryptography.fernet import Fernet

class FeedbackType(enum.Enum):
    PRODUCT_REVIEW = "product_review"
    CUSTOMER_SUPPORT = "customer_support"
    GENERAL = "general"
    SURVEY = "survey"

class SentimentType(enum.Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=True)
    feedback_type = Column(Enum(FeedbackType))
    content = Column(Text)
    rating = Column(Float)
    sentiment = Column(Enum(SentimentType))
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(String)
    is_public = Column(Boolean, default=False)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    metadata = Column(JSONB)

    # Relationships
    customer = relationship("Customer", back_populates="feedback")
    product = relationship("Product", back_populates="feedback")
    resolver = relationship("User", back_populates="resolved_feedback")

    # Indexes
    __table_args__ = (
        Index('idx_feedback_customer_timestamp', 'customer_id', 'timestamp'),
        Index('idx_feedback_product_sentiment', 'product_id', 'sentiment'),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timestamp = datetime.utcnow()
        self.nlp = spacy.load("en_core_web_sm")
        self.faker = Faker()
        self.cipher_suite = Fernet(Fernet.generate_key())

    def __repr__(self):
        return f"<Feedback {self.id} - {self.feedback_type.value}>"

    @property
    def feedback_summary(self) -> str:
        return f"{self.feedback_type.value} - {self.sentiment.value}"

    @validates('rating')
    def validate_rating(self, key, rating):
        if not 0 <= rating <= 5:
            raise ValueError("Rating must be between 0 and 5")
        return rating

    def resolve(self, resolver_id: int) -> None:
        self.is_resolved = True
        self.resolved_at = datetime.utcnow()
        self.resolved_by = resolver_id

    def update_sentiment(self, new_sentiment: SentimentType) -> None:
        self.sentiment = new_sentiment

    def update_metadata(self, new_metadata: dict) -> None:
        if not self.metadata:
            self.metadata = {}
        self.metadata.update(new_metadata)

    def make_public(self) -> None:
        self.is_public = True

    def make_private(self) -> None:
        self.is_public = False

    def get_resolution_time(self) -> float:
        if self.is_resolved and self.resolved_at:
            return (self.resolved_at - self.timestamp).total_seconds() / 3600  # in hours
        return None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def get_sentiment_score(self) -> float:
        sentiment_scores = {
            SentimentType.NEGATIVE: -1.0,
            SentimentType.NEUTRAL: 0.0,
            SentimentType.POSITIVE: 1.0
        }
        return sentiment_scores[self.sentiment]

    def is_recent(self, days: int = 30) -> bool:
        return (datetime.utcnow() - self.timestamp).days <= days

    def get_word_count(self) -> int:
        return len(re.findall(r'\w+', self.content))


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timestamp = datetime.utcnow()
        self.nlp = spacy.load("en_core_web_sm")
        self.faker = Faker()
        self.cipher_suite = Fernet(Fernet.generate_key())

    def anonymize(self) -> None:
        """
        Perform advanced anonymization on the feedback content and metadata.
        This method uses NLP for named entity recognition, handles various
        types of PII, and uses encryption for sensitive data.
        """
        # Anonymize content
        doc = self.nlp(self.content)
        anonymized_content = self.content

        # Named Entity Recognition and replacement
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
                replacement = f"[{ent.label_}]"
                anonymized_content = anonymized_content.replace(str(ent), replacement)

        # Email anonymization
        anonymized_content = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            anonymized_content
        )

        # Phone number anonymization
        anonymized_content = self._anonymize_phone_numbers(anonymized_content)

        # Credit card number anonymization
        anonymized_content = re.sub(
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            '[CREDIT_CARD]',
            anonymized_content
        )

        # Social Security Number anonymization
        anonymized_content = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            '[SSN]',
            anonymized_content
        )

        self.content = anonymized_content

        # Anonymize metadata
        if self.metadata:
            self._anonymize_metadata()

    def _anonymize_phone_numbers(self, text: str) -> str:
        """
        Anonymize phone numbers using the phonenumbers library for better accuracy.
        """
        for match in phonenumbers.PhoneNumberMatcher(text, "US"):
            text = text.replace(match.raw_string, "[PHONE]")
        return text

    def _anonymize_metadata(self) -> None:
        """
        Anonymize sensitive information in metadata.
        """
        sensitive_keys = ['address', 'birth_date', 'ip_address', 'personal_info']
        for key in sensitive_keys:
            if key in self.metadata:
                if key == 'address':
                    self.metadata[key] = self.faker.address()
                elif key == 'birth_date':
                    self.metadata[key] = self.faker.date_of_birth().isoformat()
                elif key == 'ip_address':
                    self.metadata[key] = self.faker.ipv4()
                elif key == 'personal_info':
                    # Encrypt personal info instead of deleting
                    encrypted_data = self.cipher_suite.encrypt(str(self.metadata[key]).encode())
                    self.metadata[key] = encrypted_data.decode()

    def decrypt_personal_info(self) -> dict:
        """
        Decrypt the encrypted personal information in metadata.
        """
        if 'personal_info' in self.metadata:
            decrypted_data = self.cipher_suite.decrypt(self.metadata['personal_info'].encode()).decode()
            return eval(decrypted_data)
        return {}

    @staticmethod
    def _find_dates(text: str) -> List[Tuple[str, str]]:
        """
        Find and return dates in the text along with their positions.
        """
        date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4})\b'
        return [(m.group(), m.span()) for m in re.finditer(date_pattern, text, re.IGNORECASE)]

    def redact_dates(self) -> None:
        """
        Redact dates from the feedback content.
        """
        dates = self._find_dates(self.content)
        for date, (start, end) in reversed(dates):
            self.content = self.content[:start] + '[DATE]' + self.content[end:]

