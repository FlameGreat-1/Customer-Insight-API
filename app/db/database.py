from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.core.logging import logger
import time
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager

SQLALCHEMY_DATABASE_URL = settings.SQLALCHEMY_DATABASE_URI

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=20,
    max_overflow=0,
    pool_timeout=300,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise

@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        db.close()

def check_db_connection():
    max_retries = 5
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            with engine.connect() as connection:
                connection.execute("SELECT 1")
            logger.info("Database connection successful")
            return True
        except SQLAlchemyError as e:
            logger.warning(f"Database connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error("Failed to establish database connection after multiple attempts")
                raise

def create_db_and_tables():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database and tables created successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error creating database and tables: {str(e)}")
        raise

def drop_db_and_tables():
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database and tables dropped successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error dropping database and tables: {str(e)}")
        raise
