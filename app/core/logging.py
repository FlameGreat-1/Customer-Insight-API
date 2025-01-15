import logging
from logging.handlers import RotatingFileHandler
import sys
from app.core.config import settings

# Create a custom logger
logger = logging.getLogger(settings.PROJECT_NAME)

# Set level
logger.setLevel(settings.LOG_LEVEL)

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
file_handler = RotatingFileHandler(
    settings.LOG_FILE,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(log_format)
file_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Function to get logger
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{settings.PROJECT_NAME}.{name}")

# Monkey patch the logging module to use our custom logger
logging.getLogger = get_logger
