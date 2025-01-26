import logging
from logging.handlers import RotatingFileHandler, SMTPHandler
import sys
from pathlib import Path
from app.core.config import settings
import json
from pythonjsonlogger import jsonlogger
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

# Create a custom logger
logger = logging.getLogger(settings.PROJECT_NAME)

# Set level
logger.setLevel(settings.LOG_LEVEL)

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
file_handler = RotatingFileHandler(
    settings.LOG_FILE,
    maxBytes=settings.LOG_MAX_BYTES,
    backupCount=settings.LOG_BACKUP_COUNT
)

# Create formatters and add it to handlers
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            log_record['timestamp'] = record.created
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname

json_formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
console_handler.setFormatter(json_formatter)
file_handler.setFormatter(json_formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Add email handler for error logs
if all([settings.SMTP_HOST, settings.SMTP_PORT, settings.EMAILS_FROM_EMAIL]):
    mail_handler = SMTPHandler(
        mailhost=(settings.SMTP_HOST, settings.SMTP_PORT),
        fromaddr=settings.EMAILS_FROM_EMAIL,
        toaddrs=settings.ADMINS,
        subject=f"{settings.PROJECT_NAME} - Application Error",
        credentials=(settings.SMTP_USER, settings.SMTP_PASSWORD.get_secret_value() if settings.SMTP_PASSWORD else None),
        secure=() if settings.SMTP_TLS else None,
    )
    mail_handler.setLevel(logging.ERROR)
    mail_handler.setFormatter(json_formatter)
    logger.addHandler(mail_handler)

# Initialize Sentry
if settings.SENTRY_DSN:
    sentry_logging = LoggingIntegration(
        level=logging.INFO,        # Capture info and above as breadcrumbs
        event_level=logging.ERROR  # Send errors as events
    )
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        integrations=[sentry_logging],
        environment=settings.ENVIRONMENT,
        release=settings.VERSION
    )

# Function to get logger
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{settings.PROJECT_NAME}.{name}")

# Monkey patch the logging module to use our custom logger
logging.getLogger = get_logger

class LoggerMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            logger.info(f"Request: {scope['method']} {scope['path']}")
        await self.app(scope, receive, send)

def log_request(request):
    logger.info(f"Request: {request.method} {request.url}")
    logger.debug(f"Headers: {json.dumps(dict(request.headers))}")
    logger.debug(f"Query Params: {json.dumps(dict(request.query_params))}")

def log_response(response):
    logger.info(f"Response: Status {response.status_code}")
    logger.debug(f"Headers: {json.dumps(dict(response.headers))}")

def setup_logging():
    # Ensure log directory exists
    log_dir = Path(settings.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging for other libraries
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    # Log unhandled exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

setup_logging()
