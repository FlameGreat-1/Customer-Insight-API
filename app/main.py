from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import logger, LoggerMiddleware
from app.db.database import create_db_and_tables, close_db_connection
from app.core.security import verify_api_key
from app.services.ml_service import MLService
import time
import uuid
from prometheus_fastapi_instrumentator import Instrumentator
from app.utils.rate_limiter import RateLimiter

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(LoggerMiddleware)

# Rate limiting middleware
rate_limiter = RateLimiter(requests=settings.RATE_LIMIT_PER_MINUTE, window=60)

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not await rate_limiter.is_allowed(request.client.host):
            logger.warning(f"Rate limit exceeded for IP: {request.client.host}")
            return JSONResponse(status_code=429, content={"detail": "Too many requests"})
        return await call_next(request)

app.add_middleware(RateLimitMiddleware)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"Request started: {request_id} - {request.method} {request.url}")
    logger.debug(f"Request headers: {request.headers}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Request completed: {request_id} - Status: {response.status_code} - Time: {process_time:.2f}s")
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.2f}"
    
    return response

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(f"HTTP error occurred: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error occurred: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application")
    create_db_and_tables()
    MLService.initialize_models()
    Instrumentator().instrument(app).expose(app)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the application")
    await close_db_connection()
    MLService.cleanup()

# Custom API documentation endpoints
@app.get("/api/v1/docs", include_in_schema=False)
async def custom_swagger_ui_html(req: Request):
    return get_swagger_ui_html(openapi_url="/api/v1/openapi.json", title=settings.PROJECT_NAME + " - Swagger UI")

@app.get("/api/v1/redoc", include_in_schema=False)
async def redoc_html(req: Request):
    return get_redoc_html(openapi_url="/api/v1/openapi.json", title=settings.PROJECT_NAME + " - ReDoc")

@app.get("/api/v1/openapi.json", include_in_schema=False)
async def get_openapi_endpoint(req: Request):
    return JSONResponse(get_openapi(title=settings.PROJECT_NAME, version=settings.VERSION, routes=app.routes))

# API Key authentication
async def verify_api_key_header(x_api_key: str = Depends(verify_api_key)):
    if not x_api_key:
        raise HTTPException(status_code=403, detail="Could not validate API Key")

# Include API router with API Key authentication
app.include_router(api_router, prefix=settings.API_V1_STR, dependencies=[Depends(verify_api_key_header)])

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
