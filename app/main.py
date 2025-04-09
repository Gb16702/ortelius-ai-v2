import logfire

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.v1.endpoints import chat
from app.core.config import settings

logfire.configure(
    token=settings.LOGFIRE_TOKEN,
    service_name="ai-chatbot",
    environment=settings.APP_ENV
)

logfire.info("Logfire initialization complete", app_environment=settings.APP_ENV)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logfire.info(f"Application {app.title} starting up")

    yield

    logfire.info(f"Application {app.title} shutting down")

app = FastAPI(
    title="AI Chatbot",
    lifespan=lifespan
)

logfire.instrument_fastapi(app, capture_headers=True)
logfire.instrument_httpx(capture_all=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1")

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "Running"
    }
