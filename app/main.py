import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.api.audio import router as audio_router
from app.services.classifier import AudioClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("YAMNet 모델 로딩 중...")
    app.state.classifier = AudioClassifier()
    logger.info("YAMNet 모델 로드 완료")
    async with httpx.AsyncClient(timeout=5.0) as client:
        app.state.http_client = client
        yield


app = FastAPI(title="Sound Alert Server", lifespan=lifespan)
app.include_router(audio_router)
