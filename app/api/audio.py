import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.services.backend_client import send_detection

logger = logging.getLogger(__name__)

router = APIRouter()
_last_alert_time: dict[str, float] = {}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("ESP32 연결됨")
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            result = websocket.app.state.classifier.classify(audio_bytes)
            if result is None:
                continue

            block = result["block"]
            category = result["category"]
            score = result["score"]
            now = time.time()

            logger.info("소리 분석: %s - %s (%.1f%%)", category, block, score * 100)

            if score >= settings.ALERT_THRESHOLD:
                if now - _last_alert_time.get(block, 0) > settings.COOLDOWN_SECONDS:
                    await websocket.send_text(f"ALERT:{block}")
                    logger.warning("알림 전송: %s (%.1f%%)", block, score * 100)
                    _last_alert_time[block] = now
                    await send_detection(websocket.app.state.http_client, result)

    except WebSocketDisconnect:
        logger.info("ESP32 연결 해제")
    except Exception as e:
        logger.exception("WebSocket 처리 중 에러: %s", e)
