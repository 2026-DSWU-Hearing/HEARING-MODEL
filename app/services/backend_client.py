import logging
from datetime import datetime, timezone, timedelta

import httpx
import jwt

from app.core.config import settings

logger = logging.getLogger(__name__)


def _make_token() -> str:
    payload = {
        "sub": "1",
        "source": "ai-server",
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


async def send_detection(client: httpx.AsyncClient, result: dict) -> None:
    if not all([settings.BACKEND_URL, settings.JWT_SECRET, settings.DEVICE_ID]):
        logger.warning("백엔드 환경변수 미설정 — 전송 건너뜀 (BACKEND_URL/JWT_SECRET/DEVICE_ID 확인)")
        return
    try:
        body = {
            "sound_category": result["category"],
            "sound_name": result["block"],
            "confidence": result["score"],
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("백엔드 전송 데이터: %s", body)
        resp = await client.post(
            f"{settings.BACKEND_URL}/devices/{settings.DEVICE_ID}/detections",
            json=body,
            headers={"Authorization": f"Bearer {_make_token()}"},
        )
        resp.raise_for_status()
        logger.info("백엔드 전송 완료: %s (%d)", body["sound_name"], resp.status_code)
    except Exception as e:
        logger.error("백엔드 전송 실패: %s", e)
