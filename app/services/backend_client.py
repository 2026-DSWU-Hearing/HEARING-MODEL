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

    return jwt.encode(
        payload,
        settings.JWT_SECRET,
        algorithm="HS256",
    )


async def send_detection(
    client: httpx.AsyncClient,
    result: dict | None,
) -> None:
    if not all([
        settings.BACKEND_URL,
        settings.JWT_SECRET,
        settings.DEVICE_ID,
    ]):
        logger.warning(
            "백엔드 환경변수 미설정 — 전송 건너뜀"
        )
        return

    if not result:
        logger.warning("전송할 감지 결과 없음")
        return

    top_sounds = result.get("top_sounds", [])

    if not top_sounds:
        logger.warning("top_sounds 결과 없음")
        return

    # 백엔드에는 가장 높은 결과 1개 전송
    top = top_sounds[0]

    body = {
        "sound_category": top["category"],
        "sound_name": top["block"],
        "confidence": top["score"],

        # ESP32 패킷에서 가져온 방향
        "direction": result.get(
            "direction",
            "UNKNOWN",
        ),

        "detected_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    try:
        logger.info(
            "백엔드 전송 데이터: %s",
            body,
        )

        response = await client.post(
            (
                f"{settings.BACKEND_URL}"
                f"/devices/{settings.DEVICE_ID}/detections"
            ),
            json=body,
            headers={
                "Authorization": f"Bearer {_make_token()}"
            },
        )

        response.raise_for_status()

        logger.info(
            "백엔드 전송 완료: "
            "sound=%s, direction=%s, status=%d",
            body["sound_name"],
            body["direction"],
            response.status_code,
        )

    except httpx.HTTPStatusError as error:
        logger.error(
            "백엔드 응답 오류: status=%d, response=%s",
            error.response.status_code,
            error.response.text,
        )

    except httpx.RequestError as error:
        logger.error(
            "백엔드 연결 실패: %s",
            error,
        )

    except Exception as error:
        logger.exception(
            "백엔드 전송 실패: %s",
            error,
        )