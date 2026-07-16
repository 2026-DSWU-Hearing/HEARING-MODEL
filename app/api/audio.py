import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.direction import parse_audio_packet
from app.services.backend_client import send_detection

logger = logging.getLogger(__name__)

router = APIRouter()

# 같은 소리의 반복 알림을 막기 위한 마지막 알림 시간
_last_alert_time: dict[str, float] = {}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("ESP32 연결됨")

    try:
        while True:
            # ESP32 바이너리 패킷 수신
            packet = await websocket.receive_bytes()

            try:
                # 앞 4바이트에서 방향을 추출하고,
                # 나머지 순수 PCM만 분리
                (
                    direction_value,
                    direction_name,
                    pcm_audio,
                ) = parse_audio_packet(packet)

            except ValueError as error:
                logger.warning("잘못된 오디오 패킷: %s", error)

                await websocket.send_json({
                    "status": "error",
                    "message": str(error),
                })
                continue

            logger.info(
                "오디오 패킷 수신: direction=%s(%d), "
                "packet=%d bytes, pcm=%d bytes",
                direction_name,
                direction_value,
                len(packet),
                len(pcm_audio),
            )

            # YAMNet에는 방향 헤더를 제외한 PCM만 전달
            result = websocket.app.state.classifier.classify(
                pcm_audio
            )

            if result is None:
                await websocket.send_json({
                    "status": "not_detected",
                    "direction": direction_name,
                    "direction_value": direction_value,
                })
                continue

            top_sounds = result.get("top_sounds", [])

            if not top_sounds:
                await websocket.send_json({
                    "status": "not_detected",
                    "direction": direction_name,
                    "direction_value": direction_value,
                })
                continue

            # 분석 결과에 ESP32 방향 정보 추가
            result["direction"] = direction_name
            result["direction_value"] = direction_value

            # 가장 신뢰도가 높은 소리
            top = top_sounds[0]

            block = top["block"]
            category = top["category"]
            score = top["score"]
            now = time.time()

            logger.info(
                "소리 분석: %s - %s, direction=%s (%.1f%%)",
                category,
                block,
                direction_name,
                score * 100,
            )

            # ESP32에 전체 분석 결과 반환
            await websocket.send_json({
                "status": "success",
                "direction": direction_name,
                "direction_value": direction_value,
                "top_sounds": top_sounds,
            })

            # 설정한 신뢰도 이상일 때만 알림 처리
            if score < settings.ALERT_THRESHOLD:
                continue

            # 같은 소리에 대한 쿨다운 확인
            alert_key = f"{block}:{direction_name}"
            last_alert = _last_alert_time.get(alert_key, 0)

            if now - last_alert <= settings.COOLDOWN_SECONDS:
                continue

            # 하드웨어 알림
            await websocket.send_text(
                f"ALERT:{block}:{direction_name}"
            )

            logger.warning(
                "알림 전송: %s, direction=%s (%.1f%%)",
                block,
                direction_name,
                score * 100,
            )

            _last_alert_time[alert_key] = now

            # 백엔드로 방향 포함 분석 결과 전송
            await send_detection(
                websocket.app.state.http_client,
                result,
            )

    except WebSocketDisconnect:
        logger.info("ESP32 연결 해제")

    except Exception as error:
        logger.exception(
            "WebSocket 처리 중 에러: %s",
            error,
        )

        try:
            await websocket.close()
        except Exception:
            pass