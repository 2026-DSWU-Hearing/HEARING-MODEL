import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.direction import parse_audio_packet
from app.services.backend_client import send_detection

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# 설정
# ============================================================

# /ws/analyze 전용
# 16 kHz × 1초 × int16(2 bytes)
EXPECTED_AUDIO_BYTES = 32_000

# /ws/neckband에서 같은 소리의 반복 알림 방지
_last_alert_time: dict[str, float] = {}


# ============================================================
# /ws/neckband
#
# 기존 /ws의 넥밴드 로직을 그대로 복구
# ============================================================

@router.websocket("/ws/neckband")
async def neckband_websocket(websocket: WebSocket):
    """
    넥밴드(ESP32) 메인 WebSocket.

    ESP32 -> AI 서버

    packet 구조:
        [4-byte direction header]
        [PCM audio]

    방향 헤더:
        0 = FRONT
        1 = BACK
        2 = LEFT
        3 = RIGHT
        4 = UNKNOWN

    처리:
        packet 수신
        -> 방향 헤더 파싱
        -> PCM 분리
        -> YAMNet 분석
        -> ESP32에 분석 결과 반환
        -> threshold 이상이면 ALERT 반환
        -> cooldown 확인
        -> 백엔드에 detection 전송
    """

    await websocket.accept()

    logger.info(
        "ESP32 /ws/neckband 연결됨: %s",
        websocket.client,
    )

    try:
        while True:
            # -------------------------------------------------
            # 1. ESP32 바이너리 패킷 수신
            # -------------------------------------------------
            packet = await websocket.receive_bytes()

            # -------------------------------------------------
            # 2. 방향 헤더 + PCM 분리
            # -------------------------------------------------
            try:
                (
                    direction_value,
                    direction_name,
                    pcm_audio,
                ) = parse_audio_packet(packet)

            except ValueError as error:
                logger.warning(
                    "잘못된 오디오 패킷: %s",
                    error,
                )

                await websocket.send_json({
                    "status": "error",
                    "message": str(error),
                })

                continue

            logger.info(
                "오디오 패킷 수신: "
                "direction=%s(%d), "
                "packet=%d bytes, "
                "pcm=%d bytes",
                direction_name,
                direction_value,
                len(packet),
                len(pcm_audio),
            )

            # -------------------------------------------------
            # 3. YAMNet 분석
            #
            # 방향 헤더를 제거한 PCM만 classifier에 전달
            # -------------------------------------------------
            started_at = time.perf_counter()

            result = (
                websocket
                .app
                .state
                .classifier
                .classify(
                    pcm_audio
                )
            )

            inference_time = (
                time.perf_counter()
                - started_at
            )

            logger.info(
                "YAMNet 추론 완료: %.3f초",
                inference_time,
            )

            # -------------------------------------------------
            # 4. 감지 결과 없음
            # -------------------------------------------------
            if result is None:
                await websocket.send_json({
                    "status": "not_detected",
                    "direction": direction_name,
                    "direction_value": direction_value,
                })

                continue

            top_sounds = result.get(
                "top_sounds",
                [],
            )

            if not top_sounds:
                await websocket.send_json({
                    "status": "not_detected",
                    "direction": direction_name,
                    "direction_value": direction_value,
                })

                continue

            # -------------------------------------------------
            # 5. 분석 결과에 방향 추가
            # -------------------------------------------------
            result["direction"] = (
                direction_name
            )

            result["direction_value"] = (
                direction_value
            )

            # 가장 높은 신뢰도의 소리
            top = top_sounds[0]

            block = top["block"]
            category = top["category"]
            score = top["score"]

            now = time.time()

            logger.info(
                "소리 분석: %s - %s, "
                "direction=%s (%.1f%%)",
                category,
                block,
                direction_name,
                score * 100,
            )

            # -------------------------------------------------
            # 6. ESP32에 전체 분석 결과 반환
            # -------------------------------------------------
            await websocket.send_json({
                "status": "success",
                "direction": direction_name,
                "direction_value": direction_value,
                "top_sounds": top_sounds,
            })

            # -------------------------------------------------
            # 7. Alert threshold 확인
            # -------------------------------------------------
            if (
                score
                < settings.ALERT_THRESHOLD
            ):
                continue

            # -------------------------------------------------
            # 8. 같은 소리 + 같은 방향 쿨다운 확인
            # -------------------------------------------------
            alert_key = (
                f"{block}:{direction_name}"
            )

            last_alert = (
                _last_alert_time.get(
                    alert_key,
                    0,
                )
            )

            if (
                now - last_alert
                <= settings.COOLDOWN_SECONDS
            ):
                continue

            # -------------------------------------------------
            # 9. 하드웨어 알림
            # -------------------------------------------------
            await websocket.send_text(
                f"ALERT:"
                f"{block}:"
                f"{direction_name}"
            )

            logger.warning(
                "알림 전송: %s, "
                "direction=%s (%.1f%%)",
                block,
                direction_name,
                score * 100,
            )

            _last_alert_time[
                alert_key
            ] = now

            # -------------------------------------------------
            # 10. 백엔드로 분석 결과 전송
            # -------------------------------------------------
            await send_detection(
                websocket
                .app
                .state
                .http_client,
                result,
            )

    except WebSocketDisconnect:
        logger.info(
            "ESP32 /ws/neckband 연결 해제"
        )

    except Exception as error:
        logger.exception(
            "/ws/neckband 처리 중 오류: %s",
            error,
        )

        try:
            await websocket.close()

        except Exception:
            pass


# ============================================================
# /ws/analyze
#
# 방향 헤더 없이 1초 PCM을 바로 분석하는 별도 endpoint
# ============================================================

@router.websocket("/ws/analyze")
async def analyze_websocket(
    websocket: WebSocket,
):
    """
    별도 오디오 분석용 WebSocket.

    Client -> AI 서버

    - binary frame 1개
    - PCM signed int16 little-endian
    - 16 kHz
    - mono
    - 1초
    - 정확히 32,000 bytes
    - 방향 헤더 없음
    - 링버퍼 없음

    처리:
        PCM 1 frame
        -> classify() 1회
        -> top_sounds 반환

    AI 서버 -> Client

    {
        "status": "ok",
        "top_sounds": [...]
    }
    """

    await websocket.accept()

    logger.info(
        "/ws/analyze 연결됨: %s",
        websocket.client,
    )

    try:
        while True:
            # -------------------------------------------------
            # 1. 1초 PCM frame 수신
            # -------------------------------------------------
            audio_bytes = (
                await websocket
                .receive_bytes()
            )

            received_size = len(
                audio_bytes
            )

            logger.info(
                "/ws/analyze 오디오 수신: "
                "%d bytes",
                received_size,
            )

            # -------------------------------------------------
            # 2. 32,000 byte 검사
            # -------------------------------------------------
            if (
                received_size
                != EXPECTED_AUDIO_BYTES
            ):
                logger.warning(
                    "/ws/analyze 잘못된 프레임 크기: "
                    "expected=%d, received=%d",
                    EXPECTED_AUDIO_BYTES,
                    received_size,
                )

                await websocket.send_json({
                    "status": "error",
                    "message": (
                        "invalid audio frame size: "
                        f"expected "
                        f"{EXPECTED_AUDIO_BYTES}, "
                        f"received "
                        f"{received_size}"
                    ),
                    "top_sounds": [],
                })

                continue

            # -------------------------------------------------
            # 3. YAMNet 추론
            # -------------------------------------------------
            started_at = (
                time.perf_counter()
            )

            result = (
                websocket
                .app
                .state
                .classifier
                .classify(
                    audio_bytes
                )
            )

            inference_time = (
                time.perf_counter()
                - started_at
            )

            logger.info(
                "/ws/analyze "
                "YAMNet 추론 완료: %.3f초",
                inference_time,
            )

            # -------------------------------------------------
            # 4. 결과 없음
            # -------------------------------------------------
            if result is None:
                await websocket.send_json({
                    "status": "ok",
                    "top_sounds": [],
                })

                continue

            top_sounds = result.get(
                "top_sounds",
                [],
            )

            # -------------------------------------------------
            # 5. 분석 결과 반환
            # -------------------------------------------------
            await websocket.send_json({
                "status": "ok",
                "top_sounds": top_sounds,
            })

            logger.info(
                "/ws/analyze 결과 전송 완료: "
                "%d개",
                len(top_sounds),
            )

    except WebSocketDisconnect:
        logger.info(
            "/ws/analyze 연결 해제"
        )

    except Exception as error:
        logger.exception(
            "/ws/analyze 처리 중 오류: %s",
            error,
        )

        try:
            await websocket.close()

        except Exception:
            pass