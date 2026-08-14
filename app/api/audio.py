import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# 16 kHz × 1초 × int16(2 bytes)
EXPECTED_AUDIO_BYTES = 32_000


@router.websocket("/ws/analyze")
async def analyze_websocket(websocket: WebSocket):
    """
    하드웨어 AI 오디오 분석용 WebSocket.

    ESP32 -> AI 서버
    - binary frame 1개
    - PCM signed int16 little-endian
    - 16 kHz
    - mono
    - 1초
    - 정확히 32,000 bytes
    - 방향 헤더 없음

    처리 방식
    - frame 1개 수신 = YAMNet 추론 1회
    - 별도 링버퍼 사용하지 않음

    AI 서버 -> ESP32
    {
        "status": "ok",
        "top_sounds": [...]
    }
    """

    await websocket.accept()

    logger.info(
        "ESP32 /ws/analyze 연결됨: %s",
        websocket.client,
    )

    try:
        while True:
            # -------------------------------------------------
            # 1. ESP32로부터 1초 PCM 프레임 1개 수신
            # -------------------------------------------------
            audio_bytes = await websocket.receive_bytes()

            received_size = len(audio_bytes)

            logger.info(
                "오디오 프레임 수신: %d bytes",
                received_size,
            )

            # -------------------------------------------------
            # 2. 프레임 크기 확인
            #
            # 16,000 samples
            # × int16 2 bytes
            # = 32,000 bytes
            # -------------------------------------------------
            if received_size != EXPECTED_AUDIO_BYTES:
                logger.warning(
                    "잘못된 오디오 프레임 크기: "
                    "expected=%d, received=%d",
                    EXPECTED_AUDIO_BYTES,
                    received_size,
                )

                # 응답이 없으면 하드웨어가 timeout으로
                # 재연결할 수 있으므로 오류도 즉시 반환
                await websocket.send_json({
                    "status": "error",
                    "message": (
                        f"invalid audio frame size: "
                        f"expected {EXPECTED_AUDIO_BYTES}, "
                        f"received {received_size}"
                    ),
                    "top_sounds": [],
                })

                continue

            # -------------------------------------------------
            # 3. YAMNet 추론
            #
            # 별도 링버퍼 없음.
            # 받은 1 frame = 추론 1회.
            # -------------------------------------------------
            started_at = time.perf_counter()

            result = websocket.app.state.classifier.classify(
                audio_bytes
            )

            inference_time = time.perf_counter() - started_at

            logger.info(
                "YAMNet 추론 완료: %.3f초",
                inference_time,
            )

            # -------------------------------------------------
            # 4. 결과가 없는 경우에도 반드시 응답
            # -------------------------------------------------
            if result is None:
                await websocket.send_json({
                    "status": "ok",
                    "top_sounds": [],
                })

                continue

            # classifier.py의 기존 반환값
            #
            # {
            #     "top_sounds": [...]
            # }
            #
            # 을 그대로 사용
            # -------------------------------------------------
            top_sounds = result.get(
                "top_sounds",
                [],
            )

            await websocket.send_json({
                "status": "ok",
                "top_sounds": top_sounds,
            })

            logger.info(
                "분석 결과 전송 완료: %d개",
                len(top_sounds),
            )

    except WebSocketDisconnect:
        logger.info(
            "ESP32 /ws/analyze 연결 해제"
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