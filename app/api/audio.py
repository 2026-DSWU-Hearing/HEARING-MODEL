import logging
import time

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# Audio 설정
# ============================================================

# 16 kHz × 1초 × int16(2 bytes)
EXPECTED_AUDIO_BYTES = 32_000


# ============================================================
# 공통 분석 함수
# ============================================================

async def analyze_audio_frame(
    websocket: WebSocket,
    audio_bytes: bytes,
) -> None:
    """
    1초 PCM binary frame 하나를 분석하고
    결과를 WebSocket으로 반환한다.

    입력:
    - PCM signed int16 little-endian
    - 16 kHz
    - mono
    - 1초
    - 32,000 bytes

    출력:
    {
        "status": "ok",
        "top_sounds": [...]
    }
    """

    received_size = len(
        audio_bytes
    )

    logger.info(
        "오디오 프레임 수신: %d bytes",
        received_size,
    )

    # --------------------------------------------------------
    # 프레임 크기 확인
    # --------------------------------------------------------

    if (
        received_size
        != EXPECTED_AUDIO_BYTES
    ):
        logger.warning(
            "잘못된 오디오 프레임 크기: "
            "expected=%d, received=%d",
            EXPECTED_AUDIO_BYTES,
            received_size,
        )

        await websocket.send_json({
            "status": "error",
            "message": (
                f"invalid audio frame size: "
                f"expected {EXPECTED_AUDIO_BYTES}, "
                f"received {received_size}"
            ),
            "top_sounds": [],
        })

        return

    # --------------------------------------------------------
    # YAMNet 추론
    # --------------------------------------------------------

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
        "YAMNet 추론 완료: %.3f초",
        inference_time,
    )

    # --------------------------------------------------------
    # 결과 없음
    # --------------------------------------------------------

    if result is None:

        await websocket.send_json({
            "status": "ok",
            "top_sounds": [],
        })

        return

    # --------------------------------------------------------
    # classifier 결과
    # --------------------------------------------------------

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


# ============================================================
# 1. 넥밴드 메인 WebSocket
# ============================================================

@router.websocket("/ws/neckband")
async def neckband_websocket(
    websocket: WebSocket,
):
    """
    넥밴드(ESP32) 메인 WebSocket.

    ESP32 -> AI 서버
    - binary PCM frame
    - 16 kHz
    - mono
    - signed int16 little-endian
    - 1초
    - 32,000 bytes

    AI 서버 -> ESP32
    {
        "status": "ok",
        "top_sounds": [...]
    }
    """

    await websocket.accept()

    logger.info(
        "ESP32 /ws/neckband 연결됨: %s",
        websocket.client,
    )

    try:
        while True:

            # ------------------------------------------------
            # ESP32에서 1초 PCM frame 수신
            # ------------------------------------------------

            audio_bytes = (
                await websocket
                .receive_bytes()
            )

            # ------------------------------------------------
            # 공통 분석
            # ------------------------------------------------

            await analyze_audio_frame(
                websocket,
                audio_bytes,
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
# 2. 별도 분석 WebSocket
# ============================================================

@router.websocket("/ws/analyze")
async def analyze_websocket(
    websocket: WebSocket,
):
    """
    별도 오디오 분석용 WebSocket.

    백엔드 또는 테스트 클라이언트에서
    AI 서버에 1초 PCM frame을 직접 보내
    YAMNet 분석 결과를 받을 때 사용한다.

    Client -> AI Server
    - binary frame 1개
    - PCM signed int16 little-endian
    - 16 kHz
    - mono
    - 1초
    - 정확히 32,000 bytes
    - 방향 헤더 없음
    - 링버퍼 없음

    처리 방식
    - frame 1개 수신
    - classify() 1회
    - 즉시 결과 반환

    AI Server -> Client
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

            # ------------------------------------------------
            # 1초 PCM frame 1개 수신
            # ------------------------------------------------

            audio_bytes = (
                await websocket
                .receive_bytes()
            )

            # ------------------------------------------------
            # 공통 분석
            # ------------------------------------------------

            await analyze_audio_frame(
                websocket,
                audio_bytes,
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