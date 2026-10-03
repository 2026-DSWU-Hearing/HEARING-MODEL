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
# ESP32 넥밴드 메인 WebSocket
# ============================================================

@router.websocket("/ws/neckband")
async def neckband_websocket(websocket: WebSocket):
    """
    넥밴드(ESP32) 메인 WebSocket.

    ESP32 -> AI 서버

    packet 구조:
        [0]   direction         1 byte
        [1]   ondevice_vibrated 1 byte
        [2:4] padding           2 bytes
        [4:]  PCM audio

    direction:
        0 = FRONT
        1 = BACK
        2 = LEFT
        3 = RIGHT
        4 = UNKNOWN

    ondevice_vibrated:
        0 = 온디바이스 AI에 의한 로컬 진동 없음
        1 = 온디바이스 AI가 긴급으로 판단하여 이미 로컬 진동 수행

    처리:
        packet 수신
        -> 방향 / 온디바이스 진동 여부 파싱
        -> PCM 분리
        -> YAMNet 분석
        -> ESP32에 분석 결과 반환

        ondevice_vibrated == False:
            -> ALERT_THRESHOLD 확인
            -> COOLDOWN_SECONDS 확인
            -> 하드웨어 ALERT
            -> 백엔드 detection 전송

        ondevice_vibrated == True:
            -> ALERT_THRESHOLD 무시
            -> COOLDOWN_SECONDS 무시
            -> 하드웨어는 이미 진동했으므로 ALERT 재전송하지 않음
            -> 백엔드 detection 강제 전송
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
            # 2. 방향 + 온디바이스 진동 여부 + PCM 분리
            # -------------------------------------------------

            try:
                (
                    direction_value,
                    direction_name,
                    ondevice_vibrated,
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
                "ondevice_vibrated=%s, "
                "packet=%d bytes, "
                "pcm=%d bytes",
                direction_name,
                direction_value,
                ondevice_vibrated,
                len(packet),
                len(pcm_audio),
            )

            # -------------------------------------------------
            # 3. YAMNet 분석
            #
            # 4-byte 헤더를 제거한 PCM만 classifier에 전달
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
                    "ondevice_vibrated": ondevice_vibrated,
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
                    "ondevice_vibrated": ondevice_vibrated,
                })

                continue

            # -------------------------------------------------
            # 5. 분석 결과에 방향 / 온디바이스 진동 여부 추가
            # -------------------------------------------------

            result["direction"] = (
                direction_name
            )

            result["direction_value"] = (
                direction_value
            )

            result["ondevice_vibrated"] = (
                ondevice_vibrated
            )

            # 가장 높은 신뢰도의 소리
            top = top_sounds[0]

            block = top["block"]
            category = top["category"]
            score = top["score"]

            now = time.time()

            logger.info(
                "소리 분석: %s - %s, "
                "direction=%s, "
                "ondevice_vibrated=%s "
                "(%.1f%%)",
                category,
                block,
                direction_name,
                ondevice_vibrated,
                score * 100,
            )

            # -------------------------------------------------
            # 6. ESP32에 전체 분석 결과 반환
            # -------------------------------------------------

            await websocket.send_json({
                "status": "success",
                "direction": direction_name,
                "direction_value": direction_value,
                "ondevice_vibrated": ondevice_vibrated,
                "top_sounds": top_sounds,
            })

            # -------------------------------------------------
            # 7. 백엔드 보고 여부 판단
            #
            # 긴급 소리:
            #   YAMNet이 같은 실제 소리를
            #   사이렌 → 응급차량 → 경보음 등으로
            #   다르게 분류할 수 있으므로 block 이름과 관계없이
            #   "긴급" 하나의 key로 cooldown을 적용한다.
            #
            # ondevice_vibrated == True 이면서
            # YAMNet 재분류 결과도 "긴급"인 경우:
            #   - ALERT_THRESHOLD는 무시
            #   - COOLDOWN_SECONDS는 그대로 적용
            #
            # 그 외:
            #   - 기존 ALERT_THRESHOLD 적용
            #   - COOLDOWN_SECONDS 적용
            # -------------------------------------------------

            # 긴급 소리는 이름과 상관없이 하나로 묶어서 쿨다운
            # (넥밴드 진동 쿨다운과 같은 기준)
            is_emergency = category == "긴급"

            alert_key = (
                "긴급"
                if is_emergency
                else f"{block}"
            )

            # 온디바이스에서 이미 진동했고,
            # 서버의 재분류 결과도 긴급인 경우에만
            # 신뢰도 임계값을 건너뛴다.
            skip_threshold = (
                ondevice_vibrated
                and is_emergency
            )

            # 일반적인 경우에는 기존 0.7 threshold 적용
            if (
                not skip_threshold
                and score < settings.ALERT_THRESHOLD
            ):
                continue

            # cooldown은 모든 경우에 적용
            last_alert = _last_alert_time.get(
                alert_key,
                0,
            )

            if (
                now - last_alert
                <= settings.COOLDOWN_SECONDS
            ):
                continue

            if skip_threshold:
                logger.info(
                    "온디바이스 AI 긴급 판정 감지: "
                    "threshold 우회 "
                    "(block=%s, direction=%s, score=%.1f%%)",
                    block,
                    direction_name,
                    score * 100,
                )

            # -------------------------------------------------
            # 8. 하드웨어 ALERT
            #
            # 온디바이스 AI가 이미 진동시킨 경우에는
            # 중복 진동을 방지하기 위해 ALERT를 다시 보내지 않는다.
            # -------------------------------------------------

            if not ondevice_vibrated:
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

            else:
                logger.info(
                    "하드웨어 ALERT 생략: "
                    "온디바이스 AI에서 이미 진동함 "
                    "(block=%s, direction=%s)",
                    block,
                    direction_name,
                )

            # -------------------------------------------------
            # 9. cooldown 시간 갱신
            #
            # 요구사항에 따라 ondevice_vibrated == True여도
            # _last_alert_time을 갱신한다.
            # -------------------------------------------------

            _last_alert_time[
                alert_key
            ] = now

            # -------------------------------------------------
            # 10. 백엔드로 분석 결과 전송
            #
            # ondevice_vibrated는 true / false 관계없이
            # 항상 백엔드 request body에 포함된다.
            # -------------------------------------------------

            await send_detection(
                websocket
                .app
                .state
                .http_client,
                result,
                ondevice_vibrated=ondevice_vibrated,
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