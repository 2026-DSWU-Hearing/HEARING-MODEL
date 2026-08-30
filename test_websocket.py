import asyncio
import json
import struct

import websockets


SERVER_URL = "ws://127.0.0.1:8765"

# 16 kHz × 1초 × int16 = 32,000 bytes
PCM_SIZE = 32_000

# 테스트용 무음 PCM
TEST_PCM = bytes(PCM_SIZE)


# ============================================================
# /ws/analyze 테스트
# ============================================================

async def test_analyze():
    url = f"{SERVER_URL}/ws/analyze"

    print()
    print("=" * 60)
    print("1. /ws/analyze 테스트")
    print("=" * 60)

    try:
        async with websockets.connect(
            url,
            ping_interval=None,
            ping_timeout=None,
        ) as websocket:

            print(f"[연결 성공] {url}")

            print(
                f"[전송] PCM {len(TEST_PCM)} bytes"
            )

            await websocket.send(TEST_PCM)

            response = await asyncio.wait_for(
                websocket.recv(),
                timeout=5,
            )

            print("[수신]")
            print_response(response)

    except asyncio.TimeoutError:
        print("[실패] 서버 응답 시간 초과")

    except Exception as error:
        print(
            f"[실패] /ws/analyze 테스트 오류: "
            f"{error}"
        )


# ============================================================
# /ws/neckband 테스트
# ============================================================

async def test_neckband():
    url = f"{SERVER_URL}/ws/neckband"

    print()
    print("=" * 60)
    print("2. /ws/neckband 테스트")
    print("=" * 60)

    # --------------------------------------------------------
    # 테스트 방향
    #
    # 기존 direction.py가 4-byte 정수 헤더를 읽는 구조이므로
    # little-endian signed int 형식으로 생성
    # --------------------------------------------------------

    direction_value = 0

    direction_header = struct.pack(
        "<i",
        direction_value,
    )

    # 최종 packet:
    #
    # [4-byte direction]
    # +
    # [32,000-byte PCM]
    #
    packet = (
        direction_header
        + TEST_PCM
    )

    try:
        async with websockets.connect(
            url,
            ping_interval=None,
            ping_timeout=None,
        ) as websocket:

            print(f"[연결 성공] {url}")

            print(
                f"[전송] direction={direction_value}"
            )

            print(
                f"[전송] header="
                f"{len(direction_header)} bytes"
            )

            print(
                f"[전송] PCM="
                f"{len(TEST_PCM)} bytes"
            )

            print(
                f"[전송] 전체 packet="
                f"{len(packet)} bytes"
            )

            await websocket.send(packet)

            # -------------------------------------------------
            # 첫 번째 응답
            #
            # success / not_detected / error
            # -------------------------------------------------

            response = await asyncio.wait_for(
                websocket.recv(),
                timeout=5,
            )

            print("[첫 번째 수신]")
            print_response(response)

            # -------------------------------------------------
            # threshold를 넘었다면
            #
            # ALERT:block:direction
            #
            # 메시지가 하나 더 올 수 있음.
            # -------------------------------------------------

            try:
                second_response = (
                    await asyncio.wait_for(
                        websocket.recv(),
                        timeout=1,
                    )
                )

                print("[추가 수신]")
                print_response(
                    second_response
                )

            except asyncio.TimeoutError:
                print(
                    "[추가 메시지 없음] "
                    "ALERT가 발생하지 않은 경우 정상"
                )

    except asyncio.TimeoutError:
        print("[실패] 서버 응답 시간 초과")

    except Exception as error:
        print(
            f"[실패] /ws/neckband 테스트 오류: "
            f"{error}"
        )


# ============================================================
# 응답 출력
# ============================================================

def print_response(response):
    if isinstance(response, bytes):
        print(
            f"binary: {len(response)} bytes"
        )
        return

    try:
        data = json.loads(response)

        print(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            )
        )

    except json.JSONDecodeError:
        print(response)


# ============================================================
# main
# ============================================================

async def main():
    print()
    print("HEAR:ING WebSocket 테스트 시작")
    print(f"서버: {SERVER_URL}")

    await test_analyze()

    await test_neckband()

    print()
    print("=" * 60)
    print("테스트 종료")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())