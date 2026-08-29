import asyncio
import json
import statistics
import time

import websockets


WS_URL = "ws://127.0.0.1:8765/ws/analyze"

TIMEOUT_SECONDS = 2.0
PCM_SIZE = 32_000

# 한 클라이언트가 보내는 요청 횟수
REQUESTS_PER_CLIENT = 10

# 한계 탐색 설정
START_CLIENTS = 20
STEP_CLIENTS = 10
MAX_CLIENTS = 200


AUDIO_FRAME = bytes(PCM_SIZE)


async def run_client(client_id: int):
    latencies = []
    successes = 0
    timeouts = 0
    errors = 0

    try:
        async with websockets.connect(
            WS_URL,
            max_size=None,
            open_timeout=5,
        ) as websocket:

            for request_number in range(REQUESTS_PER_CLIENT):
                started = time.perf_counter()

                try:
                    await websocket.send(AUDIO_FRAME)

                    response = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=TIMEOUT_SECONDS,
                    )

                    elapsed = time.perf_counter() - started
                    latencies.append(elapsed)

                    try:
                        data = json.loads(response)

                        if data.get("status") in ("ok", "success"):
                            successes += 1
                        else:
                            errors += 1

                    except json.JSONDecodeError:
                        errors += 1

                except asyncio.TimeoutError:
                    timeouts += 1

                except Exception as error:
                    errors += 1

                    print(
                        f"[Client {client_id}] "
                        f"request={request_number + 1} "
                        f"ERROR: {error}"
                    )

    except Exception as error:
        print(
            f"[Client {client_id}] 연결 실패: {error}"
        )

        errors += REQUESTS_PER_CLIENT

    return {
        "latencies": latencies,
        "successes": successes,
        "timeouts": timeouts,
        "errors": errors,
    }


def percentile(values, percent):
    if not values:
        return 0.0

    sorted_values = sorted(values)

    index = int(
        (len(sorted_values) - 1)
        * percent
    )

    return sorted_values[index]


async def benchmark(client_count: int):
    print()
    print("=" * 60)
    print(
        f"동시 클라이언트 테스트: "
        f"{client_count}개"
    )
    print("=" * 60)

    started = time.perf_counter()

    results = await asyncio.gather(
        *[
            run_client(client_id)
            for client_id in range(client_count)
        ]
    )

    total_elapsed = time.perf_counter() - started

    all_latencies = []

    total_success = 0
    total_timeout = 0
    total_errors = 0

    for result in results:
        all_latencies.extend(result["latencies"])
        total_success += result["successes"]
        total_timeout += result["timeouts"]
        total_errors += result["errors"]

    total_requests = (
        client_count
        * REQUESTS_PER_CLIENT
    )

    print(f"\n전체 요청: {total_requests}")
    print(f"성공: {total_success}")
    print(f"Timeout: {total_timeout}")
    print(f"Error: {total_errors}")

    mean_latency = 0.0
    median_latency = 0.0
    p95_latency = 0.0
    maximum_latency = 0.0

    if all_latencies:
        mean_latency = statistics.mean(all_latencies)
        median_latency = statistics.median(all_latencies)
        p95_latency = percentile(
            all_latencies,
            0.95,
        )
        maximum_latency = max(all_latencies)

        print()
        print(
            f"평균 응답시간: "
            f"{mean_latency * 1000:.1f} ms"
        )
        print(
            f"중앙값: "
            f"{median_latency * 1000:.1f} ms"
        )
        print(
            f"P95: "
            f"{p95_latency * 1000:.1f} ms"
        )
        print(
            f"최대: "
            f"{maximum_latency * 1000:.1f} ms"
        )

    print(
        f"전체 테스트 시간: "
        f"{total_elapsed:.2f} sec"
    )

    passed = (
        total_timeout == 0
        and total_errors == 0
        and total_success == total_requests
        and all_latencies
        and p95_latency < TIMEOUT_SECONDS
    )

    print()
    print(
        "RESULT: PASS"
        if passed
        else "RESULT: FAIL"
    )

    return {
        "client_count": client_count,
        "passed": passed,
        "success": total_success,
        "timeouts": total_timeout,
        "errors": total_errors,
        "mean": mean_latency,
        "p95": p95_latency,
        "max": maximum_latency,
    }


async def find_limit():
    print()
    print("#" * 60)
    print("WebSocket + YAMNet 동시 처리 한계 탐색 시작")
    print("#" * 60)

    last_pass = None
    first_fail = None

    client_count = START_CLIENTS

    # --------------------------------------------------------
    # 1단계
    # 20 → 30 → 40 → ...
    # 큰 단위로 FAIL 지점 찾기
    # --------------------------------------------------------

    while client_count <= MAX_CLIENTS:
        result = await benchmark(client_count)

        if result["passed"]:
            last_pass = client_count

            print(
                f"\n[{client_count}개] 안정 처리 성공"
            )

            client_count += STEP_CLIENTS

        else:
            first_fail = client_count

            print(
                f"\n[{client_count}개] 첫 실패 발생"
            )

            break

        await asyncio.sleep(2)

    # MAX_CLIENTS까지 모두 성공
    if first_fail is None:
        print()
        print("=" * 60)
        print(
            f"{MAX_CLIENTS}개까지 모두 PASS"
        )
        print(
            "현재 설정에서는 한계점을 찾지 못했습니다."
        )
        print(
            "MAX_CLIENTS를 더 높여 다시 테스트하세요."
        )
        print("=" * 60)
        return

    # 시작값부터 실패
    if last_pass is None:
        print()
        print("=" * 60)
        print(
            f"{START_CLIENTS}개에서 이미 FAIL"
        )
        print(
            "START_CLIENTS를 더 낮춰서 다시 테스트하세요."
        )
        print("=" * 60)
        return

    # --------------------------------------------------------
    # 2단계
    # 마지막 PASS와 첫 FAIL 사이를
    # 1개씩 세밀하게 탐색
    # --------------------------------------------------------

    print()
    print("#" * 60)
    print(
        f"세부 탐색 시작: "
        f"{last_pass + 1} ~ {first_fail - 1}"
    )
    print("#" * 60)

    stable_limit = last_pass

    for client_count in range(
        last_pass + 1,
        first_fail,
    ):
        await asyncio.sleep(2)

        result = await benchmark(client_count)

        if result["passed"]:
            stable_limit = client_count
        else:
            first_fail = client_count
            break

    print()
    print("=" * 60)
    print("최종 결과")
    print("=" * 60)

    print(
        f"2초 timeout 기준 "
        f"최대 안정 동시 클라이언트 수: "
        f"{stable_limit}개"
    )

    print(
        f"최초 실패 클라이언트 수: "
        f"{first_fail}개"
    )

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(find_limit())