import statistics
import time

import numpy as np

from app.services.classifier import AudioClassifier


# ============================================================
# 설정
# ============================================================

WARMUP_COUNT = 5
TEST_COUNT = 50

SAMPLE_RATE = 16_000


def percentile(
    values,
    percent,
):

    sorted_values = sorted(values)

    index = int(
        (len(sorted_values) - 1)
        * percent
    )

    return sorted_values[index]


def create_test_audio() -> bytes:
    """
    1초 PCM int16 테스트 데이터 생성.

    16,000 samples × 2 bytes
    = 32,000 bytes
    """

    rng = np.random.default_rng(
        seed=42
    )

    samples = rng.integers(
        low=-2000,
        high=2000,
        size=SAMPLE_RATE,
        dtype=np.int16,
    )

    return samples.tobytes()


def main():

    print(
        "AudioClassifier 로드 중..."
    )

    classifier = AudioClassifier()

    audio_bytes = create_test_audio()

    print(
        f"PCM 크기: "
        f"{len(audio_bytes)} bytes"
    )

    print()
    print(
        f"Warm-up {WARMUP_COUNT}회..."
    )

    # --------------------------------------------------------
    # TensorFlow 초기화 영향을 제거하기 위한 warm-up
    # --------------------------------------------------------

    for index in range(
        WARMUP_COUNT
    ):

        started = time.perf_counter()

        classifier.classify(
            audio_bytes
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        print(
            f"warm-up {index + 1}: "
            f"{elapsed * 1000:.1f} ms"
        )

    print()
    print(
        f"본 측정 {TEST_COUNT}회..."
    )

    times = []

    for index in range(
        TEST_COUNT
    ):

        started = time.perf_counter()

        classifier.classify(
            audio_bytes
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        times.append(
            elapsed
        )

        print(
            f"{index + 1:02d}: "
            f"{elapsed * 1000:.1f} ms"
        )

    mean_time = statistics.mean(
        times
    )

    median_time = statistics.median(
        times
    )

    p95_time = percentile(
        times,
        0.95,
    )

    minimum_time = min(
        times
    )

    maximum_time = max(
        times
    )

    print()
    print("=" * 50)
    print("classify() benchmark")
    print("=" * 50)

    print(
        f"측정 횟수: "
        f"{TEST_COUNT}"
    )

    print(
        f"평균: "
        f"{mean_time * 1000:.1f} ms"
    )

    print(
        f"중앙값: "
        f"{median_time * 1000:.1f} ms"
    )

    print(
        f"P95: "
        f"{p95_time * 1000:.1f} ms"
    )

    print(
        f"최소: "
        f"{minimum_time * 1000:.1f} ms"
    )

    print(
        f"최대: "
        f"{maximum_time * 1000:.1f} ms"
    )

    print()

    theoretical_rate = (
        1.0 / mean_time
    )

    print(
        "단순 이론상 직렬 처리량: "
        f"{theoretical_rate:.2f} "
        "inference/sec"
    )


if __name__ == "__main__":
    main()