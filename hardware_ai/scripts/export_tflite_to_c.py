from pathlib import Path


# ============================================================
# 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TFLITE_PATH = (
    PROJECT_ROOT
    / "models"
    / "emergency_classifier_v8_int8.tflite"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "esp32_model"
)

HEADER_PATH = (
    OUTPUT_DIR
    / "model_data.h"
)

SOURCE_PATH = (
    OUTPUT_DIR
    / "model_data.cc"
)


# ============================================================
# 설정
# ============================================================

ARRAY_NAME = "g_emergency_model"

BYTES_PER_LINE = 12


# ============================================================
# Header 생성
# ============================================================

def make_header(
    model_size: int,
) -> str:

    return f"""#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#include <cstddef>
#include <cstdint>

extern const unsigned char {ARRAY_NAME}[];
extern const unsigned int {ARRAY_NAME}_len;

#endif  // MODEL_DATA_H_
"""


# ============================================================
# C++ Source 생성
# ============================================================

def make_source(
    model_bytes: bytes,
) -> str:

    lines = []

    for start in range(
        0,
        len(model_bytes),
        BYTES_PER_LINE,
    ):
        chunk = model_bytes[
            start:
            start + BYTES_PER_LINE
        ]

        byte_text = ", ".join(
            f"0x{value:02x}"
            for value in chunk
        )

        lines.append(
            f"    {byte_text},"
        )

    array_body = "\n".join(
        lines
    )

    return f"""#include "model_data.h"

alignas(16)
const unsigned char {ARRAY_NAME}[] = {{
{array_body}
}};

const unsigned int {ARRAY_NAME}_len =
    sizeof({ARRAY_NAME});
"""


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not TFLITE_PATH.exists():

        raise FileNotFoundError(
            f"TFLite 모델이 없습니다:\n"
            f"{TFLITE_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_bytes = (
        TFLITE_PATH
        .read_bytes()
    )

    HEADER_PATH.write_text(
        make_header(
            len(model_bytes)
        ),
        encoding="utf-8",
    )

    SOURCE_PATH.write_text(
        make_source(
            model_bytes
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(
        "TFLite → C array 변환 완료"
    )
    print("=" * 70)

    print(
        f"모델 크기: "
        f"{len(model_bytes)} bytes"
    )

    print(
        f"Header:\n"
        f"{HEADER_PATH}"
    )

    print(
        f"Source:\n"
        f"{SOURCE_PATH}"
    )


if __name__ == "__main__":
    main()