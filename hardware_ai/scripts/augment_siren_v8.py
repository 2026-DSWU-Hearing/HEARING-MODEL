import csv
import random
import re
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from config import (
    DATASET_DIR,
    MANIFEST_DIR,
    SAMPLE_RATE,
    CLIP_SAMPLES,
    RANDOM_SEED,
)


# ============================================================
# 경로
# ============================================================

PREPARED_V6_DIR = (
    DATASET_DIR
    / "prepared_v6"
)

TRAIN_EMERGENCY_DIR = (
    PREPARED_V6_DIR
    / "train"
    / "emergency"
)

MANIFEST_PATH = (
    MANIFEST_DIR
    / "combined_v6.csv"
)

AUGMENTED_ROOT = (
    DATASET_DIR
    / "augmented_v8"
)

SIREN_OUTPUT_DIR = (
    AUGMENTED_ROOT
    / "train"
    / "emergency"
    / "siren"
)


# ============================================================
# 설정
# ============================================================

# 원본 siren clip 하나당
# 몇 개의 새 증강본을 만들지
AUGMENTATIONS_PER_CLIP = 2


# 너무 조용한 데이터 제외
MIN_RMS = 0.002


# ============================================================
# Seed
# ============================================================

def set_seed() -> None:

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )


# ============================================================
# 기존 증강 폴더 초기화
# ============================================================

def reset_output_dir() -> None:

    if SIREN_OUTPUT_DIR.exists():

        try:

            shutil.rmtree(
                SIREN_OUTPUT_DIR
            )

        except PermissionError:

            print(
                "[경고] 증강 폴더 전체 삭제 실패."
            )

            print(
                "OneDrive가 파일을 잡고 있을 수 있습니다."
            )

            for file in (
                SIREN_OUTPUT_DIR
                .rglob("*.wav")
            ):

                try:

                    file.unlink()

                except PermissionError:

                    print(
                        f"[삭제 실패] "
                        f"{file}"
                    )

    SIREN_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Manifest
# ============================================================

def load_manifest() -> list[dict]:

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"V6 manifest가 없습니다:\n"
            f"{MANIFEST_PATH}"
        )

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    return rows


# ============================================================
# Train siren row 번호 찾기
# ============================================================

def find_train_siren_row_indices(
    rows: list[dict],
) -> set[int]:
    """
    prepare_v6.py는 manifest row_index를
    1부터 시작해서 WAV 파일명에 넣었다.

    예:
        urbansound8k_001234_xxxxx_000.wav

    따라서 train + original_label=siren 행의
    row 번호만 모으면 prepared_v6에서
    siren clip을 정확히 찾을 수 있다.
    """

    result = set()

    for row_index, row in enumerate(
        rows,
        start=1,
    ):

        split = (
            row["split"]
            .strip()
            .lower()
        )

        original_label = (
            row["original_label"]
            .strip()
            .lower()
        )

        binary_label = int(
            row["binary_label"]
        )

        if (
            split == "train"
            and original_label == "siren"
            and binary_label == 1
        ):

            result.add(
                row_index
            )

    return result


# ============================================================
# prepared 파일에서 row index 추출
# ============================================================

def extract_row_index(
    path: Path,
) -> int | None:
    """
    파일 예:

    urbansound8k_001234_100852-0-0-0_003.wav

    여기서 001234 추출.
    """

    filename = path.name

    match = re.match(
        r".+?_(\d{6})_",
        filename,
    )

    if match is None:

        return None

    return int(
        match.group(1)
    )


# ============================================================
# Siren clip 수집
# ============================================================

def collect_siren_clips(
    siren_row_indices: set[int],
) -> list[Path]:

    if not TRAIN_EMERGENCY_DIR.exists():

        raise FileNotFoundError(
            f"train emergency 폴더가 없습니다:\n"
            f"{TRAIN_EMERGENCY_DIR}"
        )

    result = []

    for path in (
        TRAIN_EMERGENCY_DIR
        .glob("*.wav")
    ):

        row_index = extract_row_index(
            path
        )

        if row_index is None:
            continue

        if row_index in siren_row_indices:

            result.append(
                path
            )

    result.sort()

    return result


# ============================================================
# Audio load
# ============================================================

def load_audio(
    path: Path,
) -> np.ndarray:

    audio, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if len(audio) < CLIP_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                CLIP_SAMPLES
                - len(audio),
            ),
            mode="constant",
        )

    else:

        audio = audio[
            :CLIP_SAMPLES
        ]

    return audio.astype(
        np.float32
    )


# ============================================================
# RMS
# ============================================================

def calculate_rms(
    audio: np.ndarray,
) -> float:

    if audio.size == 0:

        return 0.0

    return float(
        np.sqrt(
            np.mean(
                np.square(
                    audio
                )
            )
        )
    )


# ============================================================
# 길이 정리
# ============================================================

def fix_length(
    audio: np.ndarray,
) -> np.ndarray:

    if len(audio) < CLIP_SAMPLES:

        audio = np.pad(
            audio,
            (
                0,
                CLIP_SAMPLES
                - len(audio),
            ),
            mode="constant",
        )

    elif len(audio) > CLIP_SAMPLES:

        audio = audio[
            :CLIP_SAMPLES
        ]

    return audio.astype(
        np.float32
    )


# ============================================================
# Random time shift
# ============================================================

def random_shift(
    audio: np.ndarray,
) -> np.ndarray:
    """
    최대 ±100ms 이동.
    """

    max_shift = int(
        SAMPLE_RATE * 0.10
    )

    shift = np.random.randint(
        -max_shift,
        max_shift + 1,
    )

    return np.roll(
        audio,
        shift,
    )


# ============================================================
# Gain
# ============================================================

def random_gain(
    audio: np.ndarray,
) -> np.ndarray:

    gain = np.random.uniform(
        0.75,
        1.20,
    )

    return (
        audio * gain
    )


# ============================================================
# Noise
# ============================================================

def add_noise(
    audio: np.ndarray,
) -> np.ndarray:
    """
    siren 특징이 사라지지 않을 정도의
    약한 Gaussian noise.
    """

    noise_level = np.random.uniform(
        0.001,
        0.008,
    )

    noise = np.random.normal(
        loc=0.0,
        scale=noise_level,
        size=audio.shape,
    ).astype(
        np.float32
    )

    return (
        audio + noise
    )


# ============================================================
# Pitch shift
# ============================================================

def random_pitch_shift(
    audio: np.ndarray,
) -> np.ndarray:
    """
    실제 siren도 음높이가 다양하므로
    약 ±1.5 semitone 범위 사용.
    """

    steps = np.random.uniform(
        -1.5,
        1.5,
    )

    shifted = librosa.effects.pitch_shift(
        audio,
        sr=SAMPLE_RATE,
        n_steps=steps,
    )

    return fix_length(
        shifted
    )


# ============================================================
# Time stretch
# ============================================================

def random_time_stretch(
    audio: np.ndarray,
) -> np.ndarray:
    """
    너무 과하게 늘리지 않고
    0.90~1.10 범위.
    """

    rate = np.random.uniform(
        0.90,
        1.10,
    )

    stretched = (
        librosa.effects.time_stretch(
            audio,
            rate=rate,
        )
    )

    return fix_length(
        stretched
    )


# ============================================================
# Variant A
# ============================================================

def augment_variant_a(
    audio: np.ndarray,
) -> np.ndarray:
    """
    Pitch + Gain + Noise
    """

    result = (
        audio.copy()
    )

    result = random_pitch_shift(
        result
    )

    result = random_gain(
        result
    )

    result = add_noise(
        result
    )

    result = np.clip(
        result,
        -1.0,
        1.0,
    )

    return fix_length(
        result
    )


# ============================================================
# Variant B
# ============================================================

def augment_variant_b(
    audio: np.ndarray,
) -> np.ndarray:
    """
    Time Stretch + Shift + Gain + Noise
    """

    result = (
        audio.copy()
    )

    result = random_time_stretch(
        result
    )

    result = random_shift(
        result
    )

    result = random_gain(
        result
    )

    result = add_noise(
        result
    )

    result = np.clip(
        result,
        -1.0,
        1.0,
    )

    return fix_length(
        result
    )


# ============================================================
# 하나의 clip 증강
# ============================================================

def augment_clip(
    audio: np.ndarray,
    variant_index: int,
) -> np.ndarray:

    if (
        variant_index % 2 == 0
    ):

        return augment_variant_a(
            audio
        )

    return augment_variant_b(
        audio
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    set_seed()

    print()
    print("=" * 85)
    print(
        "HEAR:ING Hardware AI "
        "- Siren Augmentation V8"
    )
    print("=" * 85)

    print()
    print(
        "대상:"
    )

    print(
        "- prepared_v6/train/emergency의 "
        "siren만 증강"
    )

    print(
        "- validation 증강 안 함"
    )

    print(
        "- final_test 증강 안 함"
    )

    print(
        f"- 원본 하나당 "
        f"{AUGMENTATIONS_PER_CLIP}개 추가"
    )

    # ========================================================
    # Manifest
    # ========================================================

    rows = load_manifest()

    siren_row_indices = (
        find_train_siren_row_indices(
            rows
        )
    )

    print()
    print(
        f"train siren 원본 manifest row: "
        f"{len(siren_row_indices)}"
    )

    # ========================================================
    # Siren prepared clips
    # ========================================================

    siren_clips = collect_siren_clips(
        siren_row_indices
    )

    if not siren_clips:

        raise RuntimeError(
            "prepared_v6 train에서 "
            "siren clip을 찾지 못했습니다."
        )

    print(
        f"train siren 1초 clip: "
        f"{len(siren_clips)}"
    )

    # ========================================================
    # Output 초기화
    # ========================================================

    reset_output_dir()

    # ========================================================
    # Augment
    # ========================================================

    created = 0
    failed = 0

    for index, source_path in enumerate(
        siren_clips,
        start=1,
    ):

        try:

            audio = load_audio(
                source_path
            )

        except Exception as error:

            print()
            print(
                f"[로드 실패] "
                f"{source_path}: "
                f"{error}"
            )

            failed += 1
            continue

        if calculate_rms(
            audio
        ) < MIN_RMS:

            failed += 1
            continue

        for variant_index in range(
            AUGMENTATIONS_PER_CLIP
        ):

            try:

                augmented = augment_clip(
                    audio,
                    variant_index,
                )

                output_name = (
                    f"siren_aug_"
                    f"{index:06d}_"
                    f"{variant_index:02d}.wav"
                )

                output_path = (
                    SIREN_OUTPUT_DIR
                    / output_name
                )

                sf.write(
                    output_path,
                    augmented,
                    SAMPLE_RATE,
                    subtype="PCM_16",
                )

                created += 1

            except Exception as error:

                print()
                print(
                    f"[증강 실패] "
                    f"{source_path}: "
                    f"{error}"
                )

                failed += 1

        if (
            index % 250 == 0
            or index == len(
                siren_clips
            )
        ):

            print(
                f"\r"
                f"{index}/{len(siren_clips)} "
                f"처리 | "
                f"생성={created}",
                end="",
                flush=True,
            )

    print()

    # ========================================================
    # 결과
    # ========================================================

    print()
    print("=" * 85)
    print("Siren 증강 완료")
    print("=" * 85)

    print(
        f"원본 siren clip : "
        f"{len(siren_clips)}"
    )

    print(
        f"새 증강 clip    : "
        f"{created}"
    )

    print(
        f"실패             : "
        f"{failed}"
    )

    print(
        f"저장 위치:\n"
        f"{SIREN_OUTPUT_DIR}"
    )

    print()
    print(
        "prepared_v6 원본 데이터는 "
        "수정하지 않았습니다."
    )

    print()
    print(
        "다음 단계:"
    )

    print(
        "V8 학습에서 "
        "기존 train + augmented_v8 siren을 "
        "함께 사용합니다."
    )


if __name__ == "__main__":
    main()