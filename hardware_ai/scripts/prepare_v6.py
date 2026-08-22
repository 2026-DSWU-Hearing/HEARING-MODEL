import csv
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
    TRAIN_HOP_SAMPLES,
    EVAL_HOP_SAMPLES,
)


# ============================================================
# V6 경로
# ============================================================

MANIFEST_PATH = (
    MANIFEST_DIR
    / "combined_v6.csv"
)

PREPARED_V6_DIR = (
    DATASET_DIR
    / "prepared_v6"
)


# ============================================================
# 출력 폴더 초기화
# ============================================================

def reset_output_dirs() -> None:
    """
    OneDrive PermissionError를 줄이기 위해
    prepared_v6 최상위 폴더 자체는 삭제하지 않는다.

    내부 class 폴더만 초기화.
    """

    PREPARED_V6_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in (
        "train",
        "val",
        "final_test",
    ):

        for class_name in (
            "normal",
            "emergency",
        ):

            output_dir = (
                PREPARED_V6_DIR
                / split
                / class_name
            )

            if output_dir.exists():

                try:

                    shutil.rmtree(
                        output_dir
                    )

                except PermissionError:

                    print(
                        f"[경고] 폴더 삭제 실패: "
                        f"{output_dir}"
                    )

                    # OneDrive가 잡고 있는 경우
                    # 파일만 최대한 삭제
                    for item in output_dir.rglob(
                        "*"
                    ):

                        try:

                            if (
                                item.is_file()
                                or item.is_symlink()
                            ):

                                item.unlink()

                        except PermissionError:

                            print(
                                f"[경고] 삭제 실패: "
                                f"{item}"
                            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


# ============================================================
# Audio
# ============================================================

def load_audio(
    path: Path,
) -> np.ndarray:

    audio, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
    )

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
# 1초 Clip
# ============================================================

def split_audio(
    audio: np.ndarray,
    hop_samples: int,
) -> list[np.ndarray]:

    clips = []

    # --------------------------------------------------------
    # 1초보다 짧음
    # --------------------------------------------------------

    if len(audio) < CLIP_SAMPLES:

        padded = np.pad(
            audio,
            (
                0,
                CLIP_SAMPLES
                - len(audio),
            ),
            mode="constant",
        )

        if calculate_rms(
            padded
        ) >= 0.002:

            clips.append(
                padded.astype(
                    np.float32
                )
            )

        return clips

    # --------------------------------------------------------
    # Sliding window
    # --------------------------------------------------------

    for start in range(
        0,
        len(audio)
        - CLIP_SAMPLES
        + 1,
        hop_samples,
    ):

        clip = audio[
            start:
            start + CLIP_SAMPLES
        ]

        if (
            len(clip)
            != CLIP_SAMPLES
        ):

            continue

        if (
            calculate_rms(
                clip
            )
            < 0.002
        ):

            continue

        clips.append(
            clip.astype(
                np.float32
            )
        )

    return clips


# ============================================================
# 한 row 처리
# ============================================================

def process_row(
    row: dict,
    row_index: int,
) -> int:

    split = (
        row["split"]
        .strip()
        .lower()
    )

    if split not in (
        "train",
        "val",
        "final_test",
    ):

        print(
            f"[잘못된 split] "
            f"{split}"
        )

        return 0

    try:

        binary_label = int(
            row["binary_label"]
        )

    except (
        ValueError,
        TypeError,
    ):

        print(
            f"[잘못된 label] "
            f"{row.get('binary_label')}"
        )

        return 0

    if binary_label == 1:

        class_name = (
            "emergency"
        )

    elif binary_label == 0:

        class_name = (
            "normal"
        )

    else:

        return 0

    source_path = Path(
        row["path"]
    )

    if not source_path.exists():

        print(
            f"[파일 없음] "
            f"{source_path}"
        )

        return 0

    try:

        audio = load_audio(
            source_path
        )

    except Exception as error:

        print(
            f"[오디오 로드 실패] "
            f"{source_path}: "
            f"{error}"
        )

        return 0

    # --------------------------------------------------------
    # Train만 overlap
    #
    # val/final_test는
    # non-overlap 1초
    # --------------------------------------------------------

    if split == "train":

        hop_samples = (
            TRAIN_HOP_SAMPLES
        )

    else:

        hop_samples = (
            EVAL_HOP_SAMPLES
        )

    clips = split_audio(
        audio,
        hop_samples,
    )

    if not clips:
        return 0

    output_dir = (
        PREPARED_V6_DIR
        / split
        / class_name
    )

    dataset_name = (
        row["dataset"]
        .strip()
        .replace(
            " ",
            "_",
        )
    )

    source_stem = (
        source_path.stem
        .replace(
            " ",
            "_",
        )
    )

    written = 0

    for clip_index, clip in enumerate(
        clips
    ):

        output_name = (
            f"{dataset_name}_"
            f"{row_index:06d}_"
            f"{source_stem}_"
            f"{clip_index:03d}.wav"
        )

        output_path = (
            output_dir
            / output_name
        )

        try:

            sf.write(
                output_path,
                clip,
                SAMPLE_RATE,
                subtype="PCM_16",
            )

            written += 1

        except Exception as error:

            print(
                f"[저장 실패] "
                f"{output_path}: "
                f"{error}"
            )

    return written


# ============================================================
# 결과 통계
# ============================================================

def count_files() -> dict:

    stats = {}

    print()
    print("=" * 90)
    print("V6 전처리 결과")
    print("=" * 90)

    for split in (
        "train",
        "val",
        "final_test",
    ):

        emergency_dir = (
            PREPARED_V6_DIR
            / split
            / "emergency"
        )

        normal_dir = (
            PREPARED_V6_DIR
            / split
            / "normal"
        )

        emergency_count = len(
            list(
                emergency_dir.glob(
                    "*.wav"
                )
            )
        )

        normal_count = len(
            list(
                normal_dir.glob(
                    "*.wav"
                )
            )
        )

        total = (
            emergency_count
            + normal_count
        )

        emergency_ratio = (
            emergency_count
            / total
            * 100
            if total
            else 0.0
        )

        stats[split] = {
            "emergency":
                emergency_count,

            "normal":
                normal_count,

            "total":
                total,
        }

        print(
            f"{split:10s} | "
            f"긴급={emergency_count:6d} | "
            f"일반={normal_count:6d} | "
            f"전체={total:6d} | "
            f"긴급비율="
            f"{emergency_ratio:5.1f}%"
        )

    return stats


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"V6 manifest가 없습니다:\n"
            f"{MANIFEST_PATH}\n\n"
            f"먼저 build_manifest_v6.py를 "
            f"실행하세요."
        )

    print()
    print("=" * 90)
    print(
        "HEAR:ING Hardware AI "
        "- Prepare V6"
    )
    print("=" * 90)

    print()
    print(
        f"Manifest: "
        f"{MANIFEST_PATH}"
    )

    print(
        f"출력 경로: "
        f"{PREPARED_V6_DIR}"
    )

    print(
        f"Sample rate: "
        f"{SAMPLE_RATE} Hz"
    )

    print(
        f"Clip samples: "
        f"{CLIP_SAMPLES}"
    )

    print(
        f"Train hop: "
        f"{TRAIN_HOP_SAMPLES}"
    )

    print(
        f"Eval hop: "
        f"{EVAL_HOP_SAMPLES}"
    )

    # ========================================================
    # 폴더 초기화
    # ========================================================

    print()
    print(
        "prepared_v6 초기화 중..."
    )

    reset_output_dirs()

    # ========================================================
    # Manifest
    # ========================================================

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    if not rows:

        raise RuntimeError(
            "combined_v6.csv가 비어 있습니다."
        )

    print(
        f"원본 파일 수: "
        f"{len(rows)}"
    )

    # ========================================================
    # Process
    # ========================================================

    total_clips = 0
    no_clip_rows = 0

    for row_index, row in enumerate(
        rows,
        start=1,
    ):

        count = process_row(
            row,
            row_index,
        )

        total_clips += count

        if count == 0:

            no_clip_rows += 1

        if (
            row_index % 250 == 0
            or row_index == len(rows)
        ):

            print(
                f"{row_index:5d}"
                f"/{len(rows)} "
                f"처리 완료 | "
                f"누적 clip="
                f"{total_clips}"
            )

    # ========================================================
    # 통계
    # ========================================================

    stats = count_files()

    # ========================================================
    # Sanity Check
    # ========================================================

    for split in (
        "train",
        "val",
        "final_test",
    ):

        if (
            stats[split][
                "emergency"
            ]
            == 0
        ):

            raise RuntimeError(
                f"{split} emergency가 "
                f"0개입니다."
            )

        if (
            stats[split][
                "normal"
            ]
            == 0
        ):

            raise RuntimeError(
                f"{split} normal이 "
                f"0개입니다."
            )

    print()
    print("=" * 90)
    print("V6 전처리 완료")
    print("=" * 90)

    print(
        f"총 생성 clip: "
        f"{total_clips}"
    )

    print(
        f"clip 생성 안 된 원본: "
        f"{no_clip_rows}"
    )

    print()
    print(
        "중요:"
    )

    print(
        "final_test는 앞으로 "
        "모델/threshold 튜닝에 사용하지 않습니다."
    )

    print(
        "V6 개발 중에는 train + val만 사용합니다."
    )


if __name__ == "__main__":
    main()