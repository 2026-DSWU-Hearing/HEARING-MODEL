import csv
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from config import (
    MANIFEST_DIR,
    PREPARED_DIR,
    SAMPLE_RATE,
    CLIP_SAMPLES,
    TRAIN_HOP_SAMPLES,
    EVAL_HOP_SAMPLES,
)


# ============================================================
# 경로
# ============================================================

MANIFEST_PATH = MANIFEST_DIR / "combined.csv"


# ============================================================
# 출력 폴더 초기화
# ============================================================

def reset_output_dirs() -> None:
    """
    prepared_v2 폴더 자체는 유지하고 내부 내용만 정리한다.

    OneDrive 환경에서는 최상위 폴더를 shutil.rmtree()로
    통째로 삭제할 때 PermissionError가 발생할 수 있으므로
    하위 폴더들만 초기화한다.
    """

    PREPARED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in ("train", "val", "test"):
        for class_name in ("normal", "emergency"):

            output_dir = (
                PREPARED_DIR
                / split
                / class_name
            )

            # 기존 출력 폴더가 있으면 내용 정리
            if output_dir.exists():
                try:
                    shutil.rmtree(output_dir)

                except PermissionError:
                    print(
                        f"[경고] 폴더 전체 삭제 실패: "
                        f"{output_dir}"
                    )

                    # OneDrive 등이 잡고 있는 경우
                    # 내부 파일부터 최대한 정리
                    for item in output_dir.rglob("*"):
                        try:
                            if item.is_file() or item.is_symlink():
                                item.unlink()
                        except PermissionError:
                            print(
                                f"[경고] 파일 삭제 실패: "
                                f"{item}"
                            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


# ============================================================
# 오디오 로드
# ============================================================

def load_audio(path: Path) -> np.ndarray:
    """
    오디오를 16 kHz mono float32 형태로 로드한다.

    ESC-50 / UrbanSound8K의 원본 샘플레이트가 달라도
    모두 SAMPLE_RATE 기준으로 resampling 된다.
    """

    audio, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    return audio.astype(
        np.float32
    )


# ============================================================
# RMS 계산
# ============================================================

def calculate_rms(
    audio: np.ndarray,
) -> float:
    """
    RMS(Root Mean Square)를 계산한다.

    거의 무음인 1초 조각을 학습 데이터에서 제거하는 데 사용한다.
    """

    if audio.size == 0:
        return 0.0

    return float(
        np.sqrt(
            np.mean(
                np.square(audio)
            )
        )
    )


# ============================================================
# 오디오 분할
# ============================================================

def split_audio(
    audio: np.ndarray,
    hop_samples: int,
) -> list[np.ndarray]:
    """
    오디오를 1초(CLIP_SAMPLES) 단위로 분할한다.

    train:
        TRAIN_HOP_SAMPLES 사용
        현재 설정이 0.5초라면 overlap 발생

    val / test:
        EVAL_HOP_SAMPLES 사용
        보통 1초이므로 overlap 없음

    1초보다 짧은 파일은 zero padding한다.
    """

    clips: list[np.ndarray] = []

    # --------------------------------------------------------
    # 1초보다 짧은 오디오
    # --------------------------------------------------------

    if len(audio) < CLIP_SAMPLES:

        padded = np.pad(
            audio,
            (
                0,
                CLIP_SAMPLES - len(audio),
            ),
            mode="constant",
        )

        # 너무 조용한 파일은 제외
        if calculate_rms(padded) >= 0.002:
            clips.append(
                padded.astype(
                    np.float32
                )
            )

        return clips

    # --------------------------------------------------------
    # 1초 이상 오디오
    # --------------------------------------------------------

    for start in range(
        0,
        len(audio) - CLIP_SAMPLES + 1,
        hop_samples,
    ):

        end = (
            start
            + CLIP_SAMPLES
        )

        clip = audio[
            start:end
        ]

        # 정확히 1초 길이가 아닌 경우 제외
        if len(clip) != CLIP_SAMPLES:
            continue

        # 거의 무음이면 제외
        if calculate_rms(clip) < 0.002:
            continue

        clips.append(
            clip.astype(
                np.float32
            )
        )

    return clips


# ============================================================
# manifest 한 행 처리
# ============================================================

def process_row(
    row: dict,
    row_index: int,
) -> int:
    """
    combined.csv 한 행을 처리한다.

    1. 원본 파일 로드
    2. split 확인
    3. 1초 단위 분할
    4. prepared_v2/{split}/{class}/ 에 저장

    반환값:
        생성된 clip 개수
    """

    split = (
        row["split"]
        .strip()
        .lower()
    )

    if split not in (
        "train",
        "val",
        "test",
    ):
        print(
            f"[잘못된 split] "
            f"{row.get('source_file')}: "
            f"{split}"
        )

        return 0

    try:
        binary_label = int(
            row["binary_label"]
        )

    except (ValueError, TypeError):
        print(
            f"[잘못된 binary_label] "
            f"{row.get('source_file')}: "
            f"{row.get('binary_label')}"
        )

        return 0

    if binary_label == 1:
        class_name = "emergency"

    elif binary_label == 0:
        class_name = "normal"

    else:
        print(
            f"[지원하지 않는 binary_label] "
            f"{binary_label}"
        )

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

    # --------------------------------------------------------
    # 오디오 로드
    # --------------------------------------------------------

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
    # train은 overlap
    # val/test는 non-overlap
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

    # --------------------------------------------------------
    # 출력 경로
    # --------------------------------------------------------

    output_dir = (
        PREPARED_DIR
        / split
        / class_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_name = (
        row.get(
            "dataset",
            "dataset"
        )
        .strip()
        .replace(" ", "_")
    )

    source_stem = (
        source_path.stem
        .replace(" ", "_")
    )

    # --------------------------------------------------------
    # WAV 저장
    # --------------------------------------------------------

    written_count = 0

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

            written_count += 1

        except Exception as error:
            print(
                f"[파일 저장 실패] "
                f"{output_path}: "
                f"{error}"
            )

    return written_count


# ============================================================
# 결과 개수 확인
# ============================================================

def count_files() -> dict:
    """
    train / val / test 각각의
    emergency / normal WAV 개수를 출력한다.
    """

    result = {}

    print()
    print("=" * 70)
    print("전처리 결과")
    print("=" * 70)

    for split in (
        "train",
        "val",
        "test",
    ):

        emergency_dir = (
            PREPARED_DIR
            / split
            / "emergency"
        )

        normal_dir = (
            PREPARED_DIR
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

        total_count = (
            emergency_count
            + normal_count
        )

        result[split] = {
            "emergency":
                emergency_count,

            "normal":
                normal_count,

            "total":
                total_count,
        }

        emergency_ratio = (
            emergency_count
            / total_count
            * 100
            if total_count > 0
            else 0.0
        )

        print(
            f"{split:5s} | "
            f"긴급 {emergency_count:6d} | "
            f"일반 {normal_count:6d} | "
            f"전체 {total_count:6d} | "
            f"긴급 비율 {emergency_ratio:5.1f}%"
        )

    return result


# ============================================================
# main
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # manifest 확인
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"manifest 파일이 없습니다:\n"
            f"{MANIFEST_PATH}\n\n"
            f"먼저 build_manifest.py를 실행하세요."
        )

    print("=" * 70)
    print("Hear:ing Hardware AI - Dataset Preparation v2")
    print("=" * 70)

    print(
        f"manifest: "
        f"{MANIFEST_PATH}"
    )

    print(
        f"출력 경로: "
        f"{PREPARED_DIR}"
    )

    print(
        f"sample rate: "
        f"{SAMPLE_RATE} Hz"
    )

    print(
        f"clip samples: "
        f"{CLIP_SAMPLES}"
    )

    print(
        f"train hop: "
        f"{TRAIN_HOP_SAMPLES} samples"
    )

    print(
        f"eval hop: "
        f"{EVAL_HOP_SAMPLES} samples"
    )

    # --------------------------------------------------------
    # 출력 초기화
    # --------------------------------------------------------

    print()
    print("기존 prepared_v2 데이터 정리 중...")

    reset_output_dirs()

    # --------------------------------------------------------
    # manifest 로드
    # --------------------------------------------------------

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        rows = list(
            csv.DictReader(
                file
            )
        )

    if not rows:
        raise RuntimeError(
            "combined.csv에 데이터가 없습니다."
        )

    print(
        f"원본 파일 수: "
        f"{len(rows)}"
    )

    # --------------------------------------------------------
    # 원본 처리
    # --------------------------------------------------------

    total_clips = 0
    failed_rows = 0

    for row_index, row in enumerate(
        rows,
        start=1,
    ):

        clip_count = process_row(
            row,
            row_index,
        )

        if clip_count == 0:
            failed_rows += 1

        total_clips += (
            clip_count
        )

        if (
            row_index % 250 == 0
            or row_index == len(rows)
        ):

            print(
                f"{row_index:5d}"
                f"/{len(rows)} "
                f"원본 처리 완료 | "
                f"누적 clip={total_clips}"
            )

    # --------------------------------------------------------
    # 결과
    # --------------------------------------------------------

    stats = count_files()

    print()
    print("=" * 70)
    print("완료")
    print("=" * 70)

    print(
        f"원본 파일: "
        f"{len(rows)}개"
    )

    print(
        f"생성 clip: "
        f"{total_clips}개"
    )

    print(
        f"clip 미생성 원본: "
        f"{failed_rows}개"
    )

    # --------------------------------------------------------
    # 최소 sanity check
    # --------------------------------------------------------

    for split in (
        "train",
        "val",
        "test",
    ):

        if stats[split]["emergency"] == 0:
            raise RuntimeError(
                f"{split} emergency 데이터가 "
                f"0개입니다."
            )

        if stats[split]["normal"] == 0:
            raise RuntimeError(
                f"{split} normal 데이터가 "
                f"0개입니다."
            )

    print()
    print(
        "전처리 데이터가 정상적으로 생성되었습니다."
    )

    print(
        "다음 단계:"
    )

    print(
        "py -3.11 scripts\\train_v2.py"
    )


if __name__ == "__main__":
    main()