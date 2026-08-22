import csv
import shutil
from pathlib import Path

from category_map import get_category_info
from config import PROJECT_ROOT
from esc50_mapping import ESC50_TO_YAMNET

ESC50_DIR = PROJECT_ROOT / "dataset" / "esc50"
AUDIO_DIR = ESC50_DIR / "audio"
CSV_PATH = ESC50_DIR / "meta" / "esc50.csv"

SELECTED_DIR = PROJECT_ROOT / "dataset" / "selected"
EMERGENCY_DIR = SELECTED_DIR / "emergency"
NORMAL_DIR = SELECTED_DIR / "normal"


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"ESC-50 메타데이터를 찾지 못했습니다: {CSV_PATH}"
        )

    EMERGENCY_DIR.mkdir(parents=True, exist_ok=True)
    NORMAL_DIR.mkdir(parents=True, exist_ok=True)

    emergency_count = 0
    normal_count = 0
    skipped_count = 0

    with open(
        CSV_PATH,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            filename = row["filename"].strip()
            esc50_label = row["category"].strip()

            yamnet_label = ESC50_TO_YAMNET.get(esc50_label)

            if yamnet_label is None:
                print(f"[매핑 없음] {esc50_label}")
                skipped_count += 1
                continue

            category, block = get_category_info(yamnet_label)

            source_path = AUDIO_DIR / filename

            if not source_path.exists():
                print(f"[파일 없음] {source_path}")
                skipped_count += 1
                continue

            is_emergency = category == "긴급"

            target_dir = (
                EMERGENCY_DIR
                if is_emergency
                else NORMAL_DIR
            )

            target_path = target_dir / filename
            shutil.copy2(source_path, target_path)

            if is_emergency:
                emergency_count += 1
            else:
                normal_count += 1

            print(
                f"{filename}: "
                f"{esc50_label} → {yamnet_label} "
                f"→ {category}/{block}"
            )

    print("\n분류 완료")
    print(f"긴급: {emergency_count}")
    print(f"일반: {normal_count}")
    print(f"건너뜀: {skipped_count}")


if __name__ == "__main__":
    main()