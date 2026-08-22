import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from config import PROJECT_ROOT

DATASET_DIR = PROJECT_ROOT / "dataset"
DOWNLOAD_DIR = DATASET_DIR / "downloads"
ESC50_DIR = DATASET_DIR / "esc50"

ESC50_URL = (
    "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
)


def download_file(url: str, output_path: Path) -> None:
    if output_path.exists():
        print(f"[이미 존재] {output_path}")
        return

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as file:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=output_path.name,
            ) as progress:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue

                    file.write(chunk)
                    progress.update(len(chunk))


def main() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ESC50_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = DOWNLOAD_DIR / "ESC-50-master.zip"

    print("ESC-50 다운로드 중...")
    download_file(ESC50_URL, zip_path)

    temp_dir = DATASET_DIR / "_esc50_temp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True)

    print("압축 해제 중...")

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(temp_dir)

    extracted_root = temp_dir / "ESC-50-master"

    if not extracted_root.exists():
        raise FileNotFoundError(
            f"압축 해제 폴더를 찾지 못했습니다: {extracted_root}"
        )

    if ESC50_DIR.exists():
        shutil.rmtree(ESC50_DIR)

    shutil.move(str(extracted_root), str(ESC50_DIR))

    shutil.rmtree(temp_dir)

    print("ESC-50 준비 완료")
    print(f"오디오: {ESC50_DIR / 'audio'}")
    print(f"메타데이터: {ESC50_DIR / 'meta' / 'esc50.csv'}")


if __name__ == "__main__":
    main()