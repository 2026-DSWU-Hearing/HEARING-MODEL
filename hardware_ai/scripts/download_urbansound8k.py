import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from config import PROJECT_ROOT


DATASET_DIR = PROJECT_ROOT / "dataset"
DOWNLOAD_DIR = DATASET_DIR / "downloads"
URBAN_DIR = DATASET_DIR / "urbansound8k"

URL = (
    "https://zenodo.org/records/1203745/files/"
    "UrbanSound8K.tar.gz"
)


def download_file(url: str, path: Path):
    if path.exists():
        print(f"[이미 존재] {path}")
        return

    with requests.get(
        url,
        stream=True,
        timeout=120,
    ) as response:

        response.raise_for_status()

        total = int(
            response.headers.get(
                "content-length",
                0
            )
        )

        with open(path, "wb") as file:
            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=path.name,
            ) as progress:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if not chunk:
                        continue

                    file.write(chunk)
                    progress.update(len(chunk))


def main():
    DOWNLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    archive_path = (
        DOWNLOAD_DIR
        / "UrbanSound8K.tar.gz"
    )

    print("UrbanSound8K 다운로드 중...")

    download_file(
        URL,
        archive_path
    )

    if URBAN_DIR.exists():
        print(
            "UrbanSound8K 폴더가 이미 존재합니다."
        )
        return

    print("압축 해제 중...")

    with tarfile.open(
        archive_path,
        "r:gz"
    ) as archive:

        archive.extractall(
            DATASET_DIR
        )

    extracted = (
        DATASET_DIR
        / "UrbanSound8K"
    )

    if extracted.exists():
        extracted.rename(
            URBAN_DIR
        )

    print("완료")
    print(
        f"데이터셋: {URBAN_DIR}"
    )


if __name__ == "__main__":
    main()