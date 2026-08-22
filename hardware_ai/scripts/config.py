from pathlib import Path


# ============================================================
# 프로젝트 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

ESC50_DIR = DATASET_DIR / "esc50"
URBANSOUND_DIR = DATASET_DIR / "UrbanSound8K"

MANIFEST_DIR = DATASET_DIR / "manifests"
PREPARED_DIR = DATASET_DIR / "prepared_v2"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ============================================================
# 오디오 설정
# ============================================================

# ESP32 / 서버 환경과 맞추기 위해 16kHz
SAMPLE_RATE = 16_000

# 하드웨어에서 1초 단위로 판별
CLIP_SECONDS = 1.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

# train 데이터에서는 0.5초 간격으로 이동하며 1초 window 생성
# → 한 원본에서 더 많은 학습 샘플 확보
TRAIN_HOP_SECONDS = 0.5
TRAIN_HOP_SAMPLES = int(SAMPLE_RATE * TRAIN_HOP_SECONDS)

# validation / test는 중복 없는 1초 단위
EVAL_HOP_SAMPLES = CLIP_SAMPLES


# ============================================================
# Log-Mel Spectrogram
# ============================================================

FRAME_LENGTH = 400       # 25ms
FRAME_STEP = 160         # 10ms
FFT_LENGTH = 512

NUM_MEL_BINS = 40

LOWER_FREQUENCY = 125.0
UPPER_FREQUENCY = 7500.0


# ============================================================
# 학습
# ============================================================

RANDOM_SEED = 42

BATCH_SIZE = 32
EPOCHS = 60

LEARNING_RATE = 0.001

# 긴급 누락을 더 강하게 패널티
EMERGENCY_CLASS_WEIGHT = 1.5


# ============================================================
# 평가
# ============================================================

DEFAULT_THRESHOLD = 0.5