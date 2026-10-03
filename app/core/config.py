import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    BACKEND_URL: str = os.getenv("BACKEND_URL", "")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "")
    DEVICE_ID: str = os.getenv("DEVICE_ID", "")

    # 알림 정책 (코드로 관리)
    ALERT_THRESHOLD: float = 0.7
    COOLDOWN_SECONDS: float = 3.0


settings = Settings()
