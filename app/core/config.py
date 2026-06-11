import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    ALERT_THRESHOLD: float = float(os.getenv("ALERT_THRESHOLD", "0.7"))
    COOLDOWN_SECONDS: float = float(os.getenv("COOLDOWN_SECONDS", "3.0"))
    BACKEND_URL: str = os.getenv("BACKEND_URL", "")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "")
    DEVICE_ID: str = os.getenv("DEVICE_ID", "")


settings = Settings()
