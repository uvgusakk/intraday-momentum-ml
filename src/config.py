"""Configuration and environment loading utilities."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# Validated strategy defaults for the current recommended ML overlay.
DEFAULT_DECISION_FREQ_MINS = 30
DEFAULT_SCOREFORWARD_LABEL_MODE = "baseline_trade"
DEFAULT_ML_LABEL_HORIZON_MINS = 60
DEFAULT_SOFT_SIZE_FLOOR = 0.15
DEFAULT_SOFT_SIZE_CAP = 2.25
DEFAULT_MM_LOOKBACK_DAYS = 20
DEFAULT_MM_FLOOR = 0.75
DEFAULT_MM_CAP = 1.25
DEFAULT_USE_NEXT_BAR_OPEN = True
DEFAULT_MINUTE_STOP_MONITORING = True
DEFAULT_EXECUTION_SPREAD_BPS = 1.0


@dataclass(frozen=True)
class AppConfig:
    """Application configuration values loaded from environment variables."""

    alpaca_api_key: str
    alpaca_api_secret: str
    alpaca_base_url: str
    alpaca_data_url: str
    data_dir: Path
    tz: str


def _required_env(name: str) -> str:
    """Fetch a required environment variable and validate it is non-empty."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def load_config() -> AppConfig:
    """Load and validate configuration from `.env` and process environment.

    Returns:
        AppConfig: Parsed application configuration.
    """
    load_dotenv()

    alpaca_api_key = _required_env("ALPACA_API_KEY")
    alpaca_api_secret = _required_env("ALPACA_API_SECRET")
    alpaca_base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").strip()
    alpaca_data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets").strip()
    data_dir = Path(os.getenv("DATA_DIR", "./data")).expanduser()
    tz = os.getenv("TZ", "America/New_York").strip()

    return AppConfig(
        alpaca_api_key=alpaca_api_key,
        alpaca_api_secret=alpaca_api_secret,
        alpaca_base_url=alpaca_base_url,
        alpaca_data_url=alpaca_data_url,
        data_dir=data_dir,
        tz=tz,
    )
