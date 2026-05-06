from __future__ import annotations

from dataclasses import dataclass

DEFAULT_BASE_URL = "http://localhost:50000"
DEFAULT_MODEL = "cosyvoice2"
DEFAULT_PROVIDER = "cosyvoice"
DEFAULT_SPEAKER = "中文女"

# CosyVoice2-0.5B official FastAPI emits 24000 Hz mono int16-LE raw PCM.
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_NUM_CHANNELS = 1

# HTTP connection settings tuned for a single local GPU-backed service.
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 5.0
DEFAULT_HTTP_CONN_LIMIT = 1


@dataclass(frozen=True, slots=True)
class CosyVoiceConfig:
    base_url: str = DEFAULT_BASE_URL
    speaker: str = DEFAULT_SPEAKER
    sample_rate: int = DEFAULT_SAMPLE_RATE
    num_channels: int = DEFAULT_NUM_CHANNELS
    http_conn_limit: int = DEFAULT_HTTP_CONN_LIMIT
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    read_timeout: float | None = DEFAULT_READ_TIMEOUT
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER


DEFAULT_CONFIG = CosyVoiceConfig()
