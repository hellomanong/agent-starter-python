from __future__ import annotations

from dataclasses import dataclass

DEFAULT_URL = "ws://localhost:10095"
DEFAULT_MODEL = "paraformer-zh"
DEFAULT_PROVIDER = "funasr-offline"
DEFAULT_LANGUAGE = "zh"

# FunASR runtime expects 16 kHz mono int16-LE PCM input.
INPUT_SAMPLE_RATE = 16000

ENABLE_ITN = True


@dataclass(frozen=True, slots=True)
class FunASROfflineConfig:
    url: str = DEFAULT_URL
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER
    language: str = DEFAULT_LANGUAGE
    input_sample_rate: int = INPUT_SAMPLE_RATE
    enable_itn: bool = ENABLE_ITN


DEFAULT_CONFIG = FunASROfflineConfig()
