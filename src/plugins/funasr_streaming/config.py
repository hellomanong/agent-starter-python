from __future__ import annotations

from dataclasses import dataclass

DEFAULT_URL = "ws://localhost:10095"
DEFAULT_MODEL = "paraformer-zh"
DEFAULT_PROVIDER = "funasr_streaming"
DEFAULT_LANGUAGE = "zh"

# FunASR runtime expects 16 kHz mono int16-LE PCM input.
INPUT_SAMPLE_RATE = 16000

# 2pass streaming defaults recommended for realtime conversation.
STREAM_CHUNK_SIZE = (5, 10, 5)
STREAM_CHUNK_INTERVAL = 10
ENABLE_ITN = True


@dataclass(frozen=True, slots=True)
class FunASRConfig:
    url: str = DEFAULT_URL
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER
    language: str = DEFAULT_LANGUAGE
    input_sample_rate: int = INPUT_SAMPLE_RATE
    stream_chunk_size: tuple[int, int, int] = STREAM_CHUNK_SIZE
    stream_chunk_interval: int = STREAM_CHUNK_INTERVAL
    enable_itn: bool = ENABLE_ITN


DEFAULT_CONFIG = FunASRConfig()
