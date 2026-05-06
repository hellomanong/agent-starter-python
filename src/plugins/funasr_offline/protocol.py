from __future__ import annotations

import json

from livekit.agents import APIConnectionError

from .config import INPUT_SAMPLE_RATE

OFFLINE_MODE = "offline"


def dumps(data: dict) -> str:
    return json.dumps(data)


def parse_json_message(msg: str | bytes, *, context: str) -> dict:
    try:
        result = json.loads(msg)
    except json.JSONDecodeError as e:
        raise APIConnectionError(f"FunASR returned non-JSON {context}: {msg!r}") from e

    if not isinstance(result, dict):
        raise APIConnectionError(f"FunASR returned malformed {context}: {result!r}")

    return result


def result_text(result: dict, *, context: str) -> str:
    if "text" not in result:
        raise APIConnectionError(f"FunASR {context} missing text: {result!r}")

    text = result["text"]
    if not isinstance(text, str):
        raise APIConnectionError(f"FunASR {context} text must be a string: {result!r}")

    return text.strip()


def offline_config(*, enable_itn: bool) -> dict:
    return {
        "mode": "offline",
        "wav_format": "pcm",
        "wav_name": "recognize",
        "is_speaking": True,
        "audio_fs": INPUT_SAMPLE_RATE,
        "itn": enable_itn,
    }


def end_signal() -> dict:
    return {"is_speaking": False}
