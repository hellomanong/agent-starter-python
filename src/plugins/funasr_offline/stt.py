from __future__ import annotations

import asyncio
import time
import uuid

import structlog
from livekit import rtc
from livekit.agents import APIConnectionError, stt
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from . import protocol, transport
from .audio import normalize_audio_frame
from .config import DEFAULT_CONFIG, FunASROfflineConfig

logger = structlog.get_logger()


class STT(stt.STT):
    """FunASR non-streaming (offline) STT plugin entrypoint."""

    def __init__(
        self,
        *,
        config: FunASROfflineConfig = DEFAULT_CONFIG,
        url: str | None = None,
        model: str | None = None,
        language: str | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._url = config.url if url is None else url
        self._model = config.model if model is None else model
        self._provider = config.provider
        self._language = config.language if language is None else language
        self._enable_itn = config.enable_itn

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    async def _recognize_impl(
        self,
        buffer: stt.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        lang = language if language is not NOT_GIVEN else self._language
        frames = [buffer] if isinstance(buffer, rtc.AudioFrame) else list(buffer)

        if not frames:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=str(uuid.uuid4()),
                alternatives=[stt.SpeechData(language=lang, text="")],
            )

        combined = normalize_audio_frame(rtc.combine_audio_frames(frames))
        t0 = time.perf_counter()

        try:
            async with transport.connect(
                self._url,
                open_timeout=conn_options.timeout,
            ) as ws:
                logger.debug(
                    "funasr_offline.connected",
                    elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
                )

                await ws.send(
                    protocol.dumps(protocol.offline_config(enable_itn=self._enable_itn))
                )
                await ws.send(bytes(combined.data))
                await ws.send(protocol.dumps(protocol.end_signal()))

                try:
                    msg = await asyncio.wait_for(
                        ws.recv(), timeout=conn_options.timeout
                    )
                except asyncio.TimeoutError as e:
                    raise APIConnectionError(
                        "FunASR offline result timeout "
                        f"(no response within {conn_options.timeout}s)"
                    ) from e

                result = protocol.parse_json_message(msg, context="offline response")

                if "error" in result:
                    raise APIConnectionError(f"FunASR server error: {result['error']}")

                if result.get("mode") != protocol.OFFLINE_MODE:
                    raise APIConnectionError(
                        f"FunASR malformed offline response: {result!r}"
                    )

                text = protocol.result_text(result, context="offline response")

        except (transport.WebSocketException, OSError) as e:
            raise APIConnectionError(f"FunASR connection error: {e}") from e

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(
            "funasr_offline.recognized", elapsed_ms=round(elapsed, 1), text=text
        )

        if not text:
            logger.warning(
                "funasr_offline.empty_result", audio_s=round(combined.duration, 1)
            )

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=str(uuid.uuid4()),
            alternatives=[stt.SpeechData(language=lang, text=text)],
        )
