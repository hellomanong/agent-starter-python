from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

import aiohttp
import structlog
from livekit.agents import APIConnectionError, APIStatusError, tts

from .protocol import is_non_audio_content_type

if TYPE_CHECKING:
    from .client import TTS

logger = structlog.get_logger()


class ChunkedStream(tts.ChunkedStream):
    """Single CosyVoice HTTP request that streams raw PCM into LiveKit."""

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        tts_instance: TTS = self._tts  # type: ignore[assignment]
        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=tts_instance.sample_rate,
            num_channels=tts_instance.num_channels,
            mime_type="audio/pcm",
            stream=False,
        )

        if not self._input_text.strip():
            logger.debug("cosyvoice.synthesis_skipped", reason="blank_text")
            return

        t0 = time.perf_counter()
        ttfb_ms: float | None = None
        bytes_received = 0
        logger.debug(
            "cosyvoice.synthesis_start",
            speaker=tts_instance._speaker,
            text_chars=len(self._input_text),
        )

        try:
            session = tts_instance._get_session()
            async with session.post(
                f"{tts_instance._base_url}/inference_sft",
                data={
                    "tts_text": self._input_text,
                    "spk_id": tts_instance._speaker,
                },
                timeout=tts_instance._request_timeout(self._conn_options),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("cosyvoice.http_error", status=resp.status, body=body)
                    raise APIStatusError(
                        f"CosyVoice error: {body}",
                        status_code=resp.status,
                        body=body,
                    )

                if is_non_audio_content_type(resp.content_type):
                    body = await resp.text()
                    logger.error(
                        "cosyvoice.unexpected_content_type",
                        content_type=resp.content_type,
                        body=body,
                    )
                    raise APIStatusError(
                        f"CosyVoice returned non-audio response: {body}",
                        status_code=resp.status,
                        body=body,
                    )

                async for chunk in resp.content.iter_any():
                    if chunk:
                        if ttfb_ms is None:
                            ttfb_ms = (time.perf_counter() - t0) * 1000
                        output_emitter.push(chunk)
                        bytes_received += len(chunk)

                elapsed = (time.perf_counter() - t0) * 1000
                audio_duration = bytes_received / (
                    2 * tts_instance.sample_rate * tts_instance.num_channels
                )

                if bytes_received == 0:
                    raise APIStatusError(
                        f"CosyVoice returned empty audio "
                        f"(speaker={tts_instance._speaker!r}, "
                        f"text={self._input_text[:80]!r})",
                        status_code=200,
                        body="",
                    )

                rtf = (elapsed / 1000) / max(audio_duration, 1e-6)
                logger.debug(
                    "cosyvoice.synthesis_done",
                    ttfb_ms=round(ttfb_ms, 1) if ttfb_ms is not None else None,
                    elapsed_ms=round(elapsed, 1),
                    audio_s=round(audio_duration, 2),
                    rtf=round(rtf, 2),
                )

        except asyncio.CancelledError:
            raise
        except (
            aiohttp.ClientConnectionError,
            aiohttp.ClientPayloadError,
            asyncio.TimeoutError,
        ) as e:
            logger.error(
                "cosyvoice.stream_interrupted",
                bytes_received=bytes_received,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
                retryable=bytes_received == 0,
                error=str(e),
            )
            raise APIConnectionError(
                f"CosyVoice connection error: {e}",
                retryable=bytes_received == 0,
            ) from e
