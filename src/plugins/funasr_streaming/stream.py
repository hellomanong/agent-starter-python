from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

import structlog
from livekit import rtc
from livekit.agents import APIConnectionError, stt
from livekit.agents.types import APIConnectOptions

from . import protocol, transport
from .audio import normalize_audio_frame

if TYPE_CHECKING:
    from .client import STT

logger = structlog.get_logger()


class RecognizeStream(stt.RecognizeStream):
    """Realtime FunASR 2pass stream."""

    def __init__(
        self,
        *,
        stt: STT,
        language: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._language = language
        self._eos_emitted = False

    async def _run(self) -> None:
        input_iter = self._input_ch.__aiter__()

        while True:
            first_frame = await self._next_audio_frame(input_iter)
            if first_frame is None:
                return

            await self._recognize_utterance(first_frame, input_iter)

    def _emit_end_of_speech_once(self) -> None:
        if self._eos_emitted:
            return
        self._eos_emitted = True
        self._event_ch.send_nowait(
            stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
        )

    async def _recognize_utterance(
        self, first_frame: rtc.AudioFrame, input_iter
    ) -> None:
        stt_instance: STT = self._stt  # type: ignore[assignment]
        request_id = str(uuid.uuid4())
        self._eos_emitted = False
        t0 = time.perf_counter()

        try:
            async with transport.connect(
                stt_instance._url,
                open_timeout=self._conn_options.timeout,
            ) as ws:
                logger.debug(
                    "funasr_streaming.stream_connected",
                    elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
                )

                await self._send_stream_config(ws)
                recv_task = asyncio.create_task(
                    self._recv_loop(ws, t_conn=t0, request_id=request_id)
                )

                try:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    )
                    audio_duration = await self._send_audio_frame(ws, first_frame)
                    audio_duration += await self._send_until_utterance_end(
                        ws, input_iter, recv_task
                    )
                    await asyncio.wait_for(
                        recv_task, timeout=self._conn_options.timeout
                    )
                    self._emit_recognition_usage(request_id, audio_duration)

                except asyncio.TimeoutError as e:
                    raise APIConnectionError(
                        "FunASR timed out waiting for final transcript "
                        f"(timeout={self._conn_options.timeout}s)"
                    ) from e
                finally:
                    if not recv_task.done():
                        recv_task.cancel()
                        await asyncio.gather(recv_task, return_exceptions=True)
        except (transport.WebSocketException, OSError) as e:
            raise APIConnectionError(f"FunASR connection error: {e}") from e

    async def _next_input(self, input_iter):
        try:
            return await input_iter.__anext__()
        except StopAsyncIteration:
            return None

    async def _next_audio_frame(self, input_iter) -> rtc.AudioFrame | None:
        while True:
            data = await self._next_input(input_iter)
            if data is None:
                return None
            if isinstance(data, stt.RecognizeStream._FlushSentinel):
                continue
            if isinstance(data, rtc.AudioFrame):
                return data

    async def _send_stream_config(self, ws: transport.WebSocketClientProtocol) -> None:
        stt_instance: STT = self._stt  # type: ignore[assignment]
        await ws.send(
            protocol.dumps(
                protocol.stream_config(
                    language=self._language,
                    chunk_size=stt_instance._stream_chunk_size,
                    chunk_interval=stt_instance._stream_chunk_interval,
                    enable_itn=stt_instance._enable_itn,
                )
            )
        )

    async def _send_audio_frame(
        self, ws: transport.WebSocketClientProtocol, frame: rtc.AudioFrame
    ) -> float:
        frame = normalize_audio_frame(frame)
        await ws.send(bytes(frame.data))
        return frame.samples_per_channel / frame.sample_rate

    def _emit_recognition_usage(self, request_id: str, audio_duration: float) -> None:
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=request_id,
                recognition_usage=stt.RecognitionUsage(audio_duration=audio_duration),
            )
        )

    async def _discard_until_flush(self, input_iter) -> None:
        while True:
            try:
                data = await asyncio.wait_for(
                    self._next_input(input_iter),
                    timeout=self._conn_options.timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "funasr_streaming.discard_until_flush_timeout",
                    timeout_s=self._conn_options.timeout,
                )
                return

            if data is None or isinstance(data, stt.RecognizeStream._FlushSentinel):
                return

    async def _send_until_utterance_end(
        self,
        ws: transport.WebSocketClientProtocol,
        input_iter,
        recv_task: asyncio.Task[None],
    ) -> float:
        audio_duration = 0.0

        while True:
            input_task = asyncio.create_task(self._next_input(input_iter))

            try:
                done, _ = await asyncio.wait(
                    {input_task, recv_task},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=self._conn_options.timeout,
                )

                if not done:
                    input_task.cancel()
                    await asyncio.gather(input_task, return_exceptions=True)
                    logger.warning(
                        "funasr_streaming.input_idle_timeout",
                        timeout_s=self._conn_options.timeout,
                    )
                    self._emit_end_of_speech_once()
                    await ws.send(protocol.dumps(protocol.end_signal()))
                    return audio_duration

                if recv_task in done:
                    input_data = None
                    if input_task.done():
                        input_data = input_task.result()
                    else:
                        input_task.cancel()
                        await asyncio.gather(input_task, return_exceptions=True)

                    exc = recv_task.exception()
                    if exc is not None:
                        raise exc

                    if input_data is not None and not isinstance(
                        input_data, stt.RecognizeStream._FlushSentinel
                    ):
                        await self._discard_until_flush(input_iter)

                    return audio_duration

                if input_task in done:
                    data = input_task.result()

                    if data is None or isinstance(
                        data, stt.RecognizeStream._FlushSentinel
                    ):
                        self._emit_end_of_speech_once()
                        await ws.send(protocol.dumps(protocol.end_signal()))
                        return audio_duration

                    if isinstance(data, rtc.AudioFrame):
                        audio_duration += await self._send_audio_frame(ws, data)

            except Exception:
                if not input_task.done():
                    input_task.cancel()
                    await asyncio.gather(input_task, return_exceptions=True)
                raise

    async def _recv_loop(
        self,
        ws: transport.WebSocketClientProtocol,
        *,
        t_conn: float,
        request_id: str,
    ) -> None:
        first_result_logged = False

        try:
            async for msg in ws:
                result = protocol.parse_json_message(msg, context="stream response")

                if "error" in result:
                    raise APIConnectionError(f"FunASR server error: {result['error']}")

                mode = result.get("mode", "")
                if mode not in protocol.STREAM_MODES:
                    raise APIConnectionError(
                        f"FunASR malformed stream response: {result!r}"
                    )

                is_final = protocol.is_final_result(result)
                text = protocol.result_text(result, context="stream response")

                if not text and not is_final:
                    continue

                if not first_result_logged:
                    first_result_logged = True
                    elapsed_ms = (time.perf_counter() - t_conn) * 1000
                    logger.debug(
                        "funasr_streaming.first_result",
                        elapsed_ms=round(elapsed_ms, 1),
                        mode=mode,
                        text=text,
                    )

                event_type = (
                    stt.SpeechEventType.FINAL_TRANSCRIPT
                    if is_final
                    else stt.SpeechEventType.INTERIM_TRANSCRIPT
                )

                if is_final:
                    self._emit_end_of_speech_once()

                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=event_type,
                        request_id=request_id,
                        alternatives=[
                            stt.SpeechData(
                                language=self._language,
                                text=text,
                            )
                        ],
                    )
                )

                if is_final:
                    return

        except transport.ConnectionClosed as e:
            raise APIConnectionError(
                "FunASR stream closed before final transcript"
            ) from e

        raise APIConnectionError("FunASR stream ended before final transcript")
