import asyncio
import json

import pytest
from livekit.agents import APIConnectionError, stt
from livekit.agents.types import APIConnectOptions

from plugins.funasr import stt as funasr_stt


class FakeAudioFrame:
    def __init__(self, data: bytes) -> None:
        self.data = data


class FakeWebSocket:
    def __init__(self, final_text: str | None) -> None:
        self.final_text = final_text
        self.sent: list[str | bytes] = []
        self._end_received = asyncio.Event()
        self._yielded = False

    async def send(self, data: str | bytes) -> None:
        self.sent.append(data)

        if isinstance(data, str) and json.loads(data).get("is_speaking") is False:
            self._end_received.set()

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        await self._end_received.wait()

        if self.final_text is None or self._yielded:
            raise StopAsyncIteration

        self._yielded = True
        return json.dumps({"mode": "2pass-offline", "text": self.final_text})


class FakeConnect:
    def __init__(self, ws: FakeWebSocket) -> None:
        self.ws = ws

    async def __aenter__(self) -> FakeWebSocket:
        return self.ws

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _patch_funasr(
    monkeypatch: pytest.MonkeyPatch, sockets: list[FakeWebSocket]
) -> None:
    monkeypatch.setattr(funasr_stt.rtc, "AudioFrame", FakeAudioFrame)

    def connect(*args, **kwargs) -> FakeConnect:
        return FakeConnect(sockets.pop(0))

    monkeypatch.setattr(funasr_stt.websockets, "connect", connect)


def _new_stream() -> funasr_stt.RecognizeStream:
    return funasr_stt.STT(url="ws://funasr.test").stream(
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0)
    )


def _flush_sentinel() -> stt.RecognizeStream._FlushSentinel:
    return stt.RecognizeStream._FlushSentinel()


async def _collect_events(stream: funasr_stt.RecognizeStream) -> list[stt.SpeechEvent]:
    events = []
    async for event in stream:
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_stream_reopens_funasr_request_after_each_flush(monkeypatch) -> None:
    sockets = [FakeWebSocket("first"), FakeWebSocket("second")]
    _patch_funasr(monkeypatch, sockets)
    stream = _new_stream()

    stream._input_ch.send_nowait(FakeAudioFrame(b"one"))
    stream._input_ch.send_nowait(_flush_sentinel())
    stream._input_ch.send_nowait(FakeAudioFrame(b"two"))
    stream._input_ch.send_nowait(_flush_sentinel())
    stream._input_ch.close()

    try:
        events = await _collect_events(stream)
    finally:
        await stream.aclose()

    transcripts = [
        event.alternatives[0].text
        for event in events
        if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ]

    assert transcripts == ["first", "second"]


@pytest.mark.asyncio
async def test_stream_emits_empty_final_transcript(monkeypatch) -> None:
    _patch_funasr(monkeypatch, [FakeWebSocket("")])
    stream = _new_stream()

    stream._input_ch.send_nowait(FakeAudioFrame(b"audio"))
    stream._input_ch.send_nowait(_flush_sentinel())
    stream._input_ch.close()

    try:
        events = await _collect_events(stream)
    finally:
        await stream.aclose()

    final_events = [
        event for event in events if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ]

    assert len(final_events) == 1
    assert final_events[0].alternatives[0].text == ""


@pytest.mark.asyncio
async def test_stream_raises_when_connection_ends_before_final(monkeypatch) -> None:
    _patch_funasr(monkeypatch, [FakeWebSocket(None)])
    stream = _new_stream()

    stream._input_ch.send_nowait(FakeAudioFrame(b"audio"))
    stream._input_ch.send_nowait(_flush_sentinel())
    stream._input_ch.close()

    try:
        with pytest.raises(APIConnectionError, match="before final transcript"):
            await _collect_events(stream)
    finally:
        await stream.aclose()
