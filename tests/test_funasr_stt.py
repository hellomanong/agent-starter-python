import asyncio
import json
from array import array

import pytest
from livekit import rtc
from livekit.agents import APIConnectionError, stt
from livekit.agents.types import APIConnectOptions

from plugins.funasr_streaming import stt as funasr_stt


class FakeAudioFrame:
    def __init__(
        self,
        data: bytes,
        sample_rate: int = 16000,
        num_channels: int = 1,
        samples_per_channel: int | None = None,
    ) -> None:
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = (
            samples_per_channel
            if samples_per_channel is not None
            else len(data) // 2 // num_channels
        )


class FakeWebSocket:
    def __init__(
        self,
        final_text: str | None = None,
        *,
        messages: list[str] | None = None,
        wait_for_end: bool = True,
    ) -> None:
        self.sent: list[str | bytes] = []
        self._end_received = asyncio.Event()
        self._messages = (
            messages
            if messages is not None
            else (
                []
                if final_text is None
                else [json.dumps({"mode": "2pass-offline", "text": final_text})]
            )
        )
        self._message_idx = 0
        self._wait_for_end = wait_for_end

    async def send(self, data: str | bytes) -> None:
        self.sent.append(data)

        if isinstance(data, str) and json.loads(data).get("is_speaking") is False:
            self._end_received.set()

    async def recv(self) -> str:
        if self._wait_for_end:
            await self._end_received.wait()

        if self._message_idx >= len(self._messages):
            await asyncio.Future()

        message = self._messages[self._message_idx]
        self._message_idx += 1
        return message

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self._wait_for_end:
            await self._end_received.wait()

        if self._message_idx >= len(self._messages):
            raise StopAsyncIteration

        message = self._messages[self._message_idx]
        self._message_idx += 1
        return message


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

    _patch_connect(monkeypatch, sockets)


def _patch_connect(
    monkeypatch: pytest.MonkeyPatch, sockets: list[FakeWebSocket]
) -> None:

    def connect(*args, **kwargs) -> FakeConnect:
        return FakeConnect(sockets.pop(0))

    monkeypatch.setattr(funasr_stt.websockets, "connect", connect)


def _new_stream() -> funasr_stt.RecognizeStream:
    return funasr_stt.STT(url="ws://funasr_streaming.test").stream(
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0)
    )


def _flush_sentinel() -> stt.RecognizeStream._FlushSentinel:
    return stt.RecognizeStream._FlushSentinel()


def _pcm(samples: list[int]) -> bytes:
    return array("h", samples).tobytes()


def _samples(data: bytes) -> list[int]:
    samples = array("h")
    samples.frombytes(data)
    return list(samples)


def _real_audio_frame(
    samples: list[int], *, sample_rate: int = 16000, num_channels: int = 1
) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=_pcm(samples),
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=len(samples) // num_channels,
    )


async def _collect_events(stream: funasr_stt.RecognizeStream) -> list[stt.SpeechEvent]:
    events = []
    async for event in stream:
        events.append(event)
    return events


async def _collect_until_usage(
    stream: funasr_stt.RecognizeStream,
) -> list[stt.SpeechEvent]:
    events = []
    while True:
        event = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
        events.append(event)
        if event.type == stt.SpeechEventType.RECOGNITION_USAGE:
            return events


@pytest.mark.asyncio
async def test_recognize_downmixes_offline_audio_to_mono(monkeypatch) -> None:
    ws = FakeWebSocket("ok")
    _patch_connect(monkeypatch, [ws])
    frame = _real_audio_frame([1000, 3000, -1000, 1000], num_channels=2)
    recognizer = funasr_stt.STT(url="ws://funasr_streaming.test")

    event = await recognizer.recognize(
        frame,
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
    )

    audio_payloads = [payload for payload in ws.sent if isinstance(payload, bytes)]
    assert [_samples(payload) for payload in audio_payloads] == [[2000, 0]]
    assert event.alternatives[0].text == "ok"
    assert event.alternatives[0].confidence == 0.0


@pytest.mark.asyncio
async def test_stream_reopens_funasr_request_after_each_flush(monkeypatch) -> None:
    sockets = [FakeWebSocket("first"), FakeWebSocket("second")]
    _patch_funasr(monkeypatch, sockets)
    stream = _new_stream()

    stream._input_ch.send_nowait(FakeAudioFrame(_pcm([1])))
    stream._input_ch.send_nowait(_flush_sentinel())
    stream._input_ch.send_nowait(FakeAudioFrame(_pcm([2])))
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

    stream._input_ch.send_nowait(FakeAudioFrame(_pcm([1])))
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

    stream._input_ch.send_nowait(FakeAudioFrame(_pcm([1])))
    stream._input_ch.send_nowait(_flush_sentinel())
    stream._input_ch.close()

    try:
        with pytest.raises(APIConnectionError, match="before final transcript"):
            await _collect_events(stream)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_stream_downmixes_audio_before_sending(monkeypatch) -> None:
    ws = FakeWebSocket("ok")
    _patch_funasr(monkeypatch, [ws])
    stream = _new_stream()

    stream._input_ch.send_nowait(
        FakeAudioFrame(_pcm([1000, 3000, -1000, 1000]), num_channels=2)
    )
    stream._input_ch.send_nowait(_flush_sentinel())
    stream._input_ch.close()

    try:
        events = await _collect_events(stream)
    finally:
        await stream.aclose()

    audio_payloads = [payload for payload in ws.sent if isinstance(payload, bytes)]
    usage_events = [
        event for event in events if event.type == stt.SpeechEventType.RECOGNITION_USAGE
    ]
    final_events = [
        event for event in events if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ]

    assert [_samples(payload) for payload in audio_payloads] == [[2000, 0]]
    assert final_events[0].alternatives[0].confidence == 0.0
    assert usage_events[0].recognition_usage.audio_duration == pytest.approx(2 / 16000)


@pytest.mark.asyncio
async def test_stream_raises_on_malformed_response(monkeypatch) -> None:
    _patch_funasr(
        monkeypatch,
        [FakeWebSocket(messages=["not-json"], wait_for_end=False)],
    )
    stream = _new_stream()

    stream._input_ch.send_nowait(FakeAudioFrame(_pcm([1])))
    stream._input_ch.send_nowait(_flush_sentinel())
    stream._input_ch.close()

    try:
        with pytest.raises(APIConnectionError, match="non-JSON"):
            await _collect_events(stream)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_stream_discards_frames_until_flush_after_early_final(
    monkeypatch,
) -> None:
    first_ws = FakeWebSocket("early", wait_for_end=False)
    second_ws = FakeWebSocket("next")
    _patch_funasr(monkeypatch, [first_ws, second_ws])
    stream = _new_stream()

    stream._input_ch.send_nowait(FakeAudioFrame(_pcm([1])))
    stream._input_ch.send_nowait(FakeAudioFrame(_pcm([2])))
    stream._input_ch.send_nowait(_flush_sentinel())
    stream._input_ch.send_nowait(FakeAudioFrame(_pcm([3])))
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
    first_audio_payloads = [
        payload for payload in first_ws.sent if isinstance(payload, bytes)
    ]
    second_audio_payloads = [
        payload for payload in second_ws.sent if isinstance(payload, bytes)
    ]

    assert transcripts == ["early", "next"]
    assert [_samples(payload) for payload in first_audio_payloads] == [[1]]
    assert [_samples(payload) for payload in second_audio_payloads] == [[3]]


@pytest.mark.asyncio
async def test_stream_disables_retries_that_cannot_replay_audio() -> None:
    stream = funasr_stt.STT(url="ws://funasr_streaming.test").stream(
        conn_options=APIConnectOptions(max_retry=3, timeout=1.0)
    )

    try:
        assert stream._conn_options.max_retry == 0
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_stream_idle_timeout_ends_utterance_without_flush(monkeypatch) -> None:
    ws = FakeWebSocket("idle")
    _patch_funasr(monkeypatch, [ws])
    stream = funasr_stt.STT(url="ws://funasr_streaming.test").stream(
        conn_options=APIConnectOptions(max_retry=0, timeout=0.01)
    )

    stream._input_ch.send_nowait(FakeAudioFrame(_pcm([1])))

    try:
        events = await _collect_until_usage(stream)
    finally:
        await stream.aclose()

    transcripts = [
        event.alternatives[0].text
        for event in events
        if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ]
    end_signals = [
        json.loads(payload)
        for payload in ws.sent
        if isinstance(payload, str) and json.loads(payload).get("is_speaking") is False
    ]

    assert transcripts == ["idle"]
    assert end_signals == [{"is_speaking": False}]


@pytest.mark.asyncio
async def test_stream_early_final_discard_times_out_without_flush(monkeypatch) -> None:
    ws = FakeWebSocket("early", wait_for_end=False)
    _patch_funasr(monkeypatch, [ws])
    stream = funasr_stt.STT(url="ws://funasr_streaming.test").stream(
        conn_options=APIConnectOptions(max_retry=0, timeout=0.01)
    )

    stream._input_ch.send_nowait(FakeAudioFrame(_pcm([1])))
    stream._input_ch.send_nowait(FakeAudioFrame(_pcm([2])))

    try:
        events = await _collect_until_usage(stream)
    finally:
        await stream.aclose()

    transcripts = [
        event.alternatives[0].text
        for event in events
        if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ]
    audio_payloads = [payload for payload in ws.sent if isinstance(payload, bytes)]

    assert transcripts == ["early"]
    assert [_samples(payload) for payload in audio_payloads] == [[1]]
