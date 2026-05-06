import asyncio
import json
from array import array

import pytest
from livekit import rtc
from livekit.agents import APIConnectionError
from livekit.agents.types import APIConnectOptions

from plugins.funasr_offline import stt as funasr_offline_stt


class FakeWebSocket:
    """Minimal async WS double for the offline request/response flow."""

    def __init__(
        self,
        *,
        messages: list[str] | None = None,
        recv_delay: float = 0.0,
    ) -> None:
        self.sent: list[str | bytes] = []
        self._messages = list(messages) if messages is not None else []
        self._idx = 0
        self._recv_delay = recv_delay

    async def send(self, data: str | bytes) -> None:
        self.sent.append(data)

    async def recv(self) -> str:
        if self._recv_delay:
            await asyncio.sleep(self._recv_delay)

        if self._idx >= len(self._messages):
            await asyncio.Future()  # block forever to trigger caller timeout

        message = self._messages[self._idx]
        self._idx += 1
        return message


class FakeConnect:
    def __init__(self, ws: FakeWebSocket) -> None:
        self.ws = ws

    async def __aenter__(self) -> FakeWebSocket:
        return self.ws

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _patch_connect(
    monkeypatch: pytest.MonkeyPatch, sockets: list[FakeWebSocket]
) -> None:
    def connect(*args, **kwargs) -> FakeConnect:
        return FakeConnect(sockets.pop(0))

    monkeypatch.setattr(funasr_offline_stt.transport, "connect", connect)


def _pcm(samples: list[int]) -> bytes:
    return array("h", samples).tobytes()


def _samples(data: bytes) -> list[int]:
    arr = array("h")
    arr.frombytes(data)
    return list(arr)


def _frame(
    samples: list[int],
    *,
    sample_rate: int = 16000,
    num_channels: int = 1,
) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=_pcm(samples),
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=len(samples) // num_channels,
    )


def _final_msg(text: str, mode: str = "offline") -> str:
    return json.dumps({"mode": mode, "text": text})


def _new_stt() -> funasr_offline_stt.STT:
    return funasr_offline_stt.STT(url="ws://funasr-offline.test")


@pytest.mark.asyncio
async def test_recognize_returns_final_text(monkeypatch) -> None:
    ws = FakeWebSocket(messages=[_final_msg("hello world")])
    _patch_connect(monkeypatch, [ws])

    event = await _new_stt().recognize(
        _frame([0, 1, 2, 3]),
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
    )

    assert event.alternatives[0].text == "hello world"

    json_payloads = [json.loads(p) for p in ws.sent if isinstance(p, str)]
    assert json_payloads[0]["mode"] == "offline"
    assert json_payloads[0]["audio_fs"] == 16000
    assert json_payloads[-1] == {"is_speaking": False}


@pytest.mark.asyncio
async def test_recognize_rejects_2pass_offline_mode(monkeypatch) -> None:
    # We never request 2pass; if the server somehow responds with a 2pass result,
    # treat it as malformed rather than silently accepting.
    ws = FakeWebSocket(messages=[_final_msg("hi", mode="2pass-offline")])
    _patch_connect(monkeypatch, [ws])

    with pytest.raises(APIConnectionError, match="malformed offline response"):
        await _new_stt().recognize(
            _frame([0, 1]),
            conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
        )


@pytest.mark.asyncio
async def test_recognize_downmixes_stereo_to_mono(monkeypatch) -> None:
    ws = FakeWebSocket(messages=[_final_msg("ok")])
    _patch_connect(monkeypatch, [ws])

    await _new_stt().recognize(
        _frame([1000, 3000, -1000, 1000], num_channels=2),
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
    )

    audio_payloads = [p for p in ws.sent if isinstance(p, bytes)]
    assert [_samples(p) for p in audio_payloads] == [[2000, 0]]


@pytest.mark.asyncio
async def test_recognize_resamples_to_16k(monkeypatch) -> None:
    ws = FakeWebSocket(messages=[_final_msg("ok")])
    _patch_connect(monkeypatch, [ws])

    # 8 kHz input — 16 samples = 2 ms; resampler should produce ~32 samples at 16 kHz.
    await _new_stt().recognize(
        _frame([0] * 16, sample_rate=8000),
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
    )

    audio_payloads = [p for p in ws.sent if isinstance(p, bytes)]
    assert audio_payloads, "expected at least one audio payload"
    total_samples = sum(len(p) // 2 for p in audio_payloads)
    # 8k -> 16k doubles sample count; allow some slack from the resampler tail.
    assert total_samples >= 24


@pytest.mark.asyncio
async def test_recognize_handles_empty_buffer_without_connecting(monkeypatch) -> None:
    sockets: list[FakeWebSocket] = []
    _patch_connect(monkeypatch, sockets)

    event = await _new_stt().recognize(
        [],
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
    )

    assert event.alternatives[0].text == ""
    # No socket should have been popped — empty buffer must short-circuit.
    assert sockets == []


@pytest.mark.asyncio
async def test_recognize_combines_multiple_frames(monkeypatch) -> None:
    ws = FakeWebSocket(messages=[_final_msg("merged")])
    _patch_connect(monkeypatch, [ws])

    frames = [_frame([1, 2]), _frame([3, 4]), _frame([5, 6])]
    await _new_stt().recognize(
        frames,
        conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
    )

    audio_payloads = [p for p in ws.sent if isinstance(p, bytes)]
    flattened: list[int] = []
    for p in audio_payloads:
        flattened.extend(_samples(p))
    assert flattened == [1, 2, 3, 4, 5, 6]


@pytest.mark.asyncio
async def test_recognize_raises_on_server_error(monkeypatch) -> None:
    ws = FakeWebSocket(messages=[json.dumps({"error": "boom"})])
    _patch_connect(monkeypatch, [ws])

    with pytest.raises(APIConnectionError, match="boom"):
        await _new_stt().recognize(
            _frame([1, 2]),
            conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
        )


@pytest.mark.asyncio
async def test_recognize_raises_on_malformed_response(monkeypatch) -> None:
    ws = FakeWebSocket(messages=["not-json"])
    _patch_connect(monkeypatch, [ws])

    with pytest.raises(APIConnectionError, match="non-JSON"):
        await _new_stt().recognize(
            _frame([1, 2]),
            conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
        )


@pytest.mark.asyncio
async def test_recognize_raises_on_unexpected_mode(monkeypatch) -> None:
    ws = FakeWebSocket(
        messages=[json.dumps({"mode": "2pass-online", "text": "interim"})]
    )
    _patch_connect(monkeypatch, [ws])

    with pytest.raises(APIConnectionError, match="malformed offline response"):
        await _new_stt().recognize(
            _frame([1, 2]),
            conn_options=APIConnectOptions(max_retry=0, timeout=1.0),
        )


@pytest.mark.asyncio
async def test_recognize_raises_on_timeout(monkeypatch) -> None:
    ws = FakeWebSocket(messages=[])  # recv blocks forever
    _patch_connect(monkeypatch, [ws])

    with pytest.raises(APIConnectionError, match="timeout"):
        await _new_stt().recognize(
            _frame([1, 2]),
            conn_options=APIConnectOptions(max_retry=0, timeout=0.05),
        )


@pytest.mark.asyncio
async def test_capabilities_disable_streaming() -> None:
    s = _new_stt()
    assert s.capabilities.streaming is False
