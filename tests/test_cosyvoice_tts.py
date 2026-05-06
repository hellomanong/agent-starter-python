import aiohttp
import pytest
from livekit.agents import APIConnectionError, APIStatusError
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from plugins.cosyvoice import tts as cosyvoice_tts


class FakeEmitter:
    def __init__(self) -> None:
        self.initialized = False
        self.initialized_with: dict[str, object] = {}
        self.pushed: list[bytes] = []

    def initialize(self, **kwargs) -> None:
        self.initialized = True
        self.initialized_with = kwargs

    def push(self, data: bytes) -> None:
        self.pushed.append(data)


class FakeContent:
    def __init__(
        self, chunks: list[bytes] | None = None, error: Exception | None = None
    ) -> None:
        self._chunks = chunks or []
        self._error = error

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk

        if self._error:
            raise self._error


class FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        chunks: list[bytes] | None = None,
        body: str = "",
        content_type: str = "application/octet-stream",
        error: Exception | None = None,
    ) -> None:
        self.status = status
        self.content = FakeContent(chunks, error)
        self.content_type = content_type
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def text(self) -> str:
        return self._body


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.post_url: str | None = None
        self.post_kwargs: dict[str, object] = {}

    def post(self, url: str, **kwargs):
        self.post_url = url
        self.post_kwargs = kwargs
        return self.response


def _stream(
    tts_instance: cosyvoice_tts.TTS,
    *,
    text: str = "你好",
    conn_options: APIConnectOptions | None = None,
) -> cosyvoice_tts.ChunkedStream:
    stream = object.__new__(cosyvoice_tts.ChunkedStream)
    stream._tts = tts_instance
    stream._input_text = text
    stream._conn_options = conn_options or APIConnectOptions(
        max_retry=0, retry_interval=0.0, timeout=1.5
    )
    return stream


@pytest.mark.asyncio
async def test_run_posts_form_data_and_uses_conn_options_timeout(monkeypatch) -> None:
    response = FakeResponse(chunks=[b"\x00\x00" * 2205])
    session = FakeSession(response)
    tts_instance = cosyvoice_tts.TTS(base_url="http://cosy.test/", speaker="中文女")
    monkeypatch.setattr(tts_instance, "_get_session", lambda: session)

    emitter = FakeEmitter()
    await _stream(tts_instance)._run(emitter)

    assert session.post_url == "http://cosy.test/inference_sft"
    assert session.post_kwargs["data"] == {"tts_text": "你好", "spk_id": "中文女"}
    timeout = session.post_kwargs["timeout"]
    assert isinstance(timeout, aiohttp.ClientTimeout)
    assert timeout.connect == 1.5
    assert timeout.sock_read == 5.0
    assert timeout.total is None
    assert emitter.initialized_with["sample_rate"] == 24000
    assert emitter.initialized_with["num_channels"] == 1
    assert emitter.pushed == [b"\x00\x00" * 2205]


@pytest.mark.asyncio
async def test_run_wraps_stream_read_errors_as_connection_errors(monkeypatch) -> None:
    response = FakeResponse(error=aiohttp.ClientPayloadError("broken stream"))
    session = FakeSession(response)
    tts_instance = cosyvoice_tts.TTS()
    monkeypatch.setattr(tts_instance, "_get_session", lambda: session)

    with pytest.raises(APIConnectionError, match="broken stream") as excinfo:
        await _stream(tts_instance)._run(FakeEmitter())

    assert excinfo.value.retryable is True


@pytest.mark.asyncio
async def test_run_marks_stream_errors_after_audio_as_not_retryable(
    monkeypatch,
) -> None:
    response = FakeResponse(
        chunks=[b"\x00\x00" * 1200],
        error=aiohttp.ClientPayloadError("broken after audio"),
    )
    session = FakeSession(response)
    tts_instance = cosyvoice_tts.TTS()
    monkeypatch.setattr(tts_instance, "_get_session", lambda: session)

    emitter = FakeEmitter()
    with pytest.raises(APIConnectionError, match="broken after audio") as excinfo:
        await _stream(tts_instance)._run(emitter)

    assert emitter.pushed == [b"\x00\x00" * 1200]
    assert excinfo.value.retryable is False


@pytest.mark.asyncio
async def test_run_skips_blank_text_without_http_request(monkeypatch) -> None:
    response = FakeResponse(chunks=[b"should not be used"])
    session = FakeSession(response)
    tts_instance = cosyvoice_tts.TTS()
    monkeypatch.setattr(tts_instance, "_get_session", lambda: session)

    emitter = FakeEmitter()
    await _stream(tts_instance, text=" \n\t ")._run(emitter)

    assert session.post_url is None
    assert emitter.initialized is True
    assert emitter.pushed == []


@pytest.mark.asyncio
async def test_run_rejects_non_audio_success_response(monkeypatch) -> None:
    response = FakeResponse(
        body='{"error":"bad speaker"}',
        content_type="application/json",
    )
    session = FakeSession(response)
    tts_instance = cosyvoice_tts.TTS()
    monkeypatch.setattr(tts_instance, "_get_session", lambda: session)

    with pytest.raises(APIStatusError, match="non-audio response"):
        await _stream(tts_instance)._run(FakeEmitter())


def test_tts_allows_audio_format_and_connector_configuration() -> None:
    tts_instance = cosyvoice_tts.TTS(
        sample_rate=24000,
        num_channels=1,
        http_conn_limit=2,
        read_timeout=12.0,
    )

    assert tts_instance.sample_rate == 24000
    assert tts_instance.num_channels == 1
    assert tts_instance._http_conn_limit == 2
    assert tts_instance._read_timeout == 12.0


def test_default_connect_options_do_not_shrink_read_timeout() -> None:
    tts_instance = cosyvoice_tts.TTS(read_timeout=30.0)

    timeout = tts_instance._request_timeout(DEFAULT_API_CONNECT_OPTIONS)

    assert timeout.connect == 10.0
    assert timeout.sock_read == 30.0
