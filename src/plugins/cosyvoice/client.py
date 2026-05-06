from __future__ import annotations

import aiohttp
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .config import DEFAULT_CONFIG, CosyVoiceConfig
from .stream import ChunkedStream


class TTS(tts.TTS):
    """CosyVoice2 TTS plugin entrypoint for LiveKit Agents."""

    def __init__(
        self,
        *,
        config: CosyVoiceConfig = DEFAULT_CONFIG,
        base_url: str | None = None,
        speaker: str | None = None,
        sample_rate: int | None = None,
        num_channels: int | None = None,
        http_conn_limit: int | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
    ) -> None:
        base_url = config.base_url if base_url is None else base_url
        speaker = config.speaker if speaker is None else speaker
        sample_rate = config.sample_rate if sample_rate is None else sample_rate
        num_channels = config.num_channels if num_channels is None else num_channels
        http_conn_limit = (
            config.http_conn_limit if http_conn_limit is None else http_conn_limit
        )
        connect_timeout = (
            config.connect_timeout if connect_timeout is None else connect_timeout
        )
        read_timeout = config.read_timeout if read_timeout is None else read_timeout

        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than 0")
        if num_channels <= 0:
            raise ValueError("num_channels must be greater than 0")
        if http_conn_limit <= 0:
            raise ValueError("http_conn_limit must be greater than 0")
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be greater than 0")
        if read_timeout is not None and read_timeout <= 0:
            raise ValueError("read_timeout must be greater than 0 or None")

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._base_url = base_url.rstrip("/")
        self._speaker = speaker
        self._model = config.model
        self._provider = config.provider
        self._http_conn_limit = http_conn_limit
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._http_session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    def _get_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            connector = aiohttp.TCPConnector(
                keepalive_timeout=30.0,
                limit=self._http_conn_limit,
            )
            timeout = aiohttp.ClientTimeout(
                connect=self._connect_timeout,
                sock_read=self._read_timeout,
                total=None,
            )
            self._http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
        return self._http_session

    def _request_timeout(
        self, conn_options: APIConnectOptions
    ) -> aiohttp.ClientTimeout:
        connect_timeout = self._connect_timeout
        if conn_options.timeout is not None:
            connect_timeout = min(connect_timeout, conn_options.timeout)

        return aiohttp.ClientTimeout(
            connect=connect_timeout,
            sock_read=self._read_timeout,
            total=None,
        )

    async def aclose(self) -> None:
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None
        await super().aclose()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)
