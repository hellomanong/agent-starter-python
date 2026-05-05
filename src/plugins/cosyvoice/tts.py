"""
CosyVoice2 TTS 插件 —— LiveKit Agents 的语音合成客户端

对接官方服务端
-------------
使用 CosyVoice 官方 FastAPI 服务（https://github.com/FunAudioLLM/CosyVoice）：
    cd CosyVoice
    python runtime/python/fastapi/server.py --model_dir iic/CosyVoice2-0.5B --port 50000

接口：POST /inference_sft，Form 表单，返回 streaming raw PCM int16。

音频格式约定
-----------
服务端输出：22050 Hz、单声道、int16-LE raw PCM（无 WAV 文件头）
本插件中的 _SAMPLE_RATE / _NUM_CHANNELS 必须与服务端输出一致。

HTTP 连接复用
-----------
TTS 实例维护共享的 aiohttp.ClientSession，通过 TCPConnector 实现 keep-alive，
避免每次合成重新 TCP 握手。在 aclose() 中统一释放。

超时策略
-------
- connect : 10s（快速发现服务不可用）
- sock_read / total : None（流式响应持续接收，长文本耗时不可预测）
"""

from __future__ import annotations

import time
import uuid

import aiohttp
import structlog
from livekit.agents import APIConnectionError, APIStatusError, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = structlog.get_logger()

DEFAULT_URL = "http://localhost:50000"
# CosyVoice2 输出 22050 Hz 单声道 int16-LE PCM，与服务端配置必须一致
_SAMPLE_RATE = 22050
_NUM_CHANNELS = 1
# HTTP 连接超时（建立 TCP 连接的最长等待时间）
_CONNECT_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# TTS 主类
# ---------------------------------------------------------------------------


class TTS(tts.TTS):
    """CosyVoice2 TTS 插件入口。

    使用示例
    --------
    tts = TTS(base_url="http://localhost:50000", speaker="中文女")
    session = AgentSession(tts=tts, ...)
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_URL,
        speaker: str = "中文女",
    ) -> None:
        """
        参数
        ----
        base_url : CosyVoice2 官方 FastAPI 服务的根地址
        speaker  : 音色名称（对应官方接口的 spk_id），需与服务端加载的模型支持的音色一致
        """
        super().__init__(
            # streaming=False：使用 synthesize() / ChunkedStream 接口，
            # 而非 stream() / SynthesizeStream（后者用于 WebSocket 流式 TTS）
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=_SAMPLE_RATE,
            num_channels=_NUM_CHANNELS,
        )
        self._base_url = base_url.rstrip("/")
        self._speaker = speaker
        # 共享 HTTP session，在第一次合成时懒加载，在 aclose() 中释放
        self._http_session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return "cosyvoice2"

    @property
    def provider(self) -> str:
        return "cosyvoice"

    def _get_session(self) -> aiohttp.ClientSession:
        """获取（或懒加载）共享 HTTP session。

        使用 TCPConnector 保持 HTTP keep-alive，避免每次合成重新握手。
        必须在异步上下文中调用（即在协程内），确保事件循环已就绪。
        """
        if self._http_session is None or self._http_session.closed:
            connector = aiohttp.TCPConnector(
                keepalive_timeout=30.0,  # 空闲连接保持 30s
                limit=5,                 # 最多 5 个并发连接（单 agent 通常只有 1 个）
            )
            # sock_read=None：不限单次读超时，流式响应需要持续接收数据
            # total=None：不限整体超时，长文本合成耗时不可预测
            timeout = aiohttp.ClientTimeout(connect=_CONNECT_TIMEOUT, sock_read=None, total=None)
            self._http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._http_session

    async def aclose(self) -> None:
        """释放共享 HTTP session（在 AgentSession 关闭时由框架调用）。"""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


# ---------------------------------------------------------------------------
# 分块流式音频流
# ---------------------------------------------------------------------------


class ChunkedStream(tts.ChunkedStream):
    """单次合成请求的 HTTP 流式接收器。

    生命周期
    --------
    1. 基类 _main_task 创建 AudioEmitter 并调用 _run(output_emitter)
    2. _run 向服务端发送 POST 请求，流式接收 PCM 数据
    3. 每收到一个 chunk，调用 output_emitter.push(bytes) 推入 LiveKit 管道
    4. _run 返回后，基类调用 output_emitter.end_input() 通知管道数据结束
    """

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # 初始化音频发射器：
        #   mime_type="audio/pcm" → 告知框架数据是 raw PCM，无需解码
        #   stream=False          → 非流式 TTS，整段合成（无 start/end_segment 调用）
        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=_SAMPLE_RATE,
            num_channels=_NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=False,
        )

        tts_instance: TTS = self._tts  # type: ignore
        t0 = time.perf_counter()
        logger.debug("cosyvoice.synthesis_start", speaker=tts_instance._speaker, text_preview=self._input_text[:50])

        try:
            session = tts_instance._get_session()
            # 官方接口：POST /inference_sft，Form 表单，字段名 tts_text + spk_id
            async with session.post(
                f"{tts_instance._base_url}/inference_sft",
                data={
                    "tts_text": self._input_text,
                    "spk_id": tts_instance._speaker,
                },
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("cosyvoice.http_error", status=resp.status, body=body)
                    raise APIStatusError(
                        f"CosyVoice error: {body}",
                        status_code=resp.status,
                        body=body,
                    )

                # 流式接收 PCM 数据，每 4096 字节（约 93ms 音频）推入一次
                # 框架会把连续的字节流切分为标准帧大小（默认 200ms）再下发
                bytes_received = 0
                async for chunk in resp.content.iter_chunked(4096):
                    if chunk:
                        output_emitter.push(chunk)
                        bytes_received += len(chunk)

                elapsed = (time.perf_counter() - t0) * 1000
                # int16 每个样本 2 字节，audio_duration = total_bytes / (2 * sr * ch)
                audio_duration = bytes_received / (2 * _SAMPLE_RATE * _NUM_CHANNELS)

                if bytes_received == 0:
                    # 服务端返回 200 但没有输出任何音频，基类会随后抛出
                    # "no audio frames were pushed" 错误，这里提前记录以便定位
                    logger.warning(
                        "cosyvoice.empty_audio",
                        text_preview=self._input_text[:80],
                    )
                else:
                    # RTF（Real-Time Factor）= 合成耗时 / 音频时长，越低越好
                    rtf = (elapsed / 1000) / max(audio_duration, 1e-6)
                    logger.debug(
                        "cosyvoice.synthesis_done",
                        elapsed_ms=round(elapsed, 1),
                        audio_s=round(audio_duration, 2),
                        rtf=round(rtf, 2),
                    )

        except aiohttp.ClientConnectionError as e:
            raise APIConnectionError(f"CosyVoice connection error: {e}") from e
