"""
FunASR STT 插件 —— LiveKit Agents 的语音识别客户端

架构说明
--------
本插件不直接加载模型，而是作为 WebSocket 客户端连接到独立运行的
FunASR 服务进程（默认监听 ws://localhost:10095）。

FunASR 服务负责：加载 paraformer 等模型、GPU 推理、返回识别结果。
本插件负责：音频格式转换、WebSocket 协议收发、事件封装。

服务端模型搭配（实时对话推荐）
-----------------------------
启动 FunASR 服务时同时加载 offline + online 两个引擎，2pass 才完整：
  --asr-model        paraformer-zh           （离线精校）
  --asr-model-online paraformer-zh-streaming （流式快速结果）
  --punc-model       ct-punc

两种工作模式
-----------
1. 离线识别（recognize / _recognize_impl）
   用于一次性识别一段完整音频。
   协议：发送 config → 发送全部 PCM → 发送结束信号 → 接收一条结果。

2. 流式识别（stream / RecognizeStream）
   用于实时对话，边说边识别。
   协议使用 FunASR "2pass" 模式：
     - 2pass-online：快速出结果（准确率略低），对应 INTERIM_TRANSCRIPT
     - 2pass-offline：说完后出精校结果，对应 FINAL_TRANSCRIPT
   注：仅当服务端同时加载了 paraformer-zh-streaming 时 2pass-online 才有效，
   否则只会收到 2pass-offline（FINAL_TRANSCRIPT）。

事件顺序（流式模式）
-------------------
START_OF_SPEECH → INTERIM_TRANSCRIPT(n) → END_OF_SPEECH → FINAL_TRANSCRIPT

FunASR WebSocket 协议参考
-------------------------
https://github.com/modelscope/FunASR/blob/main/runtime/docs/websocket_protocol.md
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import structlog
import websockets
import websockets.exceptions
from livekit import rtc
from livekit.agents import APIConnectionError, stt
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

logger = structlog.get_logger()

# FunASR 服务的默认 WebSocket 地址
DEFAULT_URL = "ws://localhost:10095"

# FunASR 固定要求 16 kHz 单声道 PCM 输入。
# RecognizeStream 把这个值传给基类，基类会自动把任意采样率重采样到 16 kHz，
# 无需在插件里手动处理重采样。
_INPUT_SAMPLE_RATE = 16000
_FINAL_MODES = {"2pass-offline", "offline"}


def _is_final_result(result: dict) -> bool:
    return result.get("mode", "") in _FINAL_MODES


# ---------------------------------------------------------------------------
# STT 主类
# ---------------------------------------------------------------------------


class STT(stt.STT):
    """FunASR STT 插件入口。

    使用示例
    --------
    # 实时流式识别（对话场景）
    stt = STT(url="ws://localhost:10095", language="zh")

    # 离线一次性识别（录音文件）
    event = await stt.recognize(buffer)
    """

    def __init__(
        self,
        *,
        url: str = DEFAULT_URL,
        model: str = "paraformer-zh",
        language: str = "zh",
    ) -> None:
        """
        参数
        ----
        url      : FunASR WebSocket 服务地址
        model    : 服务端加载的模型名（仅用于 metrics 上报，不影响连接）
        language : 默认识别语言，可被 stream()/recognize() 调用时覆盖
        """
        # 向基类声明本插件的能力：
        #   streaming=True       → 支持 stream()，音频输入是实时流式推送的
        #   interim_results=True → 协议层已实现 2pass。是否真正产生 INTERIM_TRANSCRIPT
        #                          取决于服务端：同时加载 paraformer-zh-streaming 时
        #                          才会返回 2pass-online；否则只会有 2pass-offline 的
        #                          FINAL_TRANSCRIPT。
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )
        self._url = url
        self._model = model
        self._language = language

    @property
    def model(self) -> str:
        # 供 LiveKit metrics 系统上报模型名称
        return self._model

    @property
    def provider(self) -> str:
        # 供 LiveKit metrics 系统上报服务商名称
        return "funasr"

    # ------------------------------------------------------------------
    # 离线识别：接收完整音频缓冲区，返回一个最终结果
    # 基类的 recognize() 会调用此方法，并在外层处理重试和 metrics 上报
    # ------------------------------------------------------------------

    async def _recognize_impl(
        self,
        buffer: stt.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        # 优先使用调用方传入的语言，否则使用实例默认语言
        lang = language if language is not NOT_GIVEN else self._language

        # AudioBuffer 可能是单帧或帧列表，统一转成列表再合并
        frames = [buffer] if isinstance(buffer, rtc.AudioFrame) else list(buffer)

        # 空缓冲区直接返回空结果，避免向服务端发送无意义请求
        if not frames:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=str(uuid.uuid4()),
                alternatives=[stt.SpeechData(language=lang, text="")],
            )

        # 把多帧合并为一帧，得到连续的 PCM 字节流
        combined = rtc.combine_audio_frames(frames)
        t0 = time.perf_counter()

        try:
            async with websockets.connect(
                self._url,
                open_timeout=conn_options.timeout,  # 建立连接的超时时间
            ) as ws:
                logger.debug(
                    "funasr.connected",
                    elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
                )

                # 第一步：发送识别配置（必须先于音频数据）
                # mode="offline" 告诉服务端这是一次性离线识别，不需要流式协议
                # itn=True 启用数字/日期等文字规范化（Inverse Text Normalization）
                await ws.send(
                    json.dumps(
                        {
                            "mode": "offline",
                            "wav_format": "pcm",
                            "wav_name": "recognize",
                            "is_speaking": True,  # True 表示开始发送音频
                            "audio_fs": combined.sample_rate,  # 告知服务端采样率
                            "itn": True,
                        }
                    )
                )

                # 第二步：发送全部 PCM 原始字节（int16-LE 格式）
                await ws.send(bytes(combined.data))

                # 第三步：发送结束信号，通知服务端音频已全部发送完毕
                await ws.send(json.dumps({"is_speaking": False}))

                # 第四步：接收识别结果
                # offline 模式下服务端只返回一条消息就关闭连接，break 退出循环
                text = ""
                async for msg in ws:
                    result = json.loads(msg)

                    # 检查服务端是否返回了错误
                    if "error" in result:
                        raise APIConnectionError(
                            f"FunASR server error: {result['error']}"
                        )

                    text = result.get("text", "").strip()
                    break  # offline 模式只有一条结果

        except (websockets.exceptions.WebSocketException, OSError) as e:
            # 网络错误统一转换为 LiveKit 的 APIConnectionError，
            # 基类 recognize() 会根据 conn_options.max_retry 决定是否重试
            raise APIConnectionError(f"FunASR connection error: {e}") from e

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("funasr.recognized", elapsed_ms=round(elapsed, 1), text=text)

        if not text:
            logger.warning("funasr.empty_result", audio_s=round(combined.duration, 1))

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=str(uuid.uuid4()),
            alternatives=[stt.SpeechData(language=lang, text=text, confidence=1.0)],
        )

    # ------------------------------------------------------------------
    # 流式识别入口：返回一个 RecognizeStream 对象
    # LiveKit AgentSession 在实时对话中会调用此方法
    # ------------------------------------------------------------------

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        lang = language if language is not NOT_GIVEN else self._language
        return RecognizeStream(stt=self, language=lang, conn_options=conn_options)


# ---------------------------------------------------------------------------
# 流式识别流
# ---------------------------------------------------------------------------


class RecognizeStream(stt.RecognizeStream):
    """实时流式识别，对应 FunASR 的 2pass 模式。

    数据流向
    --------
    LiveKit 音频管道
        │  push_frame(AudioFrame)         ← VAD 检测到有人说话时持续推入
        ▼
    self._input_ch  (基类维护的异步通道)
        │  _send_loop 消费
        ▼
    FunASR WebSocket 服务
        │  _recv_loop 接收
        ▼
    self._event_ch  (基类维护的事件通道)
        │
        ▼
    AgentSession（驱动 LLM 推理）

    2pass 模式时序
    -------------
    用户: "今天天气怎么样"
         ├─ [说话中] 2pass-online → INTERIM "今天天气"
         ├─ [说话中] 2pass-online → INTERIM "今天天气怎么"
         └─ [说完后] 2pass-offline → FINAL   "今天天气怎么样"
                                              ↑ 触发 LLM 推理

    完整事件序列
    -----------
    START_OF_SPEECH → INTERIM(n) → END_OF_SPEECH → FINAL_TRANSCRIPT
    """

    def __init__(
        self,
        *,
        stt: STT,
        language: str,
        conn_options: APIConnectOptions,
    ) -> None:
        # sample_rate=_INPUT_SAMPLE_RATE 告诉基类：
        # 如果上游推入的 AudioFrame 不是 16 kHz，自动做重采样后再放入 _input_ch。
        # 这样 _send_loop 收到的永远是 16 kHz 的数据，无需额外处理。
        super().__init__(
            stt=stt, conn_options=conn_options, sample_rate=_INPUT_SAMPLE_RATE
        )
        self._language = language

    async def _run(self) -> None:
        """流式识别主协程，由基类的后台任务调用。

        负责：从 LiveKit 输入流中切分每段语音，并为每段语音建立一个
        独立的 FunASR WebSocket 请求。

        LiveKit 的 flush() 表示一段语音结束，但后续还可能继续 push_frame()。
        FunASR 的 is_speaking=False 则表示当前请求的音频结束。因此这里不能
        复用同一条 WebSocket 连接跨越多个 flush，而是每段语音独立请求。

        连接断开或出错时，基类会根据 conn_options.max_retry 决定是否重新调用 _run。
        """
        input_iter = self._input_ch.__aiter__()

        while True:
            first_frame = await self._next_audio_frame(input_iter)
            if first_frame is None:
                return

            await self._recognize_utterance(first_frame, input_iter)

    async def _recognize_utterance(
        self, first_frame: rtc.AudioFrame, input_iter
    ) -> None:
        stt_instance: STT = self._stt  # type: ignore[assignment]
        t0 = time.perf_counter()

        try:
            async with websockets.connect(
                stt_instance._url,
                open_timeout=self._conn_options.timeout,
            ) as ws:
                logger.debug(
                    "funasr.stream_connected",
                    elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
                )

                await self._send_stream_config(ws)
                recv_task = asyncio.create_task(self._recv_loop(ws, t_conn=t0))

                try:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    )
                    await ws.send(bytes(first_frame.data))

                    await self._send_until_utterance_end(ws, input_iter, recv_task)
                    await asyncio.wait_for(
                        recv_task, timeout=self._conn_options.timeout
                    )

                except asyncio.TimeoutError as e:
                    raise APIConnectionError(
                        "FunASR timed out waiting for final transcript"
                    ) from e
                finally:
                    if not recv_task.done():
                        recv_task.cancel()
                        await asyncio.gather(recv_task, return_exceptions=True)
        except (websockets.exceptions.WebSocketException, OSError) as e:
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

    async def _send_stream_config(self, ws: websockets.WebSocketClientProtocol) -> None:
        # 发送流式识别配置（必须是第一条消息）。
        # mode="2pass"：声明使用双引擎模式。chunk_* 参数仅 streaming 模型生效。
        await ws.send(
            json.dumps(
                {
                    "mode": "2pass",
                    "wav_format": "pcm",
                    "wav_name": "stream",
                    "is_speaking": True,
                    "chunk_size": [5, 10, 5],
                    "chunk_interval": 10,
                    "audio_fs": _INPUT_SAMPLE_RATE,
                    "itn": True,
                    "language": self._language,
                }
            )
        )

    async def _send_until_utterance_end(
        self,
        ws: websockets.WebSocketClientProtocol,
        input_iter,
        recv_task: asyncio.Task[None],
    ) -> None:
        while True:
            input_task = asyncio.create_task(self._next_input(input_iter))

            try:
                done, _ = await asyncio.wait(
                    {input_task, recv_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if input_task in done:
                    data = input_task.result()

                    if data is None or isinstance(
                        data, stt.RecognizeStream._FlushSentinel
                    ):
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                        )
                        await ws.send(json.dumps({"is_speaking": False}))
                        return

                    if isinstance(data, rtc.AudioFrame):
                        await ws.send(bytes(data.data))

                if recv_task in done:
                    if not input_task.done():
                        input_task.cancel()
                        await asyncio.gather(input_task, return_exceptions=True)

                    exc = recv_task.exception()
                    if exc is not None:
                        raise exc

                    raise APIConnectionError(
                        "FunASR stream ended before end-of-speech signal"
                    )

            except Exception:
                if not input_task.done():
                    input_task.cancel()
                    await asyncio.gather(input_task, return_exceptions=True)
                raise

    async def _recv_loop(
        self, ws: websockets.WebSocketClientProtocol, *, t_conn: float
    ) -> None:
        """接收 FunASR 服务端的识别结果，封装为 SpeechEvent 写入事件通道。

        参数
        ----
        t_conn : 连接建立时的 time.perf_counter() 值，用于计算首次出结果的延迟。

        FunASR 2pass 模式返回的消息格式：
        {
            "text": "识别文本",
            "mode": "2pass-online" | "2pass-offline",
            "wav_name": "stream",
            "timestamp": "[[0,350],[350,800],...]"  // 可选，词级时间戳
        }
        """
        first_result_logged = False

        try:
            async for msg in ws:
                result = json.loads(msg)

                # 检查服务端是否返回了错误响应
                if "error" in result:
                    raise APIConnectionError(f"FunASR server error: {result['error']}")

                # 根据 mode 判断结果类型：
                # 2pass-online  → 中间结果，文本可能还会变化，触发预生成但不触发最终推理
                # 2pass-offline → 最终结果，触发 LLM 完整推理
                mode = result.get("mode", "")
                is_final = _is_final_result(result)
                text = result.get("text", "").strip()

                # 服务端偶尔返回空的中间结果（心跳或空帧），跳过。
                # 空 final 仍然要作为本段识别结束处理，避免等待后续消息而卡住。
                if not text and not is_final:
                    continue

                # 记录首次出结果的延迟（从连接建立到收到第一条有效结果）
                if not first_result_logged:
                    first_result_logged = True
                    elapsed_ms = (time.perf_counter() - t_conn) * 1000
                    logger.debug(
                        "funasr.first_result",
                        elapsed_ms=round(elapsed_ms, 1),
                        mode=mode,
                        text=text,
                    )

                event_type = (
                    stt.SpeechEventType.FINAL_TRANSCRIPT
                    if is_final
                    else stt.SpeechEventType.INTERIM_TRANSCRIPT
                )

                # send_nowait：事件写入是非阻塞的，避免接收循环被下游消费速度阻塞
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=event_type,
                        request_id=str(uuid.uuid4()),
                        alternatives=[
                            stt.SpeechData(
                                language=self._language,
                                text=text,
                                confidence=1.0,
                            )
                        ],
                    )
                )

                if is_final:
                    return

        except websockets.exceptions.ConnectionClosed as e:
            raise APIConnectionError(
                "FunASR stream closed before final transcript"
            ) from e

        raise APIConnectionError("FunASR stream ended before final transcript")
