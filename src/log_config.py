"""
日志系统配置（structlog + QueueHandler）

设计对标 Go zap：
- 字段是独立的键值对，不插入消息字符串
- text 模式（开发）：彩色可读输出
- json 模式（生产）：JSON Lines，直接对接 ELK / Datadog / Loki

性能设计要点
-----------
1. filter_by_level（第一个 processor）
   不满足 level 的事件立即 DropEvent，后续全部跳过，避免无效的
   contextvars 合并、时间戳生成等操作。

2. QueueHandler + QueueListener（后台线程做真正的 I/O）
   asyncio 事件循环调用 logger.info() 只是往队列里 put 一个对象（~100ns，非阻塞）。
   后台 QueueListener 线程负责真正的 stdout.write() / 文件写入。
   磁盘抖动或 stdout 慢时，音频流水线不受影响。

structlog 与 Go zap 用法对比：
    # Go zap
    log.Info("session started", zap.String("room", roomName))
    # structlog（本项目）
    log.info("session.started", room=room_name)

.env.local 配置项（均有默认值）：
    LOG_LEVEL     = INFO            # DEBUG / INFO / WARNING / ERROR
    LOG_FORMAT    = text            # text 或 json
    LOG_FILE      = logs/agent.log  # 留空则只输出到控制台
    LOG_ROTATION  = 10 MB           # 单文件大小上限（支持 KB/MB/GB）
    LOG_RETENTION = 5               # 保留的历史日志份数
"""

from __future__ import annotations

import atexit
import copy
import logging
import logging.handlers
import os
import queue
import sys
from pathlib import Path

import structlog


class _StructlogQueueHandler(logging.handlers.QueueHandler):
    """QueueHandler variant that leaves structlog event dicts intact."""

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        return copy.copy(record)


def setup_logging() -> None:
    """初始化全局日志系统。

    必须在 agent.py 中最早调用（load_dotenv 之后、其他 import 之前），
    确保第三方库的日志也被 structlog 拦截格式化。
    """
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    log_format = os.getenv("LOG_FORMAT", "text").lower()
    log_file = os.getenv("LOG_FILE", "").strip()
    rotation_str = os.getenv("LOG_ROTATION", "10 MB")
    retention = int(os.getenv("LOG_RETENTION", "5"))

    is_json = log_format == "json"

    # ── Processor 链 ─────────────────────────────────────────────────────────
    shared_processors: list[structlog.types.Processor] = [
        # ① 最先过滤：不满足 level 的事件立即 DropEvent，后续 processor 全部跳过
        structlog.stdlib.filter_by_level,
        # ② 合并当前协程的 contextvars（room、session_id 等）
        structlog.contextvars.merge_contextvars,
        # ③ 添加 log_level 字段
        structlog.stdlib.add_log_level,
        # ④ 添加 logger 字段（模块名，内部用 sys._getframe()，已被①大幅抵消）
        structlog.stdlib.add_logger_name,
        # ⑤ 添加 timestamp（ISO 8601 UTC）
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # StackInfoRenderer 已移除：我们从不传 stack_info=True，保留只是空跑
    ]

    # ── Renderer ─────────────────────────────────────────────────────────────
    if is_json:
        # 生产模式：JSON Lines，orjson 比标准库 json 快 5-10 倍
        try:
            import orjson

            def _orjson_dumps(obj: object, **_: object) -> str:
                return orjson.dumps(obj).decode()

            renderer: structlog.types.Processor = structlog.processors.JSONRenderer(
                serializer=_orjson_dumps
            )
        except ImportError:
            renderer = structlog.processors.JSONRenderer()
    else:
        # 开发模式：彩色对齐输出
        renderer = structlog.dev.ConsoleRenderer(
            colors=sys.stdout.isatty(),
        )

    # ── 配置 structlog ────────────────────────────────────────────────────────
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # ── ProcessorFormatter：统一格式化 structlog + 第三方 stdlib 日志 ────────
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # ── 真正做 I/O 的 handlers（运行在后台线程，不阻塞事件循环）───────────────
    real_handlers: list[logging.Handler] = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    real_handlers.append(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = _make_file_handler(log_file, rotation_str, retention)
        file_handler.setFormatter(formatter)
        real_handlers.append(file_handler)

    # ── QueueHandler + QueueListener ─────────────────────────────────────────
    # asyncio 事件循环只做 queue.put()（~100ns，非阻塞）。
    # QueueListener 在独立守护线程里消费队列，调用真正的 I/O handler。
    # SimpleQueue 的 put() 永远不阻塞（无 maxsize 限制）。
    log_queue: queue.SimpleQueue = queue.SimpleQueue()

    queue_handler = _StructlogQueueHandler(log_queue)  # type: ignore[arg-type]

    listener = logging.handlers.QueueListener(
        log_queue,  # type: ignore[arg-type]
        *real_handlers,
        respect_handler_level=True,  # 每个 real_handler 仍独立做 level 检查
    )
    listener.start()
    # 进程退出时自动 flush 队列并停止后台线程
    atexit.register(listener.stop)

    # ── root logger 只挂 QueueHandler，日志调用路径完全非阻塞 ─────────────────
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(queue_handler)
    root.setLevel(level)

    # 第三方库降级到 WARNING，防止心跳包等低价值日志刷屏
    for _noisy in ("websockets", "aiohttp", "livekit", "httpx", "httpcore", "uvicorn"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    log = structlog.get_logger()
    log.info(
        "logging.ready", level=level_str, format=log_format, file=log_file or "stdout"
    )


# TimedRotatingFileHandler 的合法 when 值
# 完整列表见 https://docs.python.org/3/library/logging.handlers.html#timedrotatingfilehandler
_TIMED_WHEN = {"midnight", "s", "m", "h", "d", "w0", "w1", "w2", "w3", "w4", "w5", "w6"}


def _make_file_handler(
    path: str, rotation: str, retention: int
) -> logging.handlers.BaseRotatingHandler:
    """根据 LOG_ROTATION 值选择按大小或按时间滚动的 handler。

    按大小滚动（默认）：
        LOG_ROTATION = 10 MB   → RotatingFileHandler，单文件上限 10 MB
        LOG_ROTATION = 512 KB
        LOG_ROTATION = 1 GB

    按时间滚动：
        LOG_ROTATION = midnight → 每天 00:00 新建文件（最常用）
        LOG_ROTATION = D        → 每天滚动
        LOG_ROTATION = H        → 每小时滚动
        LOG_ROTATION = W0       → 每周一滚动（W0=周一 … W6=周日）

    生成的文件名：logs/agent.log.2026-05-05
    """
    if rotation.strip().lower() in _TIMED_WHEN:
        return logging.handlers.TimedRotatingFileHandler(
            path,
            when=rotation.strip(),
            backupCount=retention,  # 保留最近 N 个文件（天/小时/周）
            encoding="utf-8",
        )
    return logging.handlers.RotatingFileHandler(
        path,
        maxBytes=_parse_size(rotation),
        backupCount=retention,
        encoding="utf-8",
    )


def _parse_size(size_str: str) -> int:
    """将 '10 MB' / '512 KB' 等字符串转换为字节数。"""
    s = size_str.strip().upper().replace(" ", "")
    units = {"GB": 1024**3, "MB": 1024**2, "KB": 1024}
    for suffix, multiplier in units.items():
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * multiplier)
    return int(s)
