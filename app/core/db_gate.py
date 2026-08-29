"""SQLite 写事务优先级门闸。

背景：SQLite 同一时刻只允许一个写事务。此前头像生成等后台任务抱着
数据库会话去等最长 600s 的 LLM 调用，把唯一的写锁长期占住，用户对话、
注册等所有写操作全部堆积超时（"database is locked"），表现为整库卡死。

这里用进程内的单例门闸把所有写事务串行化：
- 等锁者按优先级唤醒，用户对话（CHAT）永远最先进；
- acquire 必须带超时，拿不到就抛 GateAcquireTimeout，调用方降级处理，
  绝不无限等待把请求堆死；
- 统一通过 async with 使用，异常 / 取消 / 超时都保证释放锁，
  持锁时间超阈值会打告警日志，便于定位新的长事务。
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from contextlib import asynccontextmanager
from enum import IntEnum
from typing import AsyncIterator

import structlog

logger = structlog.get_logger(__name__)

# 持锁超过该秒数即打告警（不抢锁，只观测）。对话路径的写事务应在毫秒级。
_HOLD_WARN_SECONDS = 20.0


class GatePriority(IntEnum):
    """写门闸优先级，数值越小优先级越高。"""

    CHAT = 0  # 用户对话（聊天 / 私信 / 清空会话）—— 最高优先级
    INTERACT = 1  # 注册、验证码、审批等用户交互写
    DISCOVERY = 2  # 匹配探索后台任务
    AVATAR = 3  # 头像生成后台任务
    MAINTENANCE = 4  # 其他后台维护


class GateAcquireTimeout(RuntimeError):
    """在超时时间内没拿到写门闸；调用方应降级或稍后重试。"""


class _WriteGate:
    def __init__(self) -> None:
        self._locked = False
        # (priority, seq, future, label)：priority 小者先出队，同优先级 FIFO
        self._waiters: list[tuple[int, int, asyncio.Future[None], str]] = []
        self._seq = itertools.count()
        self._holder_since = 0.0
        self._holder_label = ""

    @asynccontextmanager
    async def acquire(
        self,
        priority: GatePriority,
        timeout: float,
        label: str = "",
    ) -> AsyncIterator[None]:
        granted = False
        fut: asyncio.Future[None] | None = None
        entry: tuple[int, int, asyncio.Future[None], str] | None = None
        loop = asyncio.get_running_loop()

        if not self._locked:
            self._locked = True
            self._holder_since = time.monotonic()
            self._holder_label = label
            granted = True
        else:
            fut = loop.create_future()
            entry = (int(priority), next(self._seq), fut, label)
            heapq.heappush(self._waiters, entry)
            try:
                await asyncio.wait_for(fut, timeout=max(0.1, timeout))
                granted = True
            except asyncio.TimeoutError:
                self._discard(entry)
                raise GateAcquireTimeout(
                    f"写门闸等待超时（priority={priority.name} label={label} timeout={timeout}s）"
                ) from None
            except asyncio.CancelledError:
                self._discard(entry)
                if fut.done() and not fut.cancelled():
                    # 已被移交却随即被取消：必须把锁转交给下一个等待者，否则死锁
                    self._hand_off()
                raise

        try:
            yield
        finally:
            if granted:
                self._release()
            else:  # pragma: no cover - 防御分支
                logger.error("write_gate_released_without_grant", label=label)

    def _discard(self, entry: tuple[int, int, asyncio.Future[None], str]) -> None:
        if entry in self._waiters:
            self._waiters.remove(entry)
            heapq.heapify(self._waiters)

    def _hand_off(self) -> None:
        while self._waiters:
            _prio, _seq, fut, label = heapq.heappop(self._waiters)
            if fut.done():
                continue
            fut.set_result(None)
            self._holder_since = time.monotonic()
            self._holder_label = label
            return
        self._locked = False

    def _release(self) -> None:
        held = time.monotonic() - self._holder_since
        if held > _HOLD_WARN_SECONDS:
            logger.warning(
                "write_gate_long_hold",
                label=self._holder_label,
                held_seconds=round(held, 1),
            )
        self._hand_off()


_WRITE_GATE = _WriteGate()


@asynccontextmanager
async def write_gate(
    priority: GatePriority = GatePriority.MAINTENANCE,
    timeout: float = 10.0,
    label: str = "",
) -> AsyncIterator[None]:
    """串行化 SQLite 写事务的进程级门闸；用户对话传 GatePriority.CHAT。"""
    async with _WRITE_GATE.acquire(priority, timeout, label):
        yield
