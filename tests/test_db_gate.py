"""写门闸（db_gate）行为验证：优先级、超时、取消安全。"""

from __future__ import annotations

import asyncio

import pytest

from app.core.db_gate import GateAcquireTimeout, GatePriority, _WriteGate


async def _hold_and_wake(gate, priority, hold_seconds, started: asyncio.Event):
    async with gate.acquire(priority, 5, label="holder"):
        started.set()
        await asyncio.sleep(hold_seconds)


@pytest.mark.asyncio
async def test_priority_order_chat_first():
    gate = _WriteGate()
    order: list[str] = []
    started = asyncio.Event()

    async def waiter(priority: GatePriority, name: str, delay: float):
        await asyncio.sleep(delay)
        async with gate.acquire(priority, 5, label=name):
            order.append(name)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_hold_and_wake(gate, GatePriority.MAINTENANCE, 0.3, started))
        tg.create_task(waiter(GatePriority.AVATAR, "avatar", 0.01))
        tg.create_task(waiter(GatePriority.DISCOVERY, "discovery", 0.02))
        tg.create_task(waiter(GatePriority.CHAT, "chat", 0.03))

    assert order == ["chat", "discovery", "avatar"]


@pytest.mark.asyncio
async def test_acquire_timeout_releases_waiter():
    gate = _WriteGate()
    started = asyncio.Event()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_hold_and_wake(gate, GatePriority.MAINTENANCE, 0.3, started))
        tg.create_task(_timeout_waiter(gate))

    # 超时后门闸必须能立即被重新获取（等待者已被清理）
    async with gate.acquire(GatePriority.CHAT, 1, label="after"):
        pass


async def _timeout_waiter(gate):
    await asyncio.sleep(0.01)
    with pytest.raises(GateAcquireTimeout):
        async with gate.acquire(GatePriority.MAINTENANCE, 0.1, label="slow"):
            raise AssertionError("不应拿到锁")


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_deadlock():
    gate = _WriteGate()
    started = asyncio.Event()
    holder = asyncio.create_task(_hold_and_wake(gate, GatePriority.MAINTENANCE, 0.3, started))
    await asyncio.sleep(0.01)

    task = asyncio.create_task(_cancelable_waiter(gate))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await holder
    # 被取消的等待者不能吞掉锁：必须能立刻重新拿到
    async with gate.acquire(GatePriority.CHAT, 1, label="after-cancel"):
        pass


async def _cancelable_waiter(gate):
    async with gate.acquire(GatePriority.DISCOVERY, 5, label="victim"):
        raise AssertionError("被取消的等待者不应进入临界区")
