"""头像生成消息队列。

修复的问题：新用户注册后直接 asyncio.create_task 去生成头像，每个任务
抱着数据库会话等最长 600s 的 LLM 调用；后台 backfill 又一次性在一个
事务里连做多枚。SQLite 单写锁被长期占住，聊天、注册全部
"database is locked"，表现为新用户一进来就把库卡死。

现在的流程（worker 串行，永远只有一个头像任务在跑）：

    注册 / backfill 循环 --enqueue--> 队列（去重 + 有界） --> worker
    worker：
      1) 短事务：资格检查并取走素材（会话即开即关）；
      2) 会话外调用 LLM（超时 AVATAR_LLM_TIMEOUT_SECONDS，默认 60s）；
      3) 短事务写回结果，写事务经 db_gate 以 AVATAR 优先级排队，
         对话（CHAT）永远优先。

同一 agent 在队列 / 在途只会存在一个任务；worker 与队列在 lifespan
里统一 start/stop，异常与取消都有兜底释放。
"""

from __future__ import annotations

import asyncio

import structlog
from sqlalchemy import or_, select

from app.core.config import Settings
from app.core.db_gate import GatePriority, write_gate
from app.core.time_utils import now_utc8
from app.models.models import Agent
from app.services import avatar_service
from app.services.llm_client import LLMClient

logger = structlog.get_logger(__name__)


class AvatarQueue:
    def __init__(
        self,
        session_factory,
        llm: LLMClient | None,
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._llm = llm
        self._settings = settings
        self._queue: asyncio.Queue[int] = asyncio.Queue(
            maxsize=max(1, settings.AVATAR_QUEUE_MAX_SIZE)
        )
        self._pending: set[int] = set()
        self._workers: list[asyncio.Task] = []

    @property
    def depth(self) -> int:
        return self._queue.qsize() + len(self._pending)

    def enqueue(self, agent_id: int) -> bool:
        """非阻塞投递；重复任务自动去重，队列满时丢弃（backfill 会兜底再投）。"""
        if agent_id in self._pending:
            return False
        self._pending.add(agent_id)
        try:
            self._queue.put_nowait(agent_id)
            logger.info("avatar_enqueued", agent_id=agent_id, depth=self.depth)
            return True
        except asyncio.QueueFull:
            self._pending.discard(agent_id)
            logger.warning("avatar_queue_full", agent_id=agent_id)
            return False

    def start(self, workers: int = 1) -> None:
        for i in range(max(1, workers)):
            self._workers.append(
                asyncio.create_task(self._worker_loop(i), name=f"avatar-worker-{i}")
            )

    async def stop(self) -> None:
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._pending.clear()

    async def _worker_loop(self, worker_id: int) -> None:
        while True:
            agent_id = await self._queue.get()
            try:
                await self._process(agent_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "avatar_worker_failed",
                    agent_id=agent_id,
                    error=str(exc),
                    exc_info=True,
                )
            finally:
                self._pending.discard(agent_id)
                self._queue.task_done()

    async def _process(self, agent_id: int) -> None:
        settings = self._settings

        # 1) 短事务：资格检查 + 取素材。事务内不做任何耗时操作。
        async with self._session_factory() as session:
            agent = (
                await session.execute(select(Agent).where(Agent.id == agent_id))
            ).scalar_one_or_none()
            if agent is None or not avatar_service.avatar_retry_due(agent, settings):
                return
            snapshot_name = agent.name
            snapshot_personality = agent.personality or {}

        # 2) LLM 调用：不占用任何数据库会话 / 写锁。
        if not avatar_service.llm_ready(self._llm):
            return
        svg, error = await avatar_service.request_avatar_svg(
            snapshot_name,
            snapshot_personality,
            self._llm,
            settings,
        )

        # 3) 短事务写回；写锁走门闸（AVATAR 优先级），拿不到就放弃等 backfill。
        try:
            async with write_gate(
                GatePriority.AVATAR,
                timeout=settings.DB_GATE_DEFAULT_TIMEOUT_SECONDS,
                label="avatar_write",
            ):
                async with self._session_factory() as session:
                    agent = (
                        await session.execute(select(Agent).where(Agent.id == agent_id))
                    ).scalar_one_or_none()
                    if agent is not None and not agent.avatar_svg:
                        avatar_service.apply_avatar_result(
                            session,
                            agent,
                            svg=svg,
                            error=error,
                            retry_base_seconds=settings.AVATAR_RETRY_BASE_SECONDS,
                        )
                        await session.commit()
        except Exception as exc:
            logger.warning(
                "avatar_writeback_failed",
                agent_id=agent_id,
                error=str(exc),
                exc_info=True,
            )
            return
        if svg:
            logger.info("avatar_generated", agent_id=agent_id)
        elif error:
            logger.info("avatar_failed_retry_scheduled", agent_id=agent_id, error=error)

    async def run_backfill_loop(self) -> None:
        """定期把缺头像的存量 agent 投进同一队列，而不是自己连做多枚。"""
        interval = max(10, self._settings.AVATAR_BACKFILL_INTERVAL_SECONDS)
        while True:
            try:
                if avatar_service.llm_ready(self._llm):
                    async with self._session_factory() as session:
                        result = await session.execute(
                            select(Agent.id)
                            .where(
                                or_(Agent.avatar_svg.is_(None), Agent.avatar_svg == ""),
                                Agent.avatar_attempts < max(1, self._settings.AVATAR_MAX_RETRIES),
                                or_(
                                    Agent.avatar_next_retry_at.is_(None),
                                    Agent.avatar_next_retry_at <= now_utc8(),
                                ),
                            )
                            .order_by(Agent.created_at.asc())
                            .limit(max(1, self._settings.AVATAR_BATCH_SIZE))
                        )
                        ids = [int(row) for row in result.scalars().all()]
                    for agent_id in ids:
                        self.enqueue(agent_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("avatar_backfill_failed", error=str(exc), exc_info=True)
            await asyncio.sleep(interval)
