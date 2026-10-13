"""
Centralized priority-queue task scheduler for multi-session (multi-tab) support.

Each browser tab gets a unique session_id.  Tasks are submitted in batches
(e.g. "process these 14 images"), prioritized by a pluggable function, and
executed by a fixed pool of async workers.

Default priority: earliest_session_first — all tasks from the oldest active
session complete before the next session's tasks begin.

Usage:
    scheduler = TaskScheduler(concurrency=5, priority_fn=earliest_session_first, name="process")
    scheduler.start()                     # call once the event loop is running
    session = await scheduler.submit_batch(session_id, [factory1, factory2, ...])
    async for result in read_results(session, count):
        ...
    scheduler.cleanup_session(session_id)
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List

from src.logger import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SessionInfo:
    """Tracks state for one scheduler session (one browser tab's work batch)."""

    session_id: str
    start_time: float
    pending_count: int = 0
    completed_count: int = 0
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    result_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _TaskItem:
    """Internal: a single unit of work.  Not orderable — tuple position in the
    priority queue handles ordering so comparison never reaches this object."""

    session_id: str
    coroutine_factory: Callable[[], Awaitable[Any]]
    cancelled: asyncio.Event
    result_queue: asyncio.Queue


# ---------------------------------------------------------------------------
# Priority functions (plug-and-play)
# ---------------------------------------------------------------------------

def earliest_session_first(
    session_id: str, sessions: Dict[str, SessionInfo]
) -> float:
    """All tasks from the earliest-started active session get top priority (0).
    The next-oldest gets 1, and so on.  Cancelled sessions sink to infinity."""
    active = {sid: s for sid, s in sessions.items() if not s.cancelled.is_set()}
    if session_id not in active:
        return float("inf")
    ranked = sorted(active.keys(), key=lambda sid: active[sid].start_time)
    for rank, sid in enumerate(ranked):
        if sid == session_id:
            return float(rank)
    return float("inf")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Priority-queue scheduler with configurable concurrency and priority fn."""

    def __init__(
        self,
        concurrency: int = 5,
        priority_fn: Callable[
            [str, Dict[str, SessionInfo]], float
        ] = earliest_session_first,
        name: str = "scheduler",
    ):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._concurrency = concurrency
        self._priority_fn = priority_fn
        self._name = name
        self._sessions: Dict[str, SessionInfo] = {}
        self._submit_counter: int = 0
        self._workers: List[asyncio.Task] = []
        self._running = False

    # -- Public API --------------------------------------------------------

    async def submit_batch(
        self,
        session_id: str,
        coroutine_factories: List[Callable[[], Awaitable[Any]]],
    ) -> SessionInfo:
        """Submit a batch of tasks for a session.

        If the session already exists (e.g. re-process after cancel), the old
        batch is cancelled and a fresh session is created.

        Returns the SessionInfo whose ``result_queue`` delivers results.
        """
        # Cancel old batch if the same session re-submits
        if session_id in self._sessions:
            self._sessions[session_id].cancelled.set()

        session = SessionInfo(
            session_id=session_id,
            start_time=time.monotonic(),
            pending_count=len(coroutine_factories),
        )
        self._sessions[session_id] = session

        priority = self._priority_fn(session_id, self._sessions)

        for factory in coroutine_factories:
            self._submit_counter += 1
            await self._queue.put((
                priority,
                self._submit_counter,
                _TaskItem(
                    session_id=session_id,
                    coroutine_factory=factory,
                    cancelled=session.cancelled,
                    result_queue=session.result_queue,
                ),
            ))

        logger.info(
            f"[{self._name}] Submitted {len(coroutine_factories)} tasks "
            f"for session {session_id[:8]}... (priority={priority:.0f})"
        )
        return session

    def cancel_session(self, session_id: str) -> None:
        """Cancel all pending and in-flight tasks for a session."""
        if session_id in self._sessions:
            self._sessions[session_id].cancelled.set()
            logger.info(f"[{self._name}] Cancelled session {session_id[:8]}...")

    def cleanup_session(self, session_id: str) -> None:
        """Remove session state.  Call after all results have been consumed."""
        self._sessions.pop(session_id, None)

    # -- Worker loop -------------------------------------------------------

    async def _worker(self, worker_id: int) -> None:
        while self._running:
            # Block until a task is available (with timeout so we can check _running)
            try:
                _, _, item = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            session = self._sessions.get(item.session_id)

            # Fast-skip cancelled tasks
            if item.cancelled.is_set():
                await item.result_queue.put(None)
                self._queue.task_done()
                if session:
                    session.completed_count += 1
                continue

            # Execute the task
            try:
                result = await item.coroutine_factory()
                # Re-check cancel (may have been set during execution)
                if item.cancelled.is_set():
                    await item.result_queue.put(None)
                else:
                    await item.result_queue.put(result)
            except Exception as e:
                logger.error(f"[{self._name}] Worker {worker_id} error: {e}")
                await item.result_queue.put(None)
            finally:
                self._queue.task_done()
                if session:
                    session.completed_count += 1

    # -- Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start worker coroutines.  Call once the event loop is running."""
        if self._running:
            return
        self._running = True
        for i in range(self._concurrency):
            self._workers.append(asyncio.create_task(self._worker(i)))
        logger.info(f"[{self._name}] Started {self._concurrency} workers")

    async def stop(self) -> None:
        """Gracefully shut down all workers."""
        self._running = False
        for t in self._workers:
            t.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info(f"[{self._name}] Stopped")


# ---------------------------------------------------------------------------
# Global instances (imported by app.py)
# ---------------------------------------------------------------------------

scrape_scheduler = TaskScheduler(concurrency=3, name="scrape")
process_scheduler = TaskScheduler(concurrency=5, name="process")
