from __future__ import annotations

import asyncio
import logging
from asyncio import AbstractEventLoop
from collections import defaultdict
from collections.abc import Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering
from typing import Any

_log = logging.getLogger(__name__)


@total_ordering
class Priority(Enum):
    """Priority of an event."""

    NORMAL = 0
    ABNORMAL = 1
    ABORT = 2
    INTERRUPT = 3
    SYSTEM_ERROR = 4

    def __lt__(self, other: Priority) -> bool:
        if not isinstance(other, Priority):
            return NotImplemented
        return self.value < other.value


@dataclass(frozen=True)
class Event:
    """A scheduled event."""

    coro: Callable[..., Coroutine[Any, Any, Any]]
    priority: Priority = Priority.NORMAL
    name: str = ""


class EventHandler:
    """Base class for Flight. Handles low level events."""

    def __init__(self):
        self.loop: AbstractEventLoop | None = None
        self._events: dict[Priority, set[asyncio.Task]] = defaultdict(set)
        self._delay_events: dict[Priority, set[asyncio.Task]] = defaultdict(set)

    def _schedule_event(
        self,
        coro_func: Callable[..., Coroutine[Any, Any, Any]],
        args: tuple[Any, ...] = (),
        kwds: dict[str, Any] | None = None,
        event_name: str | None = None,
        priority: Priority = Priority.NORMAL,
    ) -> asyncio.Task:
        # Use func name as default
        if event_name is None:
            event_name = getattr(coro_func, "__name__", "unnamed_event")
        wrapped = self._run_event(coro_func)
        task = self.loop.create_task(wrapped, name=f"kflight: {event_name}")
        self._events[priority].add(task)
        task.add_done_callback(self._events[priority].remove)
        return task

    def delay(self, duration: float, priority: Priority = Priority.NORMAL):
        """Schedules a delay event."""
        coro = self._run_event(asyncio.sleep, (duration,))
        event_name = f"delay_{duration}"
        task = self.loop.create_task(coro, name=f"kflight: {event_name}")
        self._delay_events[priority].add(task)
        task.add_done_callback(self._delay_events[priority].remove)
        return task

    def _cancel_below(self, loop: AbstractEventLoop, priority: Priority):
        """Cancels all events below the given priority."""
        # Cancel events then delays
        def do_cancel():
            for event_type in (self._events, self._delay_events):
                for prio, tasks in event_type.items():
                    if prio < priority:
                        for task in tasks:
                            with suppress(asyncio.CancelledError):
                                task.cancel()

        loop.call_soon_threadsafe(do_cancel)

    @staticmethod
    async def _run_event(
        coro: Callable[..., Coroutine[Any, Any, Any]],
        args: tuple[Any, ...] = (),
        kwds: dict[str, Any] | None = None,
    ) -> None:
        try:
            await coro(*args, **kwds or {})
        except asyncio.CancelledError:
            _log.debug("Event cancelled: %s", coro.__name__)
            raise
