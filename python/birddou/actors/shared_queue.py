"""Thread-safe bounded trajectory queue with explicit close semantics."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Full
from threading import Condition
from typing import Generic, TypeVar

ItemT = TypeVar("ItemT")


class QueueClosed(RuntimeError):
    """The producer/consumer boundary was closed permanently."""


@dataclass(frozen=True, slots=True)
class SharedQueueStats:
    """Constant-size diagnostics for one bounded queue."""

    size: int
    capacity: int
    maximum_size: int
    put_count: int
    get_count: int
    producer_wait_count: int
    consumer_wait_count: int
    closed: bool


class BoundedTrajectoryQueue(Generic[ItemT]):
    """A bounded MPMC queue whose close wakes every blocked producer/consumer.

    ``queue.Queue`` has no terminal close operation.  Actors and learners need
    that operation so a crashed peer cannot leave shutdown blocked forever.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("trajectory queue capacity must be positive")
        self._capacity = capacity
        self._items: deque[ItemT] = deque()
        self._condition = Condition()
        self._closed = False
        self._maximum_size = 0
        self._put_count = 0
        self._get_count = 0
        self._producer_wait_count = 0
        self._consumer_wait_count = 0

    def put(
        self,
        item: ItemT,
        *,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Add an item, applying bounded backpressure or raising ``Full``."""
        deadline = _deadline(block, timeout)
        with self._condition:
            while len(self._items) >= self._capacity:
                if self._closed:
                    raise QueueClosed("trajectory queue is closed")
                if not block:
                    raise Full
                self._producer_wait_count += 1
                if not self._wait(deadline):
                    raise Full
            if self._closed:
                raise QueueClosed("trajectory queue is closed")
            self._items.append(item)
            self._put_count += 1
            self._maximum_size = max(self._maximum_size, len(self._items))
            self._condition.notify_all()

    def get(
        self,
        *,
        block: bool = True,
        timeout: float | None = None,
    ) -> ItemT:
        """Remove an item, or raise ``QueueClosed`` once a closed queue drains."""
        deadline = _deadline(block, timeout)
        with self._condition:
            while not self._items:
                if self._closed:
                    raise QueueClosed("trajectory queue is closed and empty")
                if not block:
                    raise Empty
                self._consumer_wait_count += 1
                if not self._wait(deadline):
                    raise Empty
            item = self._items.popleft()
            self._get_count += 1
            self._condition.notify_all()
            return item

    def close(self, *, discard: bool = False) -> tuple[ItemT, ...]:
        """Close permanently, optionally discarding and returning queued items."""
        with self._condition:
            self._closed = True
            discarded: tuple[ItemT, ...] = ()
            if discard:
                discarded = tuple(self._items)
                self._items.clear()
            self._condition.notify_all()
            return discarded

    def stats(self) -> SharedQueueStats:
        """Return a lock-consistent constant-size metric snapshot."""
        with self._condition:
            return SharedQueueStats(
                size=len(self._items),
                capacity=self._capacity,
                maximum_size=self._maximum_size,
                put_count=self._put_count,
                get_count=self._get_count,
                producer_wait_count=self._producer_wait_count,
                consumer_wait_count=self._consumer_wait_count,
                closed=self._closed,
            )

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def _wait(self, deadline: float | None) -> bool:
        if deadline is None:
            self._condition.wait()
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        return self._condition.wait(remaining)


def _deadline(block: bool, timeout: float | None) -> float | None:
    if timeout is not None and (not block or not math.isfinite(timeout) or timeout < 0.0):
        raise ValueError("queue timeout requires blocking mode and must be finite/non-negative")
    return None if timeout is None else time.monotonic() + timeout


__all__ = (
    "BoundedTrajectoryQueue",
    "QueueClosed",
    "SharedQueueStats",
)
