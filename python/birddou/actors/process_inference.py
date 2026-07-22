"""Bounded IPC bridge from spawned actors to the central inference server."""

from __future__ import annotations

import asyncio
import math
import multiprocessing as mp
import operator
import os
import time
from dataclasses import dataclass
from queue import Empty, Full
from threading import Event, Lock, Thread
from typing import cast

from birddou.actors.inference_server import InferenceResult, InferenceServer
from birddou.actors.process_supervisor import InterprocessQueue
from birddou.features.ragged import RaggedBatch


class InferenceBridgeClosed(RuntimeError):
    """The cross-process inference service is not available."""


class InferenceBridgeTimeout(TimeoutError):
    """An actor did not receive a matching response before its safety timeout."""


@dataclass(frozen=True, slots=True)
class ProcessInferenceConfig:
    """Bounded cross-process request/response transport settings."""

    actor_processes: int
    request_queue_capacity: int = 1_024
    response_queue_capacity: int = 8
    actor_response_timeout_s: float = 30.0
    bridge_poll_timeout_s: float = 0.01

    def __post_init__(self) -> None:
        if (
            min(
                self.actor_processes,
                self.request_queue_capacity,
                self.response_queue_capacity,
            )
            <= 0
        ):
            raise ValueError("process inference counts and capacities must be positive")
        for name, value in (
            ("actor_response_timeout_s", self.actor_response_timeout_s),
            ("bridge_poll_timeout_s", self.bridge_poll_timeout_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class ProcessInferenceRequest:
    """One actor's ragged batch and immutable requested model version."""

    request_id: str
    actor_id: int
    policy_version: int
    batch: RaggedBatch
    submitted_ns: int


@dataclass(frozen=True, slots=True)
class ProcessInferenceReply:
    """One matching result or an explicit remote failure string."""

    request_id: str
    result: InferenceResult | None
    error_type: str | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class ProcessInferenceChannels:
    """Pickle-safe queue handles passed to spawned actor processes."""

    requests: InterprocessQueue[ProcessInferenceRequest]
    responses: tuple[InterprocessQueue[ProcessInferenceReply], ...]
    response_timeout_s: float


@dataclass(frozen=True, slots=True)
class ProcessInferenceStats:
    """Constant-size IPC service counters."""

    received_requests: int
    completed_requests: int
    failed_requests: int
    dropped_replies: int
    maximum_request_queue_size: int
    queue_wait_mean_ms: float
    queue_wait_maximum_ms: float
    running: bool


class ActorInferenceClient:
    """Blocking child-process client with correlation and finite timeout."""

    def __init__(self, actor_id: int, channels: ProcessInferenceChannels) -> None:
        if not 0 <= actor_id < len(channels.responses):
            raise ValueError("actor_id is outside configured response queues")
        self.actor_id = actor_id
        self._channels = channels
        self._counter = 0

    def infer(self, batch: RaggedBatch, policy_version: int) -> InferenceResult:
        """Submit one request with backpressure and wait only for its own reply."""
        if policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        request_id = f"{self.actor_id}:{os.getpid()}:{self._counter}"
        self._counter += 1
        request = ProcessInferenceRequest(
            request_id=request_id,
            actor_id=self.actor_id,
            policy_version=policy_version,
            batch=batch,
            submitted_ns=time.monotonic_ns(),
        )
        deadline = time.monotonic() + self._channels.response_timeout_s
        try:
            self._channels.requests.put(
                request,
                timeout=self._channels.response_timeout_s,
            )
        except Full as error:
            raise InferenceBridgeTimeout("inference request queue remained full") from error
        response_queue = self._channels.responses[self.actor_id]
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise InferenceBridgeTimeout(f"inference request {request_id} timed out")
            try:
                reply = response_queue.get(timeout=remaining)
            except Empty as error:
                raise InferenceBridgeTimeout(f"inference request {request_id} timed out") from error
            if reply.request_id != request_id:
                continue
            if reply.result is not None:
                return reply.result
            raise InferenceBridgeClosed(
                f"remote inference failed ({reply.error_type}): {reply.error_message}"
            )


class MultiprocessInferenceBridge:
    """Serve process requests on a dedicated event-loop thread and central model."""

    def __init__(
        self,
        server: InferenceServer,
        config: ProcessInferenceConfig,
        *,
        start_method: str = "spawn",
    ) -> None:
        if start_method not in mp.get_all_start_methods():
            raise ValueError(f"unsupported multiprocessing start method: {start_method}")
        context = mp.get_context(start_method)
        request_queue = cast(
            InterprocessQueue[ProcessInferenceRequest],
            context.Queue(maxsize=config.request_queue_capacity),
        )
        response_queues = tuple(
            cast(
                InterprocessQueue[ProcessInferenceReply],
                context.Queue(maxsize=config.response_queue_capacity),
            )
            for _ in range(config.actor_processes)
        )
        self.server = server
        self.config = config
        self.channels = ProcessInferenceChannels(
            requests=request_queue,
            responses=response_queues,
            response_timeout_s=config.actor_response_timeout_s,
        )
        self._stop = Event()
        self._ready = Event()
        self._thread: Thread | None = None
        self._startup_error: BaseException | None = None
        self._lock = Lock()
        self._received = 0
        self._completed = 0
        self._failed = 0
        self._dropped = 0
        self._maximum_queue_size = 0
        self._queue_wait_total_ms = 0.0
        self._queue_wait_maximum_ms = 0.0

    def start(self, timeout: float = 5.0) -> None:
        """Start the central service and fail explicitly if initialization stalls."""
        if self._thread is not None:
            raise RuntimeError("multiprocess inference bridge is already started")
        self._thread = Thread(target=self._thread_main, name="birddou-process-inference")
        self._thread.start()
        if not self._ready.wait(timeout):
            raise InferenceBridgeTimeout("inference bridge did not start in time")
        if self._startup_error is not None:
            raise InferenceBridgeClosed(f"inference bridge startup failed: {self._startup_error}")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop accepting IPC work and bound the service-thread join."""
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise InferenceBridgeTimeout("inference bridge did not stop in time")
        self._thread = None

    def stats(self) -> ProcessInferenceStats:
        """Return lock-consistent telemetry without retaining requests."""
        with self._lock:
            return ProcessInferenceStats(
                received_requests=self._received,
                completed_requests=self._completed,
                failed_requests=self._failed,
                dropped_replies=self._dropped,
                maximum_request_queue_size=self._maximum_queue_size,
                queue_wait_mean_ms=(
                    self._queue_wait_total_ms / self._received if self._received else 0.0
                ),
                queue_wait_maximum_ms=self._queue_wait_maximum_ms,
                running=self._thread is not None and self._thread.is_alive(),
            )

    def __enter__(self) -> MultiprocessInferenceBridge:
        self.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback_object: object,
    ) -> None:
        del exception_type, exception, traceback_object
        self.stop()

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            self._startup_error = error
            self._ready.set()

    async def _serve(self) -> None:
        await self.server.start()
        self._ready.set()
        live_tasks: set[asyncio.Task[None]] = set()
        try:
            while not self._stop.is_set():
                request = await asyncio.to_thread(self._get_request)
                if request is None:
                    continue
                requests = [request]
                while True:
                    try:
                        requests.append(self.channels.requests.get_nowait())
                    except Empty:
                        break
                self._record_requests(requests)
                for item in requests:
                    task = asyncio.create_task(self._serve_request(item))
                    live_tasks.add(task)
                    task.add_done_callback(live_tasks.discard)
            if live_tasks:
                await asyncio.gather(*live_tasks, return_exceptions=True)
        finally:
            await self.server.stop()
            self._fail_pending_requests()

    def _get_request(self) -> ProcessInferenceRequest | None:
        try:
            return self.channels.requests.get(timeout=self.config.bridge_poll_timeout_s)
        except Empty:
            return None

    async def _serve_request(self, request: ProcessInferenceRequest) -> None:
        try:
            result = await self.server.submit(request.batch, request.policy_version)
            reply = ProcessInferenceReply(request.request_id, result, None, None)
            failed = False
        except BaseException as error:
            reply = ProcessInferenceReply(
                request.request_id,
                None,
                type(error).__name__,
                str(error),
            )
            failed = True
        delivered = await asyncio.to_thread(self._deliver, request.actor_id, reply)
        with self._lock:
            if failed:
                self._failed += 1
            else:
                self._completed += 1
            if not delivered:
                self._dropped += 1

    def _deliver(self, actor_id: int, reply: ProcessInferenceReply) -> bool:
        if not 0 <= actor_id < len(self.channels.responses):
            return False
        try:
            self.channels.responses[actor_id].put(
                reply,
                timeout=self.config.bridge_poll_timeout_s,
            )
        except Full:
            return False
        return True

    def _record_requests(self, requests: list[ProcessInferenceRequest]) -> None:
        now_ns = time.monotonic_ns()
        waits = [max(0.0, (now_ns - request.submitted_ns) / 1_000_000.0) for request in requests]
        raw_queue = cast(object, self.channels.requests)
        size_method = operator.attrgetter("qsize")(raw_queue)
        queue_size = 0
        if callable(size_method):
            try:
                queue_size = int(size_method())
            except (NotImplementedError, OSError):
                pass
        with self._lock:
            self._received += len(requests)
            self._queue_wait_total_ms += sum(waits)
            self._queue_wait_maximum_ms = max(self._queue_wait_maximum_ms, *waits)
            self._maximum_queue_size = max(self._maximum_queue_size, queue_size)

    def _fail_pending_requests(self) -> None:
        while True:
            try:
                request = self.channels.requests.get_nowait()
            except Empty:
                return
            reply = ProcessInferenceReply(
                request.request_id,
                None,
                "InferenceBridgeClosed",
                "inference bridge stopped",
            )
            delivered = self._deliver(request.actor_id, reply)
            with self._lock:
                self._failed += 1
                if not delivered:
                    self._dropped += 1


__all__ = (
    "ActorInferenceClient",
    "InferenceBridgeClosed",
    "InferenceBridgeTimeout",
    "MultiprocessInferenceBridge",
    "ProcessInferenceChannels",
    "ProcessInferenceConfig",
    "ProcessInferenceReply",
    "ProcessInferenceRequest",
    "ProcessInferenceStats",
)
