"""Spawn-safe multi-process actor lifecycle and bounded trajectory transport."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import operator
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from pathlib import Path
from queue import Empty, Full
from typing import Generic, Protocol, TypeAlias, TypeVar, cast

ACTOR_SUPERVISOR_SCHEMA_VERSION = 1

PayloadT = TypeVar("PayloadT")
TrajectoryT = TypeVar("TrajectoryT")


class InterprocessQueue(Protocol[TrajectoryT]):
    """Minimal bounded multiprocessing queue surface exposed to an actor."""

    def put(
        self,
        item: TrajectoryT,
        block: bool = True,
        timeout: float | None = None,
    ) -> None: ...

    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> TrajectoryT: ...

    def get_nowait(self) -> TrajectoryT: ...


class StopSignal(Protocol):
    """Pickle-safe cooperative-stop event surface."""

    def is_set(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool: ...


@dataclass(frozen=True, slots=True)
class ActorSupervisorConfig:
    """Process count, environment fan-out, restart, and queue safety limits."""

    schema_version: int = ACTOR_SUPERVISOR_SCHEMA_VERSION
    actor_processes: int = 8
    envs_per_actor: int = 32
    trajectory_queue_capacity: int = 256
    max_restarts_per_actor: int = 2
    process_join_timeout_s: float = 5.0
    queue_poll_timeout_s: float = 0.05
    start_method: str = "spawn"

    def __post_init__(self) -> None:
        if self.schema_version != ACTOR_SUPERVISOR_SCHEMA_VERSION:
            raise ValueError("unsupported actor supervisor schema")
        if min(self.actor_processes, self.envs_per_actor, self.trajectory_queue_capacity) <= 0:
            raise ValueError("actor counts and queue capacity must be positive")
        if self.max_restarts_per_actor < 0:
            raise ValueError("max_restarts_per_actor must be non-negative")
        for name, value in (
            ("process_join_timeout_s", self.process_join_timeout_s),
            ("queue_poll_timeout_s", self.queue_poll_timeout_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.start_method not in mp.get_all_start_methods():
            raise ValueError(f"unsupported multiprocessing start method: {self.start_method}")


def load_actor_supervisor_config(path: Path) -> ActorSupervisorConfig:
    """Load the JSON-subset YAML process topology used by training."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
        raise ValueError("actor supervisor config must be a string-keyed mapping")
    values = cast(Mapping[str, object], raw)
    return ActorSupervisorConfig(
        schema_version=_integer(values, "schema_version"),
        actor_processes=_integer(values, "actor_processes"),
        envs_per_actor=_integer(values, "envs_per_actor"),
        trajectory_queue_capacity=_integer(values, "trajectory_queue_capacity"),
        max_restarts_per_actor=_integer(values, "max_restarts_per_actor"),
        process_join_timeout_s=_number(values, "process_join_timeout_s"),
        queue_poll_timeout_s=_number(values, "queue_poll_timeout_s"),
        start_method=_string(values, "start_method"),
    )


@dataclass(frozen=True, slots=True)
class ActorWorkerContext(Generic[TrajectoryT]):
    """Stable identity and bounded IPC channels supplied to one actor process."""

    actor_id: int
    generation: int
    envs_per_actor: int
    trajectories: InterprocessQueue[TrajectoryT]
    stop: StopSignal


ActorWorker: TypeAlias = Callable[[ActorWorkerContext[TrajectoryT], PayloadT], None]


@dataclass(frozen=True, slots=True)
class ActorFailure:
    """Serializable child-process failure diagnostic."""

    actor_id: int
    generation: int
    exception_type: str
    message: str
    traceback: str


@dataclass(frozen=True, slots=True)
class ActorSupervisorStats:
    """Constant-size lifecycle counters for monitoring and checkpoints."""

    configured_processes: int
    live_processes: int
    clean_exits: int
    crash_exits: int
    restart_count: int
    permanently_failed_actors: tuple[int, ...]
    produced_trajectories: int
    maximum_observed_queue_size: int
    running: bool


class ActorProcessFailure(RuntimeError):
    """An actor exhausted its bounded restart budget."""


@dataclass(frozen=True, slots=True)
class _ChildExit:
    actor_id: int
    generation: int
    clean: bool
    failure: ActorFailure | None


class ActorProcessSupervisor(Generic[PayloadT, TrajectoryT]):
    """Own actor processes, restart crashes, and never expose an unbounded queue."""

    def __init__(
        self,
        config: ActorSupervisorConfig,
        worker: ActorWorker[TrajectoryT, PayloadT],
        payload: PayloadT,
    ) -> None:
        self.config = config
        self._worker = worker
        self._payload = payload
        self._context: BaseContext = mp.get_context(config.start_method)
        self._trajectory_queue = cast(
            InterprocessQueue[TrajectoryT],
            self._context.Queue(maxsize=config.trajectory_queue_capacity),
        )
        self._status_queue = cast(
            InterprocessQueue[_ChildExit],
            self._context.Queue(maxsize=max(4, config.actor_processes * 4)),
        )
        self._stop = self._context.Event()
        self._processes: dict[int, BaseProcess] = {}
        self._generations = [0] * config.actor_processes
        self._restarts = [0] * config.actor_processes
        self._handled: set[tuple[int, int]] = set()
        self._clean_exits = 0
        self._crash_exits = 0
        self._permanently_failed: set[int] = set()
        self._produced_trajectories = 0
        self._maximum_queue_size = 0
        self._running = False

    def start(self) -> None:
        """Start every configured actor exactly once."""
        if self._running:
            raise RuntimeError("actor supervisor is already running")
        if self._stop.is_set():
            raise RuntimeError("actor supervisor cannot be restarted after shutdown")
        self._running = True
        for actor_id in range(self.config.actor_processes):
            self._spawn(actor_id)

    def poll(self) -> tuple[ActorFailure, ...]:
        """Reap exits, restart crashes within budget, and surface diagnostics."""
        failures: list[ActorFailure] = []
        while True:
            try:
                child_exit = self._status_queue.get_nowait()
            except Empty:
                break
            self._handle_exit(child_exit, failures)

        for actor_id, process in tuple(self._processes.items()):
            if process.exitcode is None:
                continue
            key = (actor_id, self._generations[actor_id])
            if key in self._handled:
                continue
            synthetic = ActorFailure(
                actor_id=actor_id,
                generation=self._generations[actor_id],
                exception_type="ProcessExit",
                message=f"actor exited with code {process.exitcode} before reporting status",
                traceback="",
            )
            self._handle_exit(
                _ChildExit(actor_id, self._generations[actor_id], False, synthetic),
                failures,
            )
        self._observe_queue_size()
        return tuple(failures)

    def get_trajectory(self, timeout: float | None = None) -> TrajectoryT:
        """Read one trajectory while continually enforcing actor liveness."""
        if not self._running:
            raise RuntimeError("actor supervisor is not running")
        wait = self.config.queue_poll_timeout_s if timeout is None else timeout
        if not math.isfinite(wait) or wait < 0.0:
            raise ValueError("trajectory timeout must be finite and non-negative")
        deadline = time.monotonic() + wait
        while True:
            failures = self.poll()
            if self._permanently_failed:
                detail = failures[-1].message if failures else "restart budget exhausted"
                raise ActorProcessFailure(
                    f"actors {sorted(self._permanently_failed)} permanently failed: {detail}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise Empty
            try:
                item = self._trajectory_queue.get(timeout=min(remaining, 0.05))
            except Empty:
                if not any(process.is_alive() for process in self._processes.values()):
                    raise Empty from None
                continue
            self._produced_trajectories += 1
            self._observe_queue_size()
            return item

    def terminate_actor(self, actor_id: int) -> None:
        """Force one actor down; the next poll applies the normal restart policy."""
        process = self._processes.get(actor_id)
        if process is None:
            raise KeyError(actor_id)
        process.terminate()
        process.join(self.config.process_join_timeout_s)

    def shutdown(self) -> None:
        """Request cooperative shutdown, then bound join time and terminate stragglers."""
        if not self._running:
            return
        self._stop.set()
        for process in self._processes.values():
            process.join(self.config.process_join_timeout_s)
        for process in self._processes.values():
            if process.is_alive():
                process.terminate()
                process.join(self.config.process_join_timeout_s)
            process.close()
        self._processes.clear()
        self._running = False

    def stats(self) -> ActorSupervisorStats:
        """Return bounded operational counters without copying trajectories."""
        self.poll()
        return ActorSupervisorStats(
            configured_processes=self.config.actor_processes,
            live_processes=sum(process.is_alive() for process in self._processes.values()),
            clean_exits=self._clean_exits,
            crash_exits=self._crash_exits,
            restart_count=sum(self._restarts),
            permanently_failed_actors=tuple(sorted(self._permanently_failed)),
            produced_trajectories=self._produced_trajectories,
            maximum_observed_queue_size=self._maximum_queue_size,
            running=self._running,
        )

    def __enter__(self) -> ActorProcessSupervisor[PayloadT, TrajectoryT]:
        self.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback_object: object,
    ) -> None:
        del exception_type, exception, traceback_object
        self.shutdown()

    def _spawn(self, actor_id: int) -> None:
        generation = self._generations[actor_id]
        process_factory = cast(
            Callable[..., BaseProcess], operator.attrgetter("Process")(self._context)
        )
        process = process_factory(
            target=_actor_entry,
            args=(
                actor_id,
                generation,
                self.config.envs_per_actor,
                self._trajectory_queue,
                self._status_queue,
                self._stop,
                self._worker,
                self._payload,
            ),
            name=f"birddou-actor-{actor_id}-g{generation}",
        )
        process.start()
        self._processes[actor_id] = process

    def _handle_exit(self, child_exit: _ChildExit, failures: list[ActorFailure]) -> None:
        key = (child_exit.actor_id, child_exit.generation)
        if key in self._handled or child_exit.generation != self._generations[child_exit.actor_id]:
            return
        self._handled.add(key)
        process = self._processes[child_exit.actor_id]
        process.join(self.config.process_join_timeout_s)
        if child_exit.clean or self._stop.is_set():
            self._clean_exits += 1
            return
        self._crash_exits += 1
        if child_exit.failure is not None:
            failures.append(child_exit.failure)
        if self._restarts[child_exit.actor_id] >= self.config.max_restarts_per_actor:
            self._permanently_failed.add(child_exit.actor_id)
            return
        process.close()
        self._restarts[child_exit.actor_id] += 1
        self._generations[child_exit.actor_id] += 1
        self._spawn(child_exit.actor_id)

    def _observe_queue_size(self) -> None:
        raw_queue = cast(object, self._trajectory_queue)
        size_method = getattr(raw_queue, "qsize", None)
        if not callable(size_method):
            return
        try:
            size = int(size_method())
        except (NotImplementedError, OSError):
            return
        self._maximum_queue_size = max(self._maximum_queue_size, size)


def _actor_entry(
    actor_id: int,
    generation: int,
    envs_per_actor: int,
    trajectories: InterprocessQueue[TrajectoryT],
    statuses: InterprocessQueue[_ChildExit],
    stop: StopSignal,
    worker: ActorWorker[TrajectoryT, PayloadT],
    payload: PayloadT,
) -> None:
    context = ActorWorkerContext(actor_id, generation, envs_per_actor, trajectories, stop)
    try:
        worker(context, payload)
    except BaseException as error:
        failure = ActorFailure(
            actor_id=actor_id,
            generation=generation,
            exception_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
        )
        try:
            statuses.put(_ChildExit(actor_id, generation, False, failure), timeout=0.5)
        except Full:
            pass
        raise
    else:
        try:
            statuses.put(_ChildExit(actor_id, generation, True, None), timeout=0.5)
        except Full:
            pass


def _integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"actor supervisor config {key} must be an integer")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"actor supervisor config {key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"actor supervisor config {key} must be finite")
    return result


def _string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"actor supervisor config {key} must be a non-empty string")
    return value


__all__ = (
    "ACTOR_SUPERVISOR_SCHEMA_VERSION",
    "ActorFailure",
    "ActorProcessFailure",
    "ActorProcessSupervisor",
    "ActorSupervisorConfig",
    "ActorSupervisorStats",
    "ActorWorker",
    "ActorWorkerContext",
    "InterprocessQueue",
    "StopSignal",
    "load_actor_supervisor_config",
)
