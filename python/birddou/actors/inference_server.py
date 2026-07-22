"""Bounded, versioned, action-aware asynchronous inference batching."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar, cast

import torch
from torch import Tensor

from birddou.belief.data import concatenate_ragged_batches
from birddou.features.ragged import RaggedBatch
from birddou.models.segment_ops import segment_logsumexp, segment_softmax

INFERENCE_SERVER_SCHEMA_VERSION = 1


class InferenceServerClosed(RuntimeError):
    """The inference boundary no longer accepts work."""


class UnknownPolicyVersion(KeyError):
    """A request names a policy snapshot not registered with the server."""


@dataclass(frozen=True, slots=True)
class InferenceServerConfig:
    """Hard memory/backpressure and microbatch limits."""

    schema_version: int = INFERENCE_SERVER_SCHEMA_VERSION
    max_inference_states: int = 128
    max_inference_actions: int = 8_192
    microbatch_wait_ms: float = 2.0
    max_queue_requests: int = 1_024
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.schema_version != INFERENCE_SERVER_SCHEMA_VERSION:
            raise ValueError("unsupported inference server config schema")
        if (
            min(
                self.max_inference_states,
                self.max_inference_actions,
                self.max_queue_requests,
            )
            <= 0
        ):
            raise ValueError("inference limits must be positive")
        if not math.isfinite(self.microbatch_wait_ms) or self.microbatch_wait_ms < 0.0:
            raise ValueError("microbatch_wait_ms must be finite and non-negative")
        if not self.device:
            raise ValueError("inference device cannot be empty")


def load_inference_server_config(path: Path) -> InferenceServerConfig:
    """Load a versioned JSON-subset YAML inference-server configuration."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
        raise ValueError("inference server config must be a string-keyed mapping")
    values = cast(Mapping[str, object], raw)
    return InferenceServerConfig(
        schema_version=_config_integer(values, "schema_version"),
        max_inference_states=_config_integer(values, "max_inference_states"),
        max_inference_actions=_config_integer(values, "max_inference_actions"),
        microbatch_wait_ms=_config_number(values, "microbatch_wait_ms"),
        max_queue_requests=_config_integer(values, "max_queue_requests"),
        device=_config_string(values, "device"),
    )


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """One request's ragged action outputs, returned on CPU."""

    policy_version: int
    action_offsets: Tensor
    policy_logit: Tensor
    policy_log_probability: Tensor
    policy_probability: Tensor
    mc_q: Tensor
    win_logit: Tensor
    expected_score: Tensor

    def __post_init__(self) -> None:
        if (
            self.action_offsets.dtype != torch.int64
            or self.action_offsets.ndim != 1
            or self.action_offsets.numel() < 2
            or int(self.action_offsets[0].item()) != 0
        ):
            raise ValueError("inference result offsets must be int64 [B+1] from zero")
        action_count = int(self.action_offsets[-1].item())
        fields = (
            self.policy_logit,
            self.policy_log_probability,
            self.policy_probability,
            self.mc_q,
            self.win_logit,
            self.expected_score,
        )
        if self.policy_version < 0:
            raise ValueError("inference result version must be non-negative")
        if torch.any(torch.diff(self.action_offsets) <= 0):
            raise ValueError("every inference state must have at least one action")
        if any(value.shape != (action_count,) for value in fields):
            raise ValueError("inference action outputs do not match offsets")
        if any(
            not value.is_floating_point() or not torch.isfinite(value).all() for value in fields
        ):
            raise ValueError("inference action outputs must be finite and floating")


@dataclass(frozen=True, slots=True)
class InferenceServerStats:
    """Constant-size operational metrics; no request history is retained."""

    queue_depth: int
    queue_capacity: int
    maximum_queue_depth: int
    completed_requests: int
    failed_requests: int
    cancelled_requests: int
    batch_count: int
    state_count: int
    action_count: int
    maximum_batch_states: int
    maximum_batch_actions: int
    registered_versions: tuple[int, ...]
    running: bool
    closing: bool


class _PolicyOutput(Protocol):
    policy_logit: Tensor
    policy_log_probability: Tensor
    policy_probability: Tensor
    mc_q: Tensor
    win_logit: Tensor
    expected_score: Tensor


class InferenceModel(Protocol):
    """Model surface required by the server."""

    def eval(self) -> InferenceModel: ...

    def __call__(self, batch: RaggedBatch) -> object: ...


@dataclass(slots=True)
class _Request:
    batch: RaggedBatch
    policy_version: int
    future: asyncio.Future[InferenceResult]


class InferenceServer:
    """Run exact ragged microbatches while preserving requested policy versions."""

    def __init__(self, config: InferenceServerConfig | None = None) -> None:
        self.config = config if config is not None else InferenceServerConfig()
        self._queue: asyncio.Queue[_Request] = asyncio.Queue(maxsize=self.config.max_queue_requests)
        self._models: dict[int, InferenceModel] = {}
        self._worker: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._closing = False
        self._deferred: _Request | None = None
        self._maximum_queue_depth = 0
        self._completed_requests = 0
        self._failed_requests = 0
        self._cancelled_requests = 0
        self._batch_count = 0
        self._state_count = 0
        self._action_count = 0
        self._maximum_batch_states = 0
        self._maximum_batch_actions = 0

    def register_model(
        self,
        policy_version: int,
        model: InferenceModel,
        *,
        replace: bool = False,
    ) -> None:
        """Register an immutable policy version used to route future requests."""
        if self._closing:
            raise InferenceServerClosed("inference server is closing")
        if policy_version < 0:
            raise ValueError("policy version must be non-negative")
        if policy_version in self._models and not replace:
            raise ValueError(f"policy version {policy_version} is already registered")
        self._models[policy_version] = model.eval()

    def unregister_model(self, policy_version: int) -> None:
        """Remove a policy snapshot; queued requests then fail explicitly."""
        if policy_version not in self._models:
            raise UnknownPolicyVersion(policy_version)
        del self._models[policy_version]

    async def start(self) -> None:
        """Start one batching worker; repeated calls are harmless."""
        if self._closing:
            raise InferenceServerClosed("inference server is closing")
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="birddou-inference")

    async def submit(self, batch: RaggedBatch, policy_version: int) -> InferenceResult:
        """Submit with async backpressure and cancel safely on actor disconnect."""
        if self._closing:
            raise InferenceServerClosed("inference server is closing")
        if policy_version not in self._models:
            raise UnknownPolicyVersion(policy_version)
        if batch.batch_size > self.config.max_inference_states:
            raise ValueError("request exceeds max_inference_states")
        if batch.action_count > self.config.max_inference_actions:
            raise ValueError("request exceeds max_inference_actions")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[InferenceResult] = loop.create_future()
        request = _Request(batch, policy_version, future)
        await self._enqueue(request)
        try:
            return await future
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()
            raise

    async def stop(self) -> None:
        """Stop accepting work and resolve every queued or blocked request."""
        if not self._closing:
            self._closing = True
            self._stop_event.set()
        if self._worker is not None:
            await self._worker
            self._worker = None
        else:
            self._fail_pending(InferenceServerClosed("inference server stopped"))

    def stats(self) -> InferenceServerStats:
        """Return a scalar snapshot suitable for long-running monitoring."""
        return InferenceServerStats(
            queue_depth=self._queue.qsize(),
            queue_capacity=self.config.max_queue_requests,
            maximum_queue_depth=self._maximum_queue_depth,
            completed_requests=self._completed_requests,
            failed_requests=self._failed_requests,
            cancelled_requests=self._cancelled_requests,
            batch_count=self._batch_count,
            state_count=self._state_count,
            action_count=self._action_count,
            maximum_batch_states=self._maximum_batch_states,
            maximum_batch_actions=self._maximum_batch_actions,
            registered_versions=tuple(sorted(self._models)),
            running=self._worker is not None and not self._worker.done(),
            closing=self._closing,
        )

    async def __aenter__(self) -> InferenceServer:
        await self.start()
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        del exception_type, exception, traceback
        await self.stop()

    async def _enqueue(self, request: _Request) -> None:
        put_task = asyncio.create_task(self._queue.put(request))
        stop_task = asyncio.create_task(self._stop_event.wait())
        try:
            done, _ = await asyncio.wait(
                (put_task, stop_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in done or self._closing:
                if put_task.done() and not put_task.cancelled():
                    await put_task
                    request.future.cancel()
                    self._fail_pending(
                        InferenceServerClosed("inference server stopped during submit")
                    )
                else:
                    put_task.cancel()
                    await _ignore_cancel(put_task)
                raise InferenceServerClosed("inference server stopped during submit")
            if put_task in done:
                await put_task
                self._maximum_queue_depth = max(
                    self._maximum_queue_depth,
                    self._queue.qsize(),
                )
                return
            raise RuntimeError("inference enqueue wait ended without a completed event")
        finally:
            stop_task.cancel()
            await _ignore_cancel(stop_task)

    async def _run(self) -> None:
        try:
            while not self._closing:
                first = self._take_deferred()
                if first is None:
                    first = await self._wait_for_request(None)
                if first is None:
                    continue
                requests = await self._collect_microbatch(first)
                if self._closing:
                    self._fail_requests(
                        requests,
                        InferenceServerClosed("inference server stopped"),
                    )
                    for _ in requests:
                        self._queue.task_done()
                    continue
                self._execute(requests)
        finally:
            self._fail_pending(InferenceServerClosed("inference server stopped"))

    def _take_deferred(self) -> _Request | None:
        request = self._deferred
        self._deferred = None
        return request

    async def _collect_microbatch(self, first: _Request) -> list[_Request]:
        requests = [first]
        state_count = first.batch.batch_size
        action_count = first.batch.action_count
        deadline = time.monotonic() + self.config.microbatch_wait_ms / 1_000.0
        while not self._closing:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            candidate = await self._wait_for_request(remaining)
            if candidate is None:
                break
            next_states = state_count + candidate.batch.batch_size
            next_actions = action_count + candidate.batch.action_count
            if (
                candidate.policy_version != first.policy_version
                or next_states > self.config.max_inference_states
                or next_actions > self.config.max_inference_actions
            ):
                self._deferred = candidate
                break
            requests.append(candidate)
            state_count = next_states
            action_count = next_actions
            if (
                state_count == self.config.max_inference_states
                or action_count == self.config.max_inference_actions
            ):
                break
        return requests

    async def _wait_for_request(self, timeout: float | None) -> _Request | None:
        get_task = asyncio.create_task(self._queue.get())
        stop_task = asyncio.create_task(self._stop_event.wait())
        try:
            done, _ = await asyncio.wait(
                (get_task, stop_task),
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if get_task in done:
                return await get_task
            get_task.cancel()
            await _ignore_cancel(get_task)
            return None
        finally:
            stop_task.cancel()
            await _ignore_cancel(stop_task)

    def _execute(self, requests: list[_Request]) -> None:
        live = [request for request in requests if not request.future.cancelled()]
        self._cancelled_requests += len(requests) - len(live)
        if not live:
            for _ in requests:
                self._queue.task_done()
            return
        version = live[0].policy_version
        model = self._models.get(version)
        if model is None:
            self._fail_requests(live, UnknownPolicyVersion(version))
            for _ in requests:
                self._queue.task_done()
            return
        combined = concatenate_ragged_batches([request.batch for request in live])
        try:
            with torch.inference_mode():
                raw_output = model(combined.to(self.config.device))
            output = _extract_policy_output(raw_output)
            _validate_policy_output(output, combined.action_count)
            policy_log_normalizer = segment_logsumexp(
                output.policy_logit, combined.action_offsets.to(output.policy_logit.device)
            )
            action_state_index = combined.action_state_index.to(output.policy_logit.device)
            policy_log_probability = output.policy_logit - policy_log_normalizer[action_state_index]
            policy_probability = segment_softmax(
                output.policy_logit, combined.action_offsets.to(output.policy_logit.device)
            )
            action_base = 0
            for request in live:
                action_end = action_base + request.batch.action_count
                result = InferenceResult(
                    policy_version=version,
                    action_offsets=request.batch.action_offsets.detach().cpu().clone(),
                    policy_logit=_cpu_slice(output.policy_logit, action_base, action_end),
                    policy_log_probability=_cpu_slice(
                        policy_log_probability, action_base, action_end
                    ),
                    policy_probability=_cpu_slice(policy_probability, action_base, action_end),
                    mc_q=_cpu_slice(output.mc_q, action_base, action_end),
                    win_logit=_cpu_slice(output.win_logit, action_base, action_end),
                    expected_score=_cpu_slice(output.expected_score, action_base, action_end),
                )
                if not request.future.done():
                    request.future.set_result(result)
                    self._completed_requests += 1
                action_base = action_end
            self._batch_count += 1
            self._state_count += combined.batch_size
            self._action_count += combined.action_count
            self._maximum_batch_states = max(self._maximum_batch_states, combined.batch_size)
            self._maximum_batch_actions = max(self._maximum_batch_actions, combined.action_count)
        except Exception as error:
            self._fail_requests(live, error)
        finally:
            for _ in requests:
                self._queue.task_done()

    def _fail_requests(self, requests: list[_Request], error: BaseException) -> None:
        for request in requests:
            if request.future.cancelled():
                self._cancelled_requests += 1
            elif not request.future.done():
                request.future.set_exception(error)
                self._failed_requests += 1

    def _fail_pending(self, error: BaseException) -> None:
        pending: list[_Request] = []
        if self._deferred is not None:
            pending.append(self._deferred)
            self._deferred = None
        while True:
            try:
                pending.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        self._fail_requests(pending, error)
        for _ in pending:
            self._queue.task_done()


def _extract_policy_output(output: object) -> _PolicyOutput:
    nested = getattr(output, "policy", None)
    return cast(_PolicyOutput, output if nested is None else nested)


def _validate_policy_output(output: _PolicyOutput, action_count: int) -> None:
    fields = (
        output.policy_logit,
        output.policy_log_probability,
        output.policy_probability,
        output.mc_q,
        output.win_logit,
        output.expected_score,
    )
    if any(value.shape != (action_count,) for value in fields):
        raise ValueError("model output does not match inference action count")
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in fields):
        raise ValueError("model output must be finite and floating")


def _cpu_slice(value: Tensor, start: int, end: int) -> Tensor:
    return value[start:end].detach().cpu().clone()


CancelResultT = TypeVar("CancelResultT")


async def _ignore_cancel(task: asyncio.Task[CancelResultT]) -> None:
    try:
        await task
    except asyncio.CancelledError:
        pass


def _config_integer(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"inference server config {key} must be an integer")
    return value


def _config_number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"inference server config {key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"inference server config {key} must be finite")
    return numeric


def _config_string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"inference server config {key} must be a non-empty string")
    return value


__all__ = (
    "INFERENCE_SERVER_SCHEMA_VERSION",
    "InferenceModel",
    "InferenceResult",
    "InferenceServer",
    "InferenceServerClosed",
    "InferenceServerConfig",
    "InferenceServerStats",
    "UnknownPolicyVersion",
    "load_inference_server_config",
)
