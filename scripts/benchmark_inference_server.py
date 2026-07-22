"""Bounded scheduler benchmark for the M7 ragged inference server."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from birddou import PyDdzEnv, load_rule_config
from birddou.actors import InferenceServer, InferenceServerConfig
from birddou.features import FeatureConfig, RaggedBatch, encode_ragged_batch
from birddou.models.segment_ops import segment_softmax

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class _Output:
    policy_logit: Tensor
    policy_log_probability: Tensor
    policy_probability: Tensor
    mc_q: Tensor
    win_logit: Tensor
    expected_score: Tensor


class _SchedulerProbeModel:
    """Deterministic tensor work that isolates scheduling from model throughput."""

    def eval(self) -> _SchedulerProbeModel:
        return self

    def __call__(self, batch: RaggedBatch) -> _Output:
        logits = torch.arange(batch.action_count, dtype=torch.float32) * 0.001
        probability = segment_softmax(logits, batch.action_offsets)
        return _Output(
            policy_logit=logits,
            policy_log_probability=probability.log(),
            policy_probability=probability,
            mc_q=logits,
            win_logit=logits,
            expected_score=logits,
        )


def _batch(seed: int) -> RaggedBatch:
    rules = load_rule_config(REPOSITORY_ROOT / "configs/rules/douzero_post_bid.yaml")
    environment = PyDdzEnv()
    environment.reset(seed, rules)
    return encode_ragged_batch(
        (environment.observe(environment.current_player),),
        (environment.legal_actions(),),
        rules,
        config=FeatureConfig(decomposition_features=False),
    )


async def _run(request_count: int, concurrency: int, queue_capacity: int) -> dict[str, object]:
    batch = _batch(7_070)
    server = InferenceServer(
        InferenceServerConfig(
            max_inference_states=128,
            max_inference_actions=max(8_192, batch.action_count * 128),
            microbatch_wait_ms=2.0,
            max_queue_requests=queue_capacity,
        )
    )
    server.register_model(1, _SchedulerProbeModel())
    await server.start()

    next_request = 0
    request_lock = asyncio.Lock()

    async def actor() -> None:
        nonlocal next_request
        while True:
            async with request_lock:
                if next_request >= request_count:
                    return
                next_request += 1
            await server.submit(batch, 1)

    tracemalloc.start()
    started = time.perf_counter()
    await asyncio.gather(*(actor() for _ in range(concurrency)))
    elapsed = time.perf_counter() - started
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stats = server.stats()
    await server.stop()
    return {
        "requests": request_count,
        "concurrency": concurrency,
        "elapsed_seconds": elapsed,
        "requests_per_second": request_count / elapsed,
        "current_traced_bytes": current_bytes,
        "peak_traced_bytes": peak_bytes,
        "actions_per_request": batch.action_count,
        "server": asdict(stats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=10_000)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--queue-capacity", type=int, default=64)
    arguments = parser.parse_args()
    if min(arguments.requests, arguments.concurrency, arguments.queue_capacity) <= 0:
        parser.error("requests, concurrency, and queue-capacity must be positive")
    report = asyncio.run(_run(arguments.requests, arguments.concurrency, arguments.queue_capacity))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
