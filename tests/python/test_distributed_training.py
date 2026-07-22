"""Backpressure, cancellation, versioning, replay, and fairness tests for M7."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Full
from threading import Event, Thread

import pytest
import torch
from torch import Tensor

from birddou import PyDdzEnv, load_rule_config
from birddou.actors import (
    BoundedTrajectoryQueue,
    InferenceServer,
    InferenceServerClosed,
    InferenceServerConfig,
    QueueClosed,
    load_inference_server_config,
)
from birddou.features import FeatureConfig, RaggedBatch, encode_ragged_batch
from birddou.models.segment_ops import segment_softmax
from birddou.rl import (
    ComparisonRun,
    EpisodeMeta,
    FairTrainingBudget,
    HybridLossConfig,
    LearnerTrajectoryBatch,
    PolicyLagMonitor,
    TrainerMode,
    Trajectory,
    TrajectoryReplay,
    Transition,
    VTraceConfig,
    bird_dou_learner_step,
    load_fair_comparison_config,
    reconstruct_states,
    validate_fair_comparison,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml"


@dataclass(frozen=True, slots=True)
class _FakeOutput:
    policy_logit: Tensor
    policy_log_probability: Tensor
    policy_probability: Tensor
    mc_q: Tensor
    win_logit: Tensor
    expected_score: Tensor


class _FakeModel:
    def __init__(self, bias: float) -> None:
        self.bias = bias

    def eval(self) -> _FakeModel:
        return self

    def __call__(self, batch: RaggedBatch) -> _FakeOutput:
        logits = torch.arange(batch.action_count, dtype=torch.float32) * 0.01 + self.bias
        probability = segment_softmax(logits, batch.action_offsets)
        return _FakeOutput(
            policy_logit=logits,
            policy_log_probability=probability.log(),
            policy_probability=probability,
            mc_q=logits + 1.0,
            win_logit=logits - 1.0,
            expected_score=logits * 2.0,
        )


def _single_state_batch(seed: int) -> RaggedBatch:
    rules = load_rule_config(RULES_PATH)
    environment = PyDdzEnv()
    environment.reset(seed, rules)
    return encode_ragged_batch(
        (environment.observe(environment.current_player),),
        (environment.legal_actions(),),
        rules,
        config=FeatureConfig(decomposition_features=False),
    )


def _assert_segment_probabilities(result_offsets: Tensor, probability: Tensor) -> None:
    offsets = result_offsets.tolist()
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        torch.testing.assert_close(probability[start:end].sum(), torch.tensor(1.0))


def test_shared_queue_backpressure_close_and_drain_are_bounded() -> None:
    queue: BoundedTrajectoryQueue[int] = BoundedTrajectoryQueue(1)
    queue.put(1)
    with pytest.raises(Full):
        queue.put(2, block=False)

    entered = Event()
    closed = Event()

    def blocked_producer() -> None:
        entered.set()
        try:
            queue.put(2, timeout=5.0)
        except QueueClosed:
            closed.set()

    thread = Thread(target=blocked_producer)
    thread.start()
    assert entered.wait(1.0)
    discarded = queue.close(discard=True)
    thread.join(1.0)
    assert not thread.is_alive()
    assert closed.is_set()
    assert discarded == (1,)
    assert queue.stats().maximum_size == 1
    with pytest.raises(QueueClosed):
        queue.get()

    empty: BoundedTrajectoryQueue[int] = BoundedTrajectoryQueue(1)
    with pytest.raises(Empty):
        empty.get(block=False)


def test_inference_server_microbatches_actual_actions_and_routes_versions() -> None:
    async def scenario() -> None:
        first = _single_state_batch(7101)
        second = _single_state_batch(7102)
        config = InferenceServerConfig(
            max_inference_states=2,
            max_inference_actions=first.action_count + second.action_count,
            microbatch_wait_ms=20.0,
            max_queue_requests=4,
        )
        server = InferenceServer(config)
        server.register_model(3, _FakeModel(0.0))
        server.register_model(4, _FakeModel(10.0))
        await server.start()
        left_task = asyncio.create_task(server.submit(first, 3))
        right_task = asyncio.create_task(server.submit(second, 3))
        left, right = await asyncio.wait_for(asyncio.gather(left_task, right_task), timeout=2.0)
        assert left.policy_version == right.policy_version == 3
        assert left.policy_logit.numel() == first.action_count
        assert right.policy_logit.numel() == second.action_count
        _assert_segment_probabilities(left.action_offsets, left.policy_probability)
        _assert_segment_probabilities(right.action_offsets, right.policy_probability)
        stats = server.stats()
        assert stats.batch_count == 1
        assert stats.maximum_batch_states == 2
        assert stats.maximum_batch_actions == first.action_count + second.action_count

        old_task = asyncio.create_task(server.submit(first, 3))
        new_task = asyncio.create_task(server.submit(second, 4))
        old, new = await asyncio.wait_for(asyncio.gather(old_task, new_task), timeout=2.0)
        assert old.policy_version == 3
        assert new.policy_version == 4
        assert new.policy_logit[0] - old.policy_logit[0] == 10.0
        assert server.stats().batch_count == 3
        await asyncio.wait_for(server.stop(), timeout=2.0)

    asyncio.run(scenario())


def test_inference_queue_backpressure_cancellation_and_shutdown_do_not_hang() -> None:
    async def backpressure_scenario() -> None:
        batch = _single_state_batch(7110)
        server = InferenceServer(
            InferenceServerConfig(
                max_inference_states=2,
                max_inference_actions=batch.action_count * 2,
                microbatch_wait_ms=1.0,
                max_queue_requests=1,
            )
        )
        server.register_model(1, _FakeModel(0.0))
        first = asyncio.create_task(server.submit(batch, 1))
        for _ in range(3):
            await asyncio.sleep(0)
        assert server.stats().queue_depth == 1
        second = asyncio.create_task(server.submit(batch, 1))
        for _ in range(3):
            await asyncio.sleep(0)
        assert not second.done()
        assert server.stats().maximum_queue_depth <= 1
        await server.start()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=2.0)

        cancelled = asyncio.create_task(server.submit(batch, 1))
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        await asyncio.wait_for(server.stop(), timeout=2.0)

    async def stopped_before_start_scenario() -> None:
        batch = _single_state_batch(7111)
        server = InferenceServer(
            InferenceServerConfig(
                max_inference_states=1,
                max_inference_actions=batch.action_count,
                max_queue_requests=1,
            )
        )
        server.register_model(1, _FakeModel(0.0))
        queued = asyncio.create_task(server.submit(batch, 1))
        for _ in range(3):
            await asyncio.sleep(0)
        blocked = asyncio.create_task(server.submit(batch, 1))
        for _ in range(3):
            await asyncio.sleep(0)
        await asyncio.wait_for(server.stop(), timeout=2.0)
        results = await asyncio.wait_for(
            asyncio.gather(queued, blocked, return_exceptions=True), timeout=2.0
        )
        assert all(isinstance(result, InferenceServerClosed) for result in results)

    asyncio.run(backpressure_scenario())
    asyncio.run(stopped_before_start_scenario())


def test_inference_limits_and_repository_config_are_explicit() -> None:
    config = load_inference_server_config(
        REPOSITORY_ROOT / "configs" / "train" / "inference_server.yaml"
    )
    assert config.max_inference_states == 128
    assert config.max_inference_actions == 8_192
    batch = _single_state_batch(7112)
    server = InferenceServer(InferenceServerConfig(max_inference_actions=batch.action_count - 1))
    server.register_model(0, _FakeModel(0.0))

    async def rejected() -> None:
        with pytest.raises(ValueError, match="max_inference_actions"):
            await server.submit(batch, 0)
        await server.stop()

    asyncio.run(rejected())


def test_inference_server_sustains_many_requests_without_retaining_them() -> None:
    async def scenario() -> None:
        batch = _single_state_batch(7113)
        server = InferenceServer(
            InferenceServerConfig(
                max_inference_states=8,
                max_inference_actions=batch.action_count * 8,
                microbatch_wait_ms=2.0,
                max_queue_requests=4,
            )
        )
        server.register_model(2, _FakeModel(0.0))
        await server.start()
        requests = [asyncio.create_task(server.submit(batch, 2)) for _ in range(128)]
        results = await asyncio.wait_for(asyncio.gather(*requests), timeout=5.0)
        assert len(results) == 128
        stats = server.stats()
        assert stats.completed_requests == 128
        assert stats.queue_depth == 0
        assert stats.maximum_queue_depth <= stats.queue_capacity
        assert stats.maximum_batch_states <= server.config.max_inference_states
        await server.stop()

    asyncio.run(scenario())


def _transition(seed: int, *, done: bool = True) -> Transition:
    return Transition(
        serialized_state=f"state-{seed}".encode(),
        observer=seed % 3,
        chosen_action=b"action",
        behavior_logprob=-0.5,
        policy_version=seed,
        reward=1.0,
        done=done,
        raw_score=2,
    )


def _trajectory(seed: int) -> Trajectory:
    return Trajectory(
        transitions=(_transition(seed),),
        meta=EpisodeMeta(seed, "rules", (seed, seed, seed), "landlord", (2, -1, -1)),
    )


def test_versioned_replay_is_bounded_reproducible_and_reconstructable() -> None:
    replay = TrajectoryReplay(2)
    replay.append(_trajectory(1))
    replay.append(_trajectory(2))
    replay.append(_trajectory(3))
    assert replay.stats().episode_count == 2
    assert replay.stats().transition_count == 2
    assert replay.stats().evicted_episodes == 1
    assert replay.sample(2, seed=99) == replay.sample(2, seed=99)

    batch = _single_state_batch(7120)
    restored = reconstruct_states(_trajectory(3), lambda transition: batch)
    assert restored[0] is batch
    with pytest.raises(ValueError, match="terminal"):
        Trajectory((_transition(4, done=False),), _trajectory(4).meta)
    mixed = (_transition(3), replace(_transition(4), observer=1, done=True))
    mixed = (replace(mixed[0], done=False), mixed[1])
    with pytest.raises(ValueError, match="reward perspectives"):
        Trajectory(mixed, _trajectory(4).meta)


def test_unified_learner_step_connects_versions_vtrace_and_hybrid_loss() -> None:
    batch = _single_state_batch(7121)
    chosen_batch = replace(
        batch,
        chosen_action_flat_index=torch.tensor([0], dtype=torch.int64),
    )
    logits = torch.linspace(-0.5, 0.5, batch.action_count, requires_grad=True)
    probability = segment_softmax(logits, batch.action_offsets)
    output = _FakeOutput(
        policy_logit=logits,
        policy_log_probability=probability.log(),
        policy_probability=probability,
        mc_q=logits + 1.0,
        win_logit=logits - 0.25,
        expected_score=logits * 2.0,
    )
    trajectory = LearnerTrajectoryBatch(
        behavior_log_probability=output.policy_log_probability[:1].detach().reshape(1, 1),
        actor_policy_version=torch.tensor([[5]], dtype=torch.int64),
        observer_seat=torch.tensor([[0]], dtype=torch.int64),
        raw_reward=torch.tensor([[16.0]]),
        training_reward=torch.tensor([[1.0]]),
        done=torch.tensor([[True]]),
        terminal_target=torch.tensor([[1.0]]),
        win_target=torch.tensor([[1.0]]),
        score_target=torch.tensor([[4.0]]),
        bootstrap_value=torch.tensor([0.0]),
    )
    lag = PolicyLagMonitor()
    result = bird_dou_learner_step(
        output,
        chosen_batch,
        trajectory,
        HybridLossConfig(),
        VTraceConfig(),
        learner_policy_version=7,
        lag_monitor=lag,
    )
    assert result.state_value_prediction.shape == (1, 1)
    assert result.entropy.item() > 0.0
    assert torch.isfinite(result.losses.total)
    assert lag.stats().mean_lag == 2.0
    torch.autograd.backward((result.losses.total,))
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_three_mode_comparison_rejects_any_unfair_budget_or_missing_work() -> None:
    config = load_fair_comparison_config(
        REPOSITORY_ROOT / "configs" / "train" / "algorithm_comparison.yaml"
    )
    assert config.modes == tuple(TrainerMode)
    budget = FairTrainingBudget(
        environment_frames=100,
        learner_updates=10,
        seeds=(1, 2),
        model_config="model.yaml",
        rules_config="rules.yaml",
        actor_processes=1,
        envs_per_actor=2,
        unroll_length=4,
        max_inference_states=8,
        max_inference_actions=128,
        device="cpu",
    )
    runs = tuple(
        ComparisonRun(mode, budget, 100, 10, (("win_rate", 0.5), ("score", 0.0)))
        for mode in reversed(tuple(TrainerMode))
    )
    report = validate_fair_comparison(runs)
    assert report.modes == tuple(TrainerMode)
    assert report.environment_frames_per_mode == 100
    assert report.metric_names == ("win_rate", "score")
    unfair = (replace(runs[0], budget=replace(budget, environment_frames=101)), *runs[1:])
    with pytest.raises(ValueError, match="same budget"):
        validate_fair_comparison(unfair)
    incomplete = (replace(runs[0], completed_environment_frames=99), *runs[1:])
    with pytest.raises(ValueError, match="completed"):
        validate_fair_comparison(incomplete)
