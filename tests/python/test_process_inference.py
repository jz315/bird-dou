"""End-to-end spawned Actor to central inference-server IPC test."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from birddou import PyDdzEnv, load_rule_config
from birddou.actors import (
    ActorInferenceClient,
    ActorProcessSupervisor,
    ActorSupervisorConfig,
    ActorTrajectory,
    ActorWorkerContext,
    InferenceServer,
    InferenceServerConfig,
    MultiprocessInferenceBridge,
    ProcessInferenceChannels,
    ProcessInferenceConfig,
    SelfPlayWorkerConfig,
    SelfPlayWorkerPayload,
    run_self_play_actor,
)
from birddou.features import FeatureConfig, RaggedBatch, encode_ragged_batch
from birddou.models.segment_ops import segment_softmax

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BridgeItem = tuple[int, int, int, float]
BridgePayload = tuple[ProcessInferenceChannels, str]


@dataclass(frozen=True, slots=True)
class _FakeOutput:
    policy_logit: Tensor
    policy_log_probability: Tensor
    policy_probability: Tensor
    mc_q: Tensor
    win_logit: Tensor
    expected_score: Tensor


class _FakeModel:
    def eval(self) -> _FakeModel:
        return self

    def __call__(self, batch: RaggedBatch) -> _FakeOutput:
        logits = torch.arange(batch.action_count, dtype=torch.float32) * 0.01
        probability = segment_softmax(logits, batch.action_offsets)
        return _FakeOutput(
            policy_logit=logits,
            policy_log_probability=probability.log(),
            policy_probability=probability,
            mc_q=logits,
            win_logit=logits,
            expected_score=logits,
        )


def _remote_inference_actor(
    context: ActorWorkerContext[BridgeItem],
    payload: BridgePayload,
) -> None:
    channels, rules_path = payload
    rules = load_rule_config(Path(rules_path))
    environment = PyDdzEnv()
    environment.reset(9_100 + context.actor_id, rules)
    batch = encode_ragged_batch(
        (environment.observe(environment.current_player),),
        (environment.legal_actions(),),
        rules,
        config=FeatureConfig(decomposition_features=False),
    )
    result = ActorInferenceClient(context.actor_id, channels).infer(batch, policy_version=7)
    context.trajectories.put(
        (
            context.actor_id,
            result.policy_version,
            result.policy_probability.numel(),
            float(result.policy_probability.sum().item()),
        ),
        timeout=5.0,
    )


def test_spawned_actors_share_one_versioned_central_inference_server() -> None:
    actor_count = 2
    server = InferenceServer(
        InferenceServerConfig(
            max_inference_states=actor_count,
            max_inference_actions=8_192,
            microbatch_wait_ms=50.0,
            max_queue_requests=actor_count,
        )
    )
    server.register_model(7, _FakeModel())
    bridge = MultiprocessInferenceBridge(
        server,
        ProcessInferenceConfig(
            actor_processes=actor_count,
            request_queue_capacity=actor_count,
            response_queue_capacity=2,
            actor_response_timeout_s=10.0,
        ),
    )
    supervisor = ActorProcessSupervisor(
        ActorSupervisorConfig(
            actor_processes=actor_count,
            envs_per_actor=3,
            trajectory_queue_capacity=actor_count,
            max_restarts_per_actor=0,
            process_join_timeout_s=3.0,
            queue_poll_timeout_s=0.02,
        ),
        _remote_inference_actor,
        (bridge.channels, str(REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml")),
    )
    try:
        bridge.start()
        supervisor.start()
        results = tuple(supervisor.get_trajectory(timeout=15.0) for _ in range(actor_count))
        assert {item[0] for item in results} == {0, 1}
        assert all(item[1] == 7 for item in results)
        assert all(item[2] > 0 for item in results)
        assert all(abs(item[3] - 1.0) < 1e-6 for item in results)
        bridge_stats = bridge.stats()
        assert bridge_stats.received_requests == actor_count
        assert bridge_stats.completed_requests == actor_count
        assert bridge_stats.failed_requests == 0
        assert bridge_stats.dropped_replies == 0
        assert server.stats().maximum_batch_states <= actor_count
    finally:
        supervisor.shutdown()
        bridge.stop()


def test_vectorized_spawned_actor_completes_native_games_and_trajectories() -> None:
    rules = load_rule_config(REPOSITORY_ROOT / "configs" / "rules" / "douzero_post_bid.yaml")
    server = InferenceServer(
        InferenceServerConfig(
            max_inference_states=2,
            max_inference_actions=8_192,
            microbatch_wait_ms=1.0,
            max_queue_requests=2,
        )
    )
    server.register_model(11, _FakeModel())
    bridge = MultiprocessInferenceBridge(
        server,
        ProcessInferenceConfig(
            actor_processes=1,
            request_queue_capacity=2,
            response_queue_capacity=2,
            actor_response_timeout_s=20.0,
        ),
    )
    payload = SelfPlayWorkerPayload(
        rules=rules,
        features=FeatureConfig(decomposition_features=False),
        channels=bridge.channels,
        config=SelfPlayWorkerConfig(
            episodes_per_actor=2,
            master_seed=442,
            policy_version=11,
        ),
    )
    supervisor: ActorProcessSupervisor[SelfPlayWorkerPayload, ActorTrajectory] = (
        ActorProcessSupervisor(
            ActorSupervisorConfig(
                actor_processes=1,
                envs_per_actor=2,
                trajectory_queue_capacity=2,
                max_restarts_per_actor=0,
                process_join_timeout_s=3.0,
                queue_poll_timeout_s=0.02,
            ),
            run_self_play_actor,
            payload,
        )
    )
    try:
        bridge.start()
        supervisor.start()
        outputs = tuple(supervisor.get_trajectory(timeout=30.0) for _ in range(2))
        assert {output.identity for output in outputs} == {(0, 0), (0, 1)}
        for output in outputs:
            trajectory = output.trajectory
            assert output.inference_requests == len(trajectory.transitions)
            assert trajectory.transitions[-1].done
            assert not any(item.done for item in trajectory.transitions[:-1])
            assert all(item.policy_version == 11 for item in trajectory.transitions)
            assert trajectory.meta.model_versions == (11, 11, 11)
            assert sum(trajectory.meta.raw_payoff) == 0
            assert sum(item.reward != 0.0 for item in trajectory.transitions) <= 1
        assert bridge.stats().completed_requests > 0
        assert server.stats().maximum_batch_states == 2
    finally:
        supervisor.shutdown()
        bridge.stop()
