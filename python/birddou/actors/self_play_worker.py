"""Vectorized spawned self-play worker backed by the central inference bridge."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import cast

import numpy as np

from birddou import PyDdzEnv
from birddou.actors.process_inference import ActorInferenceClient, ProcessInferenceChannels
from birddou.actors.process_supervisor import ActorWorkerContext
from birddou.env_types import Action, RuleConfig, StepResult
from birddou.eval.paired_deals import splitmix64
from birddou.features.ragged import FeatureConfig, encode_ragged_batch
from birddou.rl.replay import EpisodeMeta, Trajectory, Transition

SELF_PLAY_WORKER_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SelfPlayWorkerConfig:
    """Finite actor chunk and reproducible sampling controls."""

    schema_version: int = SELF_PLAY_WORKER_SCHEMA_VERSION
    episodes_per_actor: int = 32
    master_seed: int = 0
    policy_version: int = 0
    maximum_actions: int = 1_000
    maximum_redeals: int = 32

    def __post_init__(self) -> None:
        if self.schema_version != SELF_PLAY_WORKER_SCHEMA_VERSION:
            raise ValueError("unsupported self-play worker schema")
        if self.episodes_per_actor <= 0:
            raise ValueError("episodes_per_actor must be positive")
        if self.master_seed < 0 or self.policy_version < 0:
            raise ValueError("self-play seed and policy version must be non-negative")
        if self.maximum_actions <= 0 or self.maximum_redeals < 0:
            raise ValueError("self-play action/redeal limits are invalid")


@dataclass(frozen=True, slots=True)
class SelfPlayWorkerPayload:
    """Pickle-safe rules, feature schema, and inference handles."""

    rules: RuleConfig
    features: FeatureConfig
    channels: ProcessInferenceChannels
    config: SelfPlayWorkerConfig


@dataclass(frozen=True, slots=True)
class ActorTrajectory:
    """One seat trajectory with actor generation and stable episode identity."""

    actor_id: int
    actor_generation: int
    episode_index: int
    perspective_seat: int
    inference_requests: int
    trajectory: Trajectory

    @property
    def identity(self) -> tuple[int, int, int]:
        """Identity remains stable across actor restarts and seat partitioning."""
        return self.actor_id, self.episode_index, self.perspective_seat


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    serialized_state: bytes
    observer: int
    chosen_action: bytes
    behavior_logprob: float
    policy_version: int


@dataclass(slots=True)
class _EnvironmentSlot:
    environment: PyDdzEnv
    episode_index: int
    seed: int
    active_seed: int
    rng: np.random.Generator
    pending: list[_PendingDecision]
    action_count: int = 0
    redeal_count: int = 0
    inference_requests: int = 0


def run_self_play_actor(
    context: ActorWorkerContext[ActorTrajectory],
    payload: SelfPlayWorkerPayload,
) -> None:
    """Drive several native environments per process and emit complete trajectories."""
    client = ActorInferenceClient(context.actor_id, payload.channels)
    rules_hash = hashlib.sha256(
        json.dumps(payload.rules, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    next_episode = 0
    slots: list[_EnvironmentSlot] = []
    while next_episode < min(context.envs_per_actor, payload.config.episodes_per_actor):
        slots.append(_new_slot(context.actor_id, next_episode, payload))
        next_episode += 1

    while slots and not context.stop.is_set():
        observations = tuple(
            slot.environment.observe(slot.environment.current_player) for slot in slots
        )
        legal_actions = tuple(tuple(slot.environment.legal_actions()) for slot in slots)
        batch = encode_ragged_batch(
            observations,
            legal_actions,
            payload.rules,
            config=payload.features,
        )
        inference = client.infer(batch, payload.config.policy_version)
        if inference.policy_version != payload.config.policy_version:
            raise RuntimeError("central inference returned a different policy version")
        offsets = tuple(int(value) for value in inference.action_offsets.tolist())
        completed_indices: list[int] = []
        for slot_index, slot in enumerate(slots):
            start, end = offsets[slot_index], offsets[slot_index + 1]
            probability = np.asarray(
                inference.policy_probability[start:end].numpy(),
                dtype=np.float64,
            )
            if (
                len(probability) != len(legal_actions[slot_index])
                or not np.isfinite(probability).all()
                or np.any(probability < 0.0)
                or not np.isclose(probability.sum(), 1.0, atol=1e-6)
            ):
                raise RuntimeError("central inference returned invalid segment probabilities")
            selected = int(slot.rng.choice(len(probability), p=probability / probability.sum()))
            chosen = legal_actions[slot_index][selected]
            slot.pending.append(
                _PendingDecision(
                    serialized_state=slot.environment.serialize(),
                    observer=slot.environment.current_player,
                    chosen_action=_serialize_action(chosen),
                    behavior_logprob=math.log(max(float(probability[selected]), 1e-38)),
                    policy_version=inference.policy_version,
                )
            )
            slot.inference_requests += 1
            slot.action_count += 1
            if slot.action_count > payload.config.maximum_actions:
                raise RuntimeError("self-play environment exceeded maximum_actions")
            result = slot.environment.step(chosen)
            if not result["terminal"]:
                continue
            terminal_observation = slot.environment.observe(slot.environment.current_player)
            if terminal_observation["landlord"] is None:
                if not payload.rules["bidding"]["redeal_on_all_pass"]:
                    raise RuntimeError("self-play reached all-pass terminal with redeal disabled")
                if slot.redeal_count >= payload.config.maximum_redeals:
                    raise RuntimeError("self-play environment exceeded maximum_redeals")
                slot.redeal_count += 1
                slot.active_seed = splitmix64((slot.seed + slot.redeal_count) & ((1 << 64) - 1))
                slot.environment.reset(slot.active_seed, payload.rules)
                continue
            trajectories = _finish_trajectories(
                slot, result, rules_hash, payload.config.policy_version
            )
            for trajectory in trajectories:
                context.trajectories.put(
                    ActorTrajectory(
                        actor_id=context.actor_id,
                        actor_generation=context.generation,
                        episode_index=slot.episode_index,
                        perspective_seat=trajectory.perspective_seat,
                        inference_requests=len(trajectory.transitions),
                        trajectory=trajectory,
                    ),
                    timeout=payload.channels.response_timeout_s,
                )
            completed_indices.append(slot_index)

        for slot_index in reversed(completed_indices):
            if next_episode < payload.config.episodes_per_actor:
                slots[slot_index] = _new_slot(context.actor_id, next_episode, payload)
                next_episode += 1
            else:
                del slots[slot_index]


def _new_slot(
    actor_id: int,
    episode_index: int,
    payload: SelfPlayWorkerPayload,
) -> _EnvironmentSlot:
    seed = splitmix64(
        (payload.config.master_seed + actor_id * 0x9E3779B97F4A7C15 + episode_index)
        & ((1 << 64) - 1)
    )
    environment = PyDdzEnv()
    environment.reset(seed, payload.rules)
    return _EnvironmentSlot(
        environment=environment,
        episode_index=episode_index,
        seed=seed,
        active_seed=seed,
        rng=np.random.default_rng(splitmix64(seed ^ payload.config.policy_version)),
        pending=[],
    )


def _finish_trajectories(
    slot: _EnvironmentSlot,
    result: StepResult,
    rules_hash: str,
    policy_version: int,
) -> tuple[Trajectory, ...]:
    """Split a game by acting seat so V-trace never crosses opposing rewards."""
    raw = _payoff(result["raw_payoff"], "raw_payoff")
    objective = _payoff(result["objective_payoff"], "objective_payoff")
    winner_seat = result["event"]["actor"]
    landlord = slot.environment.observe(winner_seat)["landlord"]
    if landlord is None:
        raise RuntimeError("terminal self-play trajectory has no landlord")
    meta = EpisodeMeta(
        seed=slot.seed,
        rules_hash=rules_hash,
        model_versions=(policy_version, policy_version, policy_version),
        winner="landlord" if winner_seat == landlord else "farmer",
        raw_payoff=raw,
    )
    trajectories: list[Trajectory] = []
    for seat in range(3):
        decisions = tuple(item for item in slot.pending if item.observer == seat)
        if not decisions:
            continue
        transitions = tuple(
            Transition(
                serialized_state=item.serialized_state,
                observer=seat,
                chosen_action=item.chosen_action,
                behavior_logprob=item.behavior_logprob,
                policy_version=item.policy_version,
                reward=float(objective[seat]) if index == len(decisions) - 1 else 0.0,
                done=index == len(decisions) - 1,
                raw_score=raw[seat] if index == len(decisions) - 1 else 0,
            )
            for index, item in enumerate(decisions)
        )
        trajectories.append(Trajectory(transitions=transitions, meta=meta))
    return tuple(trajectories)


def _serialize_action(action: Action) -> bytes:
    return json.dumps(action, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _payoff(values: list[int], label: str) -> tuple[int, int, int]:
    if len(values) != 3 or any(isinstance(value, bool) for value in values):
        raise RuntimeError(f"terminal {label} must contain three integers")
    return cast(tuple[int, int, int], tuple(values))


__all__ = (
    "SELF_PLAY_WORKER_SCHEMA_VERSION",
    "ActorTrajectory",
    "SelfPlayWorkerConfig",
    "SelfPlayWorkerPayload",
    "run_self_play_actor",
)
