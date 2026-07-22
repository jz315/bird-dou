"""Deterministic single-process actor used by the E015 DMC smoke loop."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from birddou import PyDdzEnv
from birddou.env_types import RuleConfig
from birddou.eval.paired_deals import SeatRole, role_for_seat
from birddou.features import encode_douzero_features


class ActorModel(Protocol):
    """Value-network surface required by the local actor."""

    def eval(self) -> ActorModel: ...

    def __call__(
        self,
        z: Tensor,
        x: Tensor,
        *,
        return_value: bool,
    ) -> Mapping[str, Tensor]: ...


@dataclass(frozen=True, slots=True)
class DmcTransition:
    """One chosen candidate with its terminal Monte Carlo target."""

    serialized_state: bytes
    seat: int
    role: SeatRole
    x: NDArray[np.float32]
    z: NDArray[np.float32]
    chosen_action_index: int
    behavior_logprob: float
    policy_version: int
    target: float


@dataclass(frozen=True, slots=True)
class DmcEpisode:
    """A complete self-play game and all role-labeled decisions."""

    seed: int
    transitions: tuple[DmcTransition, ...]
    action_count: int
    winner_seat: int
    raw_payoff: tuple[int, int, int]
    objective_payoff: tuple[int, int, int]

    def transitions_for(self, role: SeatRole) -> tuple[DmcTransition, ...]:
        """Return the stable subsequence belonging to one role network."""
        return tuple(item for item in self.transitions if item.role is role)


@dataclass(frozen=True, slots=True)
class _PendingTransition:
    serialized_state: bytes
    seat: int
    role: SeatRole
    x: NDArray[np.float32]
    z: NDArray[np.float32]
    chosen_action_index: int
    behavior_logprob: float
    policy_version: int


def collect_dmc_episode(
    seed: int,
    rules: RuleConfig,
    models: Mapping[SeatRole, ActorModel],
    rng: np.random.Generator,
    *,
    epsilon: float,
    policy_version: int,
    device: str = "cpu",
    max_actions: int = 1_000,
) -> DmcEpisode:
    """Play a full epsilon-greedy game and attach role-specific terminal returns."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in 0..1")
    if set(models) != set(SeatRole):
        raise ValueError("actor requires landlord, landlord_down, and landlord_up models")
    environment = PyDdzEnv()
    environment.reset(seed, rules)
    pending: list[_PendingTransition] = []
    terminal_result: Mapping[str, object] | None = None

    while not environment.terminal:
        if len(pending) >= max_actions:
            raise RuntimeError(f"DMC actor exceeded {max_actions} actions for seed {seed}")
        seat = environment.current_player
        role = role_for_seat(seat)
        observation = environment.observe(seat)
        legal_actions = environment.legal_actions()
        features = encode_douzero_features(observation, legal_actions)
        action_count = len(legal_actions)
        model = models[role].eval()
        with torch.inference_mode():
            output = model(
                torch.from_numpy(features.z_batch).to(device),
                torch.from_numpy(features.x_batch).to(device),
                return_value=True,
            )
        scores = np.asarray(output["values"].detach().cpu().numpy(), dtype=np.float32)[:, 0]
        if scores.shape != (action_count,) or not np.isfinite(scores).all():
            raise RuntimeError(f"actor model returned invalid {role.value} scores")
        greedy = int(np.argmax(scores))
        explore = action_count > 1 and float(rng.random()) < epsilon
        selected = int(rng.integers(action_count)) if explore else greedy
        random_probability = epsilon / action_count
        probability = random_probability + (1.0 - epsilon if selected == greedy else 0.0)
        pending.append(
            _PendingTransition(
                serialized_state=environment.serialize(),
                seat=seat,
                role=role,
                x=features.x_batch[selected].copy(),
                z=features.z_batch[selected].copy(),
                chosen_action_index=selected,
                behavior_logprob=math.log(probability),
                policy_version=policy_version,
            )
        )
        terminal_result = cast(Mapping[str, object], environment.step(legal_actions[selected]))

    if terminal_result is None:
        raise RuntimeError("DMC actor produced no terminal result")
    raw = _payoff_tuple(terminal_result["raw_payoff"], "raw_payoff")
    objective = _payoff_tuple(terminal_result["objective_payoff"], "objective_payoff")
    event = terminal_result["event"]
    if not isinstance(event, Mapping) or not isinstance(event.get("actor"), int):
        raise RuntimeError("terminal event has no winner actor")
    transitions = tuple(
        DmcTransition(
            serialized_state=item.serialized_state,
            seat=item.seat,
            role=item.role,
            x=item.x,
            z=item.z,
            chosen_action_index=item.chosen_action_index,
            behavior_logprob=item.behavior_logprob,
            policy_version=item.policy_version,
            target=float(objective[item.seat]),
        )
        for item in pending
    )
    return DmcEpisode(
        seed=seed,
        transitions=transitions,
        action_count=len(transitions),
        winner_seat=cast(int, event["actor"]),
        raw_payoff=raw,
        objective_payoff=objective,
    )


def _payoff_tuple(value: object, label: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise RuntimeError(f"terminal {label} must contain three integer values")
    return cast(tuple[int, int, int], tuple(value))


__all__ = (
    "ActorModel",
    "DmcEpisode",
    "DmcTransition",
    "collect_dmc_episode",
)
