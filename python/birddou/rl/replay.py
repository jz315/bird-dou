"""Versioned compact trajectories and bounded replay storage."""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from birddou.features.ragged import RaggedBatch


@dataclass(frozen=True, slots=True)
class Transition:
    """One actor decision with enough state to reconstruct legal features."""

    serialized_state: bytes
    observer: int
    chosen_action: bytes
    behavior_logprob: float
    policy_version: int
    reward: float
    done: bool
    raw_score: int

    def __post_init__(self) -> None:
        if not self.serialized_state:
            raise ValueError("transition serialized_state cannot be empty")
        if not 0 <= self.observer <= 2:
            raise ValueError("transition observer must be a seat in 0..2")
        if not self.chosen_action:
            raise ValueError("transition chosen_action cannot be empty")
        if not math.isfinite(self.behavior_logprob) or self.behavior_logprob > 0.0:
            raise ValueError("behavior_logprob must be finite and no greater than zero")
        if self.policy_version < 0:
            raise ValueError("transition policy_version must be non-negative")
        if not math.isfinite(self.reward):
            raise ValueError("transition reward must be finite")
        if not isinstance(self.done, bool):
            raise ValueError("transition done must be bool")
        if not isinstance(self.raw_score, int) or isinstance(self.raw_score, bool):
            raise ValueError("transition raw_score must be an integer")


@dataclass(frozen=True, slots=True)
class EpisodeMeta:
    """Auditable deal, rules, policy, winner, and platform-payoff metadata."""

    seed: int
    rules_hash: str
    model_versions: tuple[int, int, int]
    winner: str
    raw_payoff: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("episode seed must be non-negative")
        if not self.rules_hash:
            raise ValueError("episode rules_hash cannot be empty")
        if len(self.model_versions) != 3 or any(version < 0 for version in self.model_versions):
            raise ValueError("episode model versions must be non-negative")
        if not self.winner:
            raise ValueError("episode winner cannot be empty")
        if len(self.raw_payoff) != 3 or any(
            not isinstance(value, int) or isinstance(value, bool) for value in self.raw_payoff
        ):
            raise ValueError("episode raw payoff must contain integers")
        if sum(self.raw_payoff) != 0:
            raise ValueError("episode raw payoff must be zero-sum")


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One role-homogeneous decision trajectory from a complete game."""

    transitions: tuple[Transition, ...]
    meta: EpisodeMeta

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ValueError("trajectory must contain at least one transition")
        if not self.transitions[-1].done:
            raise ValueError("trajectory final transition must be terminal")
        if any(transition.done for transition in self.transitions[:-1]):
            raise ValueError("trajectory cannot contain transitions after terminal")
        perspective = self.transitions[0].observer
        if any(transition.observer != perspective for transition in self.transitions[1:]):
            raise ValueError("V-trace trajectory cannot mix observer reward perspectives")

    @property
    def perspective_seat(self) -> int:
        """Return the one seat whose value/reward semantics define this trajectory."""
        return self.transitions[0].observer


@dataclass(frozen=True, slots=True)
class ReplayStats:
    """Constant-size bounded replay diagnostics."""

    episode_count: int
    transition_count: int
    episode_capacity: int
    evicted_episodes: int
    added_episodes: int


class TrajectoryReplay:
    """A reproducibly sampled replay bounded by complete episode count."""

    def __init__(self, episode_capacity: int) -> None:
        if episode_capacity <= 0:
            raise ValueError("replay episode capacity must be positive")
        self._capacity = episode_capacity
        self._episodes: deque[Trajectory] = deque()
        self._transition_count = 0
        self._evicted = 0
        self._added = 0

    def append(self, trajectory: Trajectory) -> None:
        """Append atomically, evicting the oldest complete episode if full."""
        if len(self._episodes) == self._capacity:
            removed = self._episodes.popleft()
            self._transition_count -= len(removed.transitions)
            self._evicted += 1
        self._episodes.append(trajectory)
        self._transition_count += len(trajectory.transitions)
        self._added += 1

    def sample(self, count: int, *, seed: int) -> tuple[Trajectory, ...]:
        """Sample episodes without replacement using only the supplied seed."""
        if count <= 0:
            raise ValueError("replay sample count must be positive")
        if count > len(self._episodes):
            raise ValueError("replay sample exceeds stored episode count")
        if seed < 0:
            raise ValueError("replay sample seed must be non-negative")
        indices = random.Random(seed).sample(range(len(self._episodes)), count)
        values = tuple(self._episodes)
        return tuple(values[index] for index in indices)

    def stats(self) -> ReplayStats:
        """Return storage metrics without exposing or copying payloads."""
        return ReplayStats(
            episode_count=len(self._episodes),
            transition_count=self._transition_count,
            episode_capacity=self._capacity,
            evicted_episodes=self._evicted,
            added_episodes=self._added,
        )

    def __len__(self) -> int:
        return len(self._episodes)


def reconstruct_states(
    trajectory: Trajectory,
    decoder: Callable[[Transition], RaggedBatch],
) -> tuple[RaggedBatch, ...]:
    """Rebuild one-state public observations/legal actions in stable order."""
    batches: list[RaggedBatch] = []
    for transition in trajectory.transitions:
        batch = decoder(transition)
        if batch.batch_size != 1:
            raise ValueError("trajectory decoder must return one state at a time")
        batches.append(batch)
    return tuple(batches)


__all__ = (
    "EpisodeMeta",
    "ReplayStats",
    "Trajectory",
    "TrajectoryReplay",
    "Transition",
    "reconstruct_states",
)
