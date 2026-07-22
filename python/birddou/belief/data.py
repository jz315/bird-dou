"""Leakage-isolated generation and storage of supervised hidden-hand labels."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from birddou import PyDdzEnv
from birddou.env_types import Observation, RuleConfig
from birddou.eval.baselines import Policy, PolicyDecisionContext
from birddou.eval.paired_deals import role_for_seat, splitmix64
from birddou.features import FEATURE_SCHEMA_VERSION, FeatureConfig, RaggedBatch, encode_ragged_batch
from birddou.models.belief_bird_dou import belief_constraints_from_batch
from birddou.models.segment_ops import segment_state_index

BELIEF_DATASET_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BeliefDataset:
    """Public RaggedBatch states paired with training-only next-player labels."""

    schema_version: int
    batch: RaggedBatch
    true_assignment_a: Tensor
    policy_index: Tensor
    policy_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != BELIEF_DATASET_SCHEMA_VERSION:
            raise ValueError("unsupported Belief dataset schema")
        if self.true_assignment_a.dtype != torch.int64 or self.true_assignment_a.shape != (
            self.batch.batch_size,
            15,
        ):
            raise ValueError("Belief dataset labels must be int64 [N, 15]")
        if self.policy_index.dtype != torch.int64 or self.policy_index.shape != (
            self.batch.batch_size,
        ):
            raise ValueError("Belief dataset policy indices must be int64 [N]")
        if not self.policy_ids or len(set(self.policy_ids)) != len(self.policy_ids):
            raise ValueError("Belief dataset policy IDs must be non-empty and unique")
        if any(not value for value in self.policy_ids):
            raise ValueError("Belief dataset policy IDs cannot be empty")
        if torch.any((self.policy_index < 0) | (self.policy_index >= len(self.policy_ids))):
            raise ValueError("Belief dataset policy index is out of range")
        unknown, capacity_a, _ = belief_constraints_from_batch(self.batch)
        if torch.any(
            (self.true_assignment_a < 0) | (self.true_assignment_a > unknown)
        ) or not torch.equal(self.true_assignment_a.sum(dim=1), capacity_a):
            raise ValueError("Belief dataset label violates public conservation constraints")

    @property
    def state_count(self) -> int:
        return self.batch.batch_size

    def select(self, indices: Tensor) -> BeliefDataset:
        """Select state rows and rebase every associated ragged action segment."""
        if indices.dtype != torch.int64 or indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("Belief dataset indices must be non-empty int64 [K]")
        if indices.device.type != "cpu":
            raise ValueError("Belief dataset selection indices must be on CPU")
        if torch.any((indices < 0) | (indices >= self.state_count)):
            raise ValueError("Belief dataset selection index is out of range")
        action_rows: list[Tensor] = []
        lengths: list[int] = []
        chosen_rows: list[int] = []
        action_base = 0
        for state in indices.tolist():
            start = int(self.batch.action_offsets[state].item())
            end = int(self.batch.action_offsets[state + 1].item())
            action_rows.append(torch.arange(start, end, dtype=torch.int64))
            lengths.append(end - start)
            original_chosen = int(self.batch.chosen_action_flat_index[state].item())
            chosen_rows.append(-1 if original_chosen < 0 else action_base + original_chosen - start)
            action_base += end - start
        flat_actions = torch.cat(action_rows)
        offsets = torch.tensor([0, *np.cumsum(lengths).tolist()], dtype=torch.int64)
        selected_batch = RaggedBatch(
            schema_version=self.batch.schema_version,
            rank_categorical=self.batch.rank_categorical[indices],
            rank_numeric=self.batch.rank_numeric[indices],
            history_rank_counts=self.batch.history_rank_counts[indices],
            history_meta=self.batch.history_meta[indices],
            history_mask=self.batch.history_mask[indices],
            scalars=self.batch.scalars[indices],
            action_rank_counts=self.batch.action_rank_counts[flat_actions],
            post_hand_counts=self.batch.post_hand_counts[flat_actions],
            action_meta=self.batch.action_meta[flat_actions],
            action_state_index=segment_state_index(offsets),
            action_offsets=offsets,
            chosen_action_flat_index=torch.tensor(chosen_rows, dtype=torch.int64),
        )
        return BeliefDataset(
            self.schema_version,
            selected_batch,
            self.true_assignment_a[indices],
            self.policy_index[indices],
            self.policy_ids,
        )


@dataclass(frozen=True, slots=True)
class BeliefDatasetArtifact:
    dataset_path: Path
    manifest_path: Path
    sha256: str
    state_count: int
    game_count: int
    policy_counts: Mapping[str, int]


def generate_belief_dataset(
    game_count: int,
    master_seed: int,
    rules: RuleConfig,
    policies: Sequence[Policy],
    feature_config: FeatureConfig,
    *,
    max_actions: int = 1_000,
) -> BeliefDataset:
    """Collect mixed-policy public states and labels from privileged serialization."""
    if game_count <= 0:
        raise ValueError("Belief dataset game_count must be positive")
    if not 0 <= master_seed < 1 << 64:
        raise ValueError("Belief dataset master_seed must fit uint64")
    if not policies or len({policy.policy_id for policy in policies}) != len(policies):
        raise ValueError("Belief dataset policies must be non-empty with unique IDs")
    batches: list[RaggedBatch] = []
    labels: list[list[int]] = []
    policy_indices: list[int] = []
    for game_index in range(game_count):
        game_seed = splitmix64((master_seed + game_index) & ((1 << 64) - 1))
        environment = PyDdzEnv()
        environment.reset(game_seed, rules)
        decision_index = [0, 0, 0]
        action_count = 0
        while not environment.terminal:
            if action_count >= max_actions:
                raise RuntimeError(f"Belief dataset game {game_index} exceeded max_actions")
            seat = environment.current_player
            observation = environment.observe(seat)
            legal_actions = tuple(environment.legal_actions())
            policy_index = splitmix64(game_seed ^ action_count) % len(policies)
            policy = policies[policy_index]
            context = PolicyDecisionContext(
                deal_index=game_index,
                deal_seed=game_seed,
                match_id=f"belief-{game_index}",
                seat=seat,
                role=role_for_seat(seat),
                decision_index=decision_index[seat],
            )
            selected = policy.select_action(observation, legal_actions, context)
            if isinstance(selected, bool) or not 0 <= selected < len(legal_actions):
                raise RuntimeError(f"Belief dataset policy {policy.policy_id} returned bad action")
            batch = encode_ragged_batch(
                (observation,),
                (legal_actions,),
                rules,
                chosen_action_indices=(selected,),
                config=feature_config,
            )
            label = extract_hidden_assignment(environment.serialize(), observation)
            batches.append(batch)
            labels.append(label)
            policy_indices.append(policy_index)
            environment.step(legal_actions[selected])
            decision_index[seat] += 1
            action_count += 1
    combined = concatenate_ragged_batches(batches)
    return BeliefDataset(
        schema_version=BELIEF_DATASET_SCHEMA_VERSION,
        batch=combined,
        true_assignment_a=torch.tensor(labels, dtype=torch.int64),
        policy_index=torch.tensor(policy_indices, dtype=torch.int64),
        policy_ids=tuple(policy.policy_id for policy in policies),
    )


def extract_hidden_assignment(serialized_state: bytes, observation: Observation) -> list[int]:
    """Training-only oracle: assign the next relative player's true remaining hand."""
    try:
        envelope = json.loads(serialized_state)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("serialized training state is not valid JSON") from error
    root = _object_mapping(envelope, "state envelope")
    state = _object_mapping(root.get("state"), "serialized game state")
    hands = state.get("hands")
    if not isinstance(hands, list) or len(hands) != 3:
        raise ValueError("serialized game state must contain three hands")
    decoded = [_rank_counts(value, f"hidden hand {seat}") for seat, value in enumerate(hands)]
    observer = observation["observer"]
    if state.get("current_player") != observer:
        raise ValueError("serialized state and public observation refer to different actors")
    hidden_a = decoded[(observer + 1) % 3]
    hidden_b = decoded[(observer + 2) % 3]
    combined = [left + right for left, right in zip(hidden_a, hidden_b, strict=True)]
    if combined != observation["unknown_pool"]:
        raise ValueError("oracle hands do not reconstruct the public unknown pool")
    if sum(hidden_a) != observation["cards_left"][(observer + 1) % 3]:
        raise ValueError("oracle next-player hand violates the public card count")
    return hidden_a


def concatenate_ragged_batches(batches: Sequence[RaggedBatch]) -> RaggedBatch:
    """Concatenate non-empty batches while rebasing offsets and chosen rows."""
    if not batches:
        raise ValueError("at least one RaggedBatch is required")
    history_length = batches[0].history_rank_counts.shape[1]
    if any(batch.history_rank_counts.shape[1] != history_length for batch in batches):
        raise ValueError("RaggedBatch history lengths must match")
    offsets = [0]
    chosen: list[int] = []
    action_base = 0
    for batch in batches:
        local_offsets = batch.action_offsets.tolist()
        offsets.extend(action_base + int(value) for value in local_offsets[1:])
        chosen.extend(
            -1 if int(value) < 0 else action_base + int(value)
            for value in batch.chosen_action_flat_index.tolist()
        )
        action_base += batch.action_count
    action_offsets = torch.tensor(offsets, dtype=torch.int64)
    return RaggedBatch(
        schema_version=FEATURE_SCHEMA_VERSION,
        rank_categorical=torch.cat([batch.rank_categorical for batch in batches]),
        rank_numeric=torch.cat([batch.rank_numeric for batch in batches]),
        history_rank_counts=torch.cat([batch.history_rank_counts for batch in batches]),
        history_meta=torch.cat([batch.history_meta for batch in batches]),
        history_mask=torch.cat([batch.history_mask for batch in batches]),
        scalars=torch.cat([batch.scalars for batch in batches]),
        action_rank_counts=torch.cat([batch.action_rank_counts for batch in batches]),
        post_hand_counts=torch.cat([batch.post_hand_counts for batch in batches]),
        action_meta=torch.cat([batch.action_meta for batch in batches]),
        action_state_index=segment_state_index(action_offsets),
        action_offsets=action_offsets,
        chosen_action_flat_index=torch.tensor(chosen, dtype=torch.int64),
    )


def save_belief_dataset(
    dataset: BeliefDataset,
    path: Path,
    *,
    game_count: int,
    master_seed: int,
) -> BeliefDatasetArtifact:
    """Write compressed tensors plus an auditable JSON manifest and digest."""
    if game_count <= 0:
        raise ValueError("Belief dataset manifest game_count must be positive")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    batch = dataset.batch
    np.savez_compressed(
        path,
        schema_version=np.array([dataset.schema_version], dtype=np.int64),
        rank_categorical=batch.rank_categorical.numpy(),
        rank_numeric=batch.rank_numeric.numpy(),
        history_rank_counts=batch.history_rank_counts.numpy(),
        history_meta=batch.history_meta.numpy(),
        history_mask=batch.history_mask.numpy(),
        scalars=batch.scalars.numpy(),
        action_rank_counts=batch.action_rank_counts.numpy(),
        post_hand_counts=batch.post_hand_counts.numpy(),
        action_meta=batch.action_meta.numpy(),
        action_state_index=batch.action_state_index.numpy(),
        action_offsets=batch.action_offsets.numpy(),
        chosen_action_flat_index=batch.chosen_action_flat_index.numpy(),
        true_assignment_a=dataset.true_assignment_a.numpy(),
        policy_index=dataset.policy_index.numpy(),
        policy_ids=np.asarray(dataset.policy_ids, dtype=np.str_),
    )
    digest = _sha256_file(path)
    counts = Counter(dataset.policy_ids[int(index)] for index in dataset.policy_index.tolist())
    manifest_path = path.with_suffix(".manifest.json")
    manifest = {
        "schema_version": BELIEF_DATASET_SCHEMA_VERSION,
        "dataset_file": path.name,
        "dataset_sha256": digest,
        "state_count": dataset.state_count,
        "game_count": game_count,
        "master_seed": master_seed,
        "feature_schema_version": dataset.batch.schema_version,
        "policy_ids": list(dataset.policy_ids),
        "policy_counts": dict(sorted(counts.items())),
        "label_scope": "training_only_next_relative_hidden_hand",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return BeliefDatasetArtifact(
        path, manifest_path, digest, dataset.state_count, game_count, counts
    )


def load_belief_dataset(path: Path) -> BeliefDataset:
    """Load a compressed dataset without permitting pickled/object arrays."""
    with np.load(path.resolve(), allow_pickle=False) as arrays:
        schema = int(arrays["schema_version"][0])
        batch = RaggedBatch(
            schema_version=FEATURE_SCHEMA_VERSION,
            rank_categorical=_tensor(arrays["rank_categorical"]),
            rank_numeric=_tensor(arrays["rank_numeric"]),
            history_rank_counts=_tensor(arrays["history_rank_counts"]),
            history_meta=_tensor(arrays["history_meta"]),
            history_mask=_tensor(arrays["history_mask"]),
            scalars=_tensor(arrays["scalars"]),
            action_rank_counts=_tensor(arrays["action_rank_counts"]),
            post_hand_counts=_tensor(arrays["post_hand_counts"]),
            action_meta=_tensor(arrays["action_meta"]),
            action_state_index=_tensor(arrays["action_state_index"]),
            action_offsets=_tensor(arrays["action_offsets"]),
            chosen_action_flat_index=_tensor(arrays["chosen_action_flat_index"]),
        )
        labels = _tensor(arrays["true_assignment_a"])
        policy_index = _tensor(arrays["policy_index"])
        policy_ids = tuple(str(value) for value in arrays["policy_ids"].tolist())
    return BeliefDataset(schema, batch, labels, policy_index, policy_ids)


def _tensor(value: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> Tensor:
    return torch.from_numpy(value.copy())


def _rank_counts(value: object, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 15
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"{label} must contain fifteen integer counts")
    result = cast(list[int], value)
    if any(count < 0 or count > (1 if rank >= 13 else 4) for rank, count in enumerate(result)):
        raise ValueError(f"{label} contains an impossible rank count")
    return result


def _object_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "BELIEF_DATASET_SCHEMA_VERSION",
    "BeliefDataset",
    "BeliefDatasetArtifact",
    "concatenate_ragged_batches",
    "extract_hidden_assignment",
    "generate_belief_dataset",
    "load_belief_dataset",
    "save_belief_dataset",
)
