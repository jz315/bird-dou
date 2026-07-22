"""Versioned fixed-deal generation and seat-balanced evaluation schedules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import cast

PAIRED_DEAL_SCHEMA_VERSION = 1
DEAL_GENERATOR_ALGORITHM = "splitmix64_v1"
_U64_MASK = (1 << 64) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15


class SeatRole(StrEnum):
    """DouZero-compatible role attached to each stable Rust seat."""

    LANDLORD = "landlord"
    LANDLORD_DOWN = "landlord_down"
    LANDLORD_UP = "landlord_up"


SEAT_ROLES: tuple[SeatRole, SeatRole, SeatRole] = (
    SeatRole.LANDLORD,
    SeatRole.LANDLORD_DOWN,
    SeatRole.LANDLORD_UP,
)


def role_for_seat(seat: int) -> SeatRole:
    """Return the named role for a validated seat in `0..=2`."""
    if not 0 <= seat < len(SEAT_ROLES):
        raise ValueError(f"seat {seat} is outside 0..=2")
    return SEAT_ROLES[seat]


def role_for_game_seat(seat: int, landlord: int) -> SeatRole:
    """Map an absolute seat to its role relative to the resolved landlord."""
    role_for_seat(seat)
    role_for_seat(landlord)
    return role_for_seat((seat - landlord) % 3)


def splitmix64(value: int) -> int:
    """Apply the fixed SplitMix64 output permutation to one unsigned integer."""
    if not 0 <= value <= _U64_MASK:
        raise ValueError("SplitMix64 input must fit uint64")
    mixed = value
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return (mixed ^ (mixed >> 31)) & _U64_MASK


@dataclass(frozen=True, slots=True)
class PairedDeal:
    """One immutable environment seed in a fixed evaluation set."""

    schema_version: int
    deal_index: int
    seed: int
    deal_id: str

    def __post_init__(self) -> None:
        if self.schema_version != PAIRED_DEAL_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported paired-deal schema {self.schema_version}; "
                f"expected {PAIRED_DEAL_SCHEMA_VERSION}"
            )
        if self.deal_index < 0:
            raise ValueError("deal_index must be non-negative")
        if not 0 <= self.seed <= _U64_MASK:
            raise ValueError("deal seed must fit uint64")
        if not self.deal_id:
            raise ValueError("deal_id must be non-empty")


@dataclass(frozen=True, slots=True)
class PairedDealSet:
    """Auditable fixed-seed collection generated from one master seed."""

    schema_version: int
    master_seed: int
    algorithm: str
    deals: tuple[PairedDeal, ...]

    def __post_init__(self) -> None:
        if self.schema_version != PAIRED_DEAL_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported paired-deal schema {self.schema_version}; "
                f"expected {PAIRED_DEAL_SCHEMA_VERSION}"
            )
        if not 0 <= self.master_seed <= _U64_MASK:
            raise ValueError("master_seed must fit uint64")
        if self.algorithm != DEAL_GENERATOR_ALGORITHM:
            raise ValueError(f"unsupported deal generator algorithm: {self.algorithm}")
        if not self.deals:
            raise ValueError("paired deal set must contain at least one deal")
        expected_indices = tuple(range(len(self.deals)))
        actual_indices = tuple(deal.deal_index for deal in self.deals)
        if actual_indices != expected_indices:
            raise ValueError("paired deals must use contiguous stable indices")
        if len({deal.seed for deal in self.deals}) != len(self.deals):
            raise ValueError("paired deal seeds must be unique")

    @property
    def count(self) -> int:
        """Number of independent deal clusters."""
        return len(self.deals)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable manifest."""
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class SeatAssignment:
    """Policy identifier assigned to each stable seat `0, 1, 2`."""

    policy_ids: tuple[str, str, str]

    def __post_init__(self) -> None:
        if len(self.policy_ids) != len(SEAT_ROLES):
            raise ValueError("seat assignment must contain exactly three policy IDs")
        if any(not policy_id for policy_id in self.policy_ids):
            raise ValueError("policy IDs must be non-empty")

    def policy_for_seat(self, seat: int) -> str:
        """Return the assigned policy for a validated seat."""
        role_for_seat(seat)
        return self.policy_ids[seat]


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Two symmetric cross-play matches for one fixed deal and focal role."""

    pair_id: str
    deal: PairedDeal
    focal_seat: int
    focal_role: SeatRole
    candidate_policy_id: str
    baseline_policy_id: str
    candidate_assignment: SeatAssignment
    baseline_assignment: SeatAssignment

    def __post_init__(self) -> None:
        if not self.pair_id:
            raise ValueError("pair_id must be non-empty")
        if role_for_seat(self.focal_seat) is not self.focal_role:
            raise ValueError("focal role does not match focal seat")
        if self.candidate_policy_id == self.baseline_policy_id:
            raise ValueError("candidate and baseline policy IDs must differ")
        if self.candidate_assignment.policy_for_seat(self.focal_seat) != self.candidate_policy_id:
            raise ValueError("candidate must occupy the focal seat in candidate_assignment")
        if self.baseline_assignment.policy_for_seat(self.focal_seat) != self.baseline_policy_id:
            raise ValueError("baseline must occupy the focal seat in baseline_assignment")


@dataclass(frozen=True, slots=True)
class ScheduledMatch:
    """One stable arbitrary cross-play match."""

    match_id: str
    deal: PairedDeal
    assignment: SeatAssignment

    def __post_init__(self) -> None:
        if not self.match_id:
            raise ValueError("match_id must be non-empty")


def generate_paired_deals(master_seed: int, count: int) -> PairedDealSet:
    """Generate `count` unique deterministic environment seeds."""
    if not 0 <= master_seed <= _U64_MASK:
        raise ValueError("master_seed must fit uint64")
    if count <= 0:
        raise ValueError("count must be positive")
    if count > _U64_MASK:
        raise ValueError("count exceeds the SplitMix64 period")

    state = master_seed
    deals: list[PairedDeal] = []
    for deal_index in range(count):
        state = (state + _SPLITMIX_INCREMENT) & _U64_MASK
        seed = splitmix64(state)
        deal_id = f"{master_seed:016x}-{deal_index:08x}-{seed:016x}"
        deals.append(
            PairedDeal(
                schema_version=PAIRED_DEAL_SCHEMA_VERSION,
                deal_index=deal_index,
                seed=seed,
                deal_id=deal_id,
            )
        )
    return PairedDealSet(
        schema_version=PAIRED_DEAL_SCHEMA_VERSION,
        master_seed=master_seed,
        algorithm=DEAL_GENERATOR_ALGORITHM,
        deals=tuple(deals),
    )


def generate_paired_comparisons(
    deal_set: PairedDealSet,
    candidate_policy_id: str,
    baseline_policy_id: str,
) -> tuple[PairedComparison, ...]:
    """Rotate the candidate and baseline symmetrically through all three roles."""
    if not candidate_policy_id or not baseline_policy_id:
        raise ValueError("policy IDs must be non-empty")
    if candidate_policy_id == baseline_policy_id:
        raise ValueError("candidate and baseline policy IDs must differ")

    comparisons: list[PairedComparison] = []
    for deal in deal_set.deals:
        for focal_seat, role in enumerate(SEAT_ROLES):
            candidate_ids = [baseline_policy_id] * len(SEAT_ROLES)
            baseline_ids = [candidate_policy_id] * len(SEAT_ROLES)
            candidate_ids[focal_seat] = candidate_policy_id
            baseline_ids[focal_seat] = baseline_policy_id
            comparisons.append(
                PairedComparison(
                    pair_id=f"{deal.deal_id}-{role.value}",
                    deal=deal,
                    focal_seat=focal_seat,
                    focal_role=role,
                    candidate_policy_id=candidate_policy_id,
                    baseline_policy_id=baseline_policy_id,
                    candidate_assignment=SeatAssignment(
                        (candidate_ids[0], candidate_ids[1], candidate_ids[2])
                    ),
                    baseline_assignment=SeatAssignment(
                        (baseline_ids[0], baseline_ids[1], baseline_ids[2])
                    ),
                )
            )
    return tuple(comparisons)


def generate_cross_play_schedule(
    deal_set: PairedDealSet,
    landlord_policy_ids: tuple[str, ...],
    farmer_policy_ids: tuple[str, ...],
) -> tuple[ScheduledMatch, ...]:
    """Build the full ordered landlord-versus-farmer-policy matrix."""
    if not landlord_policy_ids or not farmer_policy_ids:
        raise ValueError("cross-play policy lists must be non-empty")
    if any(not policy_id for policy_id in landlord_policy_ids + farmer_policy_ids):
        raise ValueError("cross-play policy IDs must be non-empty")
    if len(set(landlord_policy_ids)) != len(landlord_policy_ids):
        raise ValueError("landlord policy IDs must be unique")
    if len(set(farmer_policy_ids)) != len(farmer_policy_ids):
        raise ValueError("farmer policy IDs must be unique")

    schedule: list[ScheduledMatch] = []
    for landlord_id in landlord_policy_ids:
        for farmer_id in farmer_policy_ids:
            assignment = SeatAssignment((landlord_id, farmer_id, farmer_id))
            for deal in deal_set.deals:
                schedule.append(
                    ScheduledMatch(
                        match_id=f"{deal.deal_id}-L={landlord_id}-F={farmer_id}",
                        deal=deal,
                        assignment=assignment,
                    )
                )
    return tuple(schedule)


__all__ = (
    "DEAL_GENERATOR_ALGORITHM",
    "PAIRED_DEAL_SCHEMA_VERSION",
    "SEAT_ROLES",
    "PairedComparison",
    "PairedDeal",
    "PairedDealSet",
    "ScheduledMatch",
    "SeatAssignment",
    "SeatRole",
    "generate_cross_play_schedule",
    "generate_paired_comparisons",
    "generate_paired_deals",
    "role_for_game_seat",
    "role_for_seat",
    "splitmix64",
)
