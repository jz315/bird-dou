"""Typed Python object protocol for the native single-game environment."""

from typing import Literal, TypeAlias, TypedDict

RankCounts: TypeAlias = list[int]
Phase: TypeAlias = Literal["bidding", "doubling", "card_play", "terminal"]
Role: TypeAlias = Literal["unassigned", "landlord", "farmer"]


class Move(TypedDict):
    """Canonical card-play payload emitted by Rust."""

    kind: str
    cards: RankCounts
    main_rank: int
    chain_len: int
    total_cards: int


BidAction: TypeAlias = Literal["pass", "call", "rob"] | dict[str, int]
DoubleAction: TypeAlias = Literal["decline", "double"]


class BidGameAction(TypedDict):
    """Bidding action representation reserved for the complete-game profile."""

    bid: BidAction


class DoubleGameAction(TypedDict):
    """Doubling action representation reserved for the complete-game profile."""

    double: DoubleAction


class PlayGameAction(TypedDict):
    """Card-play action accepted by the E010 post-bid environment."""

    play: Move


Action: TypeAlias = BidGameAction | DoubleGameAction | PlayGameAction


class BiddingRules(TypedDict):
    """Landlord-selection configuration."""

    mode: Literal["disabled", "score", "rob"]
    max_bid: int | None
    redeal_on_all_pass: bool


class SpringRules(TypedDict):
    """Spring scoring configuration."""

    landlord_spring_enabled: bool
    anti_spring_enabled: bool
    multiplier: int


class FourWithTwoRules(TypedDict):
    """Four-with-two attachment boundaries."""

    two_singles_enabled: bool
    two_pairs_enabled: bool
    single_attachments: Literal["distinct_ranks", "may_share_rank"]
    pair_attachments: Literal["distinct_ranks", "may_share_rank"]


class AirplaneRules(TypedDict):
    """Airplane attachment boundaries."""

    single_attachments: Literal["distinct_ranks", "may_share_rank"]
    pair_attachments: Literal["distinct_ranks", "may_share_rank"]


class RuleConfig(TypedDict):
    """Versioned rule dictionary consumed by :meth:`PyDdzEnv.reset`."""

    schema_version: int
    rule_config_id: int
    profile: Literal["douzero_post_bid", "canonical_full"]
    landlord_plays_first: bool
    bidding: BiddingRules
    bottom_cards_public: bool
    doubling_enabled: bool
    bomb_multiplier: int
    rocket_multiplier: int
    spring: SpringRules
    four_with_two: FourWithTwoRules
    airplane: AirplaneRules
    score_cap: int | None
    reward_mode: Literal[
        "win_percentage",
        "average_difference_points",
        "log_average_difference_points",
        "raw_score",
    ]


class GameEvent(TypedDict):
    """One authoritative state-transition event."""

    sequence: int
    actor: int
    action: Action


class BidEvent(TypedDict):
    """One public bidding event."""

    sequence: int
    actor: int
    action: BidAction


class Observation(TypedDict):
    """Information-set-safe observation returned as Python-native buffers."""

    schema_version: int
    phase: Phase
    observer: int
    role: Role
    own_hand: RankCounts
    public_played: list[RankCounts]
    public_bottom_cards: RankCounts
    unknown_pool: RankCounts
    cards_left: list[int]
    current_player: int
    landlord: int | None
    last_non_pass: Move | None
    consecutive_passes: int
    bid_history: list[BidEvent]
    history: list[GameEvent]
    multiplier_exp: int
    bomb_count: int


class StepResult(TypedDict):
    """Result of one successful native transition."""

    event: GameEvent
    next_player: int | None
    terminal: bool
    raw_payoff: list[int]
    objective_payoff: list[int]


class ExactSolveResult(TypedDict):
    """Proven perfect-information endgame result from the native solver."""

    landlord_forced_win: bool
    plies_to_terminal: int
    best_action: Action | None
    nodes: int
    cache_hits: int


__all__ = (
    "Action",
    "AirplaneRules",
    "BidAction",
    "BidEvent",
    "BidGameAction",
    "BiddingRules",
    "DoubleAction",
    "DoubleGameAction",
    "ExactSolveResult",
    "FourWithTwoRules",
    "GameEvent",
    "Move",
    "Observation",
    "Phase",
    "PlayGameAction",
    "RankCounts",
    "Role",
    "RuleConfig",
    "SpringRules",
    "StepResult",
)
