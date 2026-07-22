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
    """Legacy v1 rule dictionary consumed by :meth:`PyDdzEnv.reset`."""

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


class DealRulesV2(TypedDict):
    """Fixed deal and turn-order choices for the Huanle v2 schema."""

    player_count: int
    cards_per_player: int
    bottom_card_count: int
    bottom_visible_before_landlord: bool
    bottom_visible_after_landlord: bool
    landlord_plays_first: bool


class RevealRulesV2(TypedDict):
    """All Huanle reveal opportunities and their explicit factors."""

    before_deal_enabled: bool
    before_deal_factor: int
    during_deal_enabled: bool
    factor_by_cards_received: list[int]
    after_bottom_enabled: bool
    after_bottom_factor: int
    maximum_factor_only: bool
    after_bottom_eligible_roles: Literal["landlord_only", "any_player"]


class CallingRulesV2(TypedDict):
    """Huanle call-stage policy."""

    first_caller_policy: Literal["first_revealer_else_seeded_seat", "seeded_seat"]
    call_ends_immediately: bool
    all_pass_policy: Literal[
        "redeal",
        "first_revealer_becomes_landlord",
        "first_revealer_becomes_landlord_else_redeal",
    ]
    passed_call_loses_rob_eligibility: bool


class RobbingRulesV2(TypedDict):
    """Huanle rob-stage policy, including the explicit reclaim choice."""

    enabled: bool
    factor_per_successful_rob: int
    each_eligible_player_once: bool
    caller_can_reclaim: bool


class DoublingRulesV2(TypedDict):
    """Per-seat Huanle doubling policy."""

    enabled: bool
    factor: int
    eligibility_mode: Literal["all_players", "room_balance_threshold"]
    room_balance_threshold: int


class CardPlayRulesV2(TypedDict):
    """Huanle card-play policy with explicit attachment choices."""

    wildcards_enabled: bool
    four_with_two: FourWithTwoRules
    airplane: AirplaneRules


class SettlementRulesV2(TypedDict):
    """Pairwise Huanle settlement policy."""

    base_unit: int
    pairwise_landlord_farmer: bool
    spring: SpringRules
    score_cap: int | None
    bean_cap_policy: Literal["none", "available_balance"]


class RewardRulesV2(TypedDict):
    """Training-facing conversion after authoritative Huanle settlement."""

    mode: Literal[
        "win_percentage",
        "average_difference_points",
        "log_average_difference_points",
        "raw_score",
    ]


class RuleConfigV2(TypedDict):
    """Parsed Huanle v2 rule dictionary; it is not executable by :class:`PyDdzEnv` yet."""

    schema_version: Literal[2]
    rule_config_id: int
    profile: Literal["huanle_classic_v1"]
    deal: DealRulesV2
    reveal: RevealRulesV2
    calling: CallingRulesV2
    robbing: RobbingRulesV2
    doubling: DoublingRulesV2
    card_play: CardPlayRulesV2
    settlement: SettlementRulesV2
    reward: RewardRulesV2


VersionedRuleConfig: TypeAlias = RuleConfig | RuleConfigV2


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
    "CallingRulesV2",
    "CardPlayRulesV2",
    "DealRulesV2",
    "DoubleAction",
    "DoubleGameAction",
    "DoublingRulesV2",
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
    "RuleConfigV2",
    "RobbingRulesV2",
    "RewardRulesV2",
    "RevealRulesV2",
    "SettlementRulesV2",
    "SpringRules",
    "StepResult",
    "VersionedRuleConfig",
)
