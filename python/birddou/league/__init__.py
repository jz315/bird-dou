"""League population, deterministic matchmaking, snapshots, and promotion gates."""

from .matchmaking import (
    LEAGUE_MATCHMAKING_SCHEMA_VERSION,
    LeagueMatch,
    LeagueMatchmaker,
    LeagueMatchmakingConfig,
    MatchCategory,
    load_league_matchmaking_config,
)
from .population import (
    LEAGUE_POPULATION_SCHEMA_VERSION,
    LeagueMember,
    LeagueMemberKind,
    LeaguePopulation,
    LeagueRole,
)
from .snapshot import (
    LEAGUE_SNAPSHOT_SCHEMA_VERSION,
    LeagueSnapshot,
    PromotionMetrics,
    PromotionReport,
    PromotionThresholds,
    create_self_play_snapshot,
    evaluate_promotion,
)

__all__ = (
    "LEAGUE_MATCHMAKING_SCHEMA_VERSION",
    "LEAGUE_POPULATION_SCHEMA_VERSION",
    "LEAGUE_SNAPSHOT_SCHEMA_VERSION",
    "LeagueMatch",
    "LeagueMatchmaker",
    "LeagueMatchmakingConfig",
    "LeagueMember",
    "LeagueMemberKind",
    "LeaguePopulation",
    "LeagueRole",
    "LeagueSnapshot",
    "MatchCategory",
    "PromotionMetrics",
    "PromotionReport",
    "PromotionThresholds",
    "create_self_play_snapshot",
    "evaluate_promotion",
    "load_league_matchmaking_config",
)
