//! Rule configuration, move detection, and legal-action generation for BIRD-Dou.

mod config;
mod deal;
mod detector;
mod engine;
mod generator;
mod match_v2;

pub use config::{
    AirplaneRules, AllPassPolicy, AttachmentMultiplicity, BeanCapPolicy, BiddingMode, BiddingRules,
    CallingRules, CardPlayRules, DealRules, DoubleEligibilityMode, DoublingRules,
    FirstCallerPolicy, FourWithTwoRules, RevealRoleEligibility, RevealRules, RewardMode,
    RewardRules, RobbingRules, RuleConfig, RuleConfigError, RuleConfigV1, RuleConfigV2,
    RuleProfile, SettlementRules, SpringRules, VersionedRuleConfig, RULE_CONFIG_SCHEMA_VERSION,
    RULE_CONFIG_V1_SCHEMA_VERSION, RULE_CONFIG_V2_SCHEMA_VERSION,
};
pub use deal::{
    deal_complete, deal_game, deal_post_bid, derive_attempt_seed, shuffled_deck_for_seed,
    SeededDealError, ATTEMPT_SEED_DERIVATION_ALGORITHM, PLAYER_COUNT, POST_BID_LANDLORD,
    SHUFFLE_ALGORITHM,
};
pub use detector::{detect_move, detect_move_with_rules, DetectMoveError};
pub use engine::{
    GameDeserializeError, GameError, GameInitError, GameRestoreError, HiddenSampleError,
    PostBidGame, UndoError, UndoToken,
};
pub use generator::{generate_follow_moves, generate_lead_moves, GenerateMovesError};
pub use match_v2::{
    AttemptActionRecordV2, AttemptCompletionReasonV2, AttemptStatusV2, AttemptSummaryV2,
    CallDecisionV2, DealAttemptStateV2, DoubleDecisionV2, GameActionV2, HuanleMatchV2,
    MatchCompletionV2, MatchDecisionEventV2, MatchError, MatchStateV2, RevealDecisionV2,
    RobDecisionV2, SystemEventRecordV2, SystemEventV2,
};
