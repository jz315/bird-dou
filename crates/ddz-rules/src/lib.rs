//! Rule configuration, move detection, and legal-action generation for BIRD-Dou.

mod config;
mod deal;
mod detector;
mod engine;
mod generator;

pub use config::{
    AirplaneRules, AllPassPolicy, AttachmentMultiplicity, BeanCapPolicy, BiddingMode, BiddingRules,
    CallingRules, CardPlayRules, DealRules, DoubleEligibilityMode, DoublingRules,
    FirstCallerPolicy, FourWithTwoRules, RevealRoleEligibility, RevealRules, RewardMode,
    RewardRules, RobbingRules, RuleConfig, RuleConfigError, RuleConfigV1, RuleConfigV2,
    RuleProfile, SettlementRules, SpringRules, VersionedRuleConfig, RULE_CONFIG_SCHEMA_VERSION,
    RULE_CONFIG_V1_SCHEMA_VERSION, RULE_CONFIG_V2_SCHEMA_VERSION,
};
pub use deal::{
    deal_complete, deal_game, deal_post_bid, SeededDealError, POST_BID_LANDLORD, SHUFFLE_ALGORITHM,
};
pub use detector::{detect_move, detect_move_with_rules, DetectMoveError};
pub use engine::{
    GameDeserializeError, GameError, GameInitError, GameRestoreError, HiddenSampleError,
    PostBidGame, UndoError, UndoToken,
};
pub use generator::{generate_follow_moves, generate_lead_moves, GenerateMovesError};
