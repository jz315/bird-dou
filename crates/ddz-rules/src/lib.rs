//! Rule configuration, move detection, and legal-action generation for BIRD-Dou.

mod config;
mod deal;
mod detector;
mod engine;
mod generator;

pub use config::{
    AirplaneRules, AttachmentMultiplicity, BiddingMode, BiddingRules, FourWithTwoRules, RewardMode,
    RuleConfig, RuleConfigError, RuleProfile, SpringRules, RULE_CONFIG_SCHEMA_VERSION,
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
