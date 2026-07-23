#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_field_names
)]
#![doc = "Authoritative rule configuration, move logic, and transitions for BIRD-Dou."]

mod deal;
mod economy;
mod engine;
mod moves;
mod settlement;

pub mod config;

pub use config::{
    AirplaneRules, AttachmentMultiplicity, CallingRules, DoublingRules, FourWithTwoRules,
    MoveRules, RevealRules, RewardMode, RobbingRules, RuleConfig, RuleConfigError, RuleProfile,
    SettlementRules, SpringRules, RULE_CONFIG_SCHEMA_VERSION,
};
pub use deal::{
    deal_plan_for_attempt, derive_attempt_seed, first_player_for_attempt, shuffled_deck,
    DealError, ATTEMPT_SEED_ALGORITHM, FIRST_PLAYER_ALGORITHM, SHUFFLE_ALGORITHM,
};
pub use economy::EconomyContext;
pub use engine::{Game, GameError, GameRestoreError, RuleStateError, StepResult, UndoToken};
pub use moves::{
    detect_move, detect_move_with_rules, generate_follow_moves, generate_lead_moves, move_beats,
    validate_move_for_rules, DetectMoveError, GenerateMovesError,
};
pub use settlement::{settle_game, terminal_reward, SettlementError};
