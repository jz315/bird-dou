#![forbid(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate
)]
#![doc = "Two-deck, four-player Guandan rules with physical suits and level wildcards."]

mod card;
mod config;
mod game;
mod movement;
mod report;

pub use card::{
    all_cards, Card, CardError, Hand, HandError, Rank, Seat, SeatError, Suit, Team,
    CARDS_PER_PLAYER, CARD_COUNT, PLAYER_COUNT,
};
pub use config::{
    RuleConfig, RuleConfigError, RuleProfile, RULE_CONFIG_SCHEMA_VERSION, RULE_SOURCE_URL,
};
pub use game::{
    Action, GameError, MatchProgress, Round, RoundOutcome, StepResult, TributeAssignment,
    TributeError, TributeMode, TributePlan, TributeResolution, TributeTransfer,
};
pub use movement::{beats, detect_move, generate_legal_moves, DetectError, Move, MoveKind};
pub use report::{report_duty, ReportDuty};
