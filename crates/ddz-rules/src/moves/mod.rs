//! Move detection, comparison, and rank-template generation.

mod attachments;
mod compare;
mod detect;
mod generate;

pub use compare::move_beats;
pub use detect::{
    detect_move, detect_move_with_rules, validate_move_for_rules, DetectMoveError,
};
pub use generate::{generate_follow_moves, generate_lead_moves, GenerateMovesError};
