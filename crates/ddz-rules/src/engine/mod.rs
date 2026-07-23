//! Authoritative game wrapper and phase transitions.

mod automatic;
mod error;
mod game;
mod history;
mod invariants;
mod landlord;
mod legal;
mod observe;
mod restore;
mod settle;
mod transition;

pub use error::{GameError, GameRestoreError};
pub use invariants::RuleStateError;
pub use game::{Game, StepResult, UndoToken};
