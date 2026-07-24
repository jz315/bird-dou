mod compare;
mod detect;
mod generate;
mod kind;
mod sequence;
pub(crate) mod strength;

pub use compare::beats;
pub use detect::{detect_move, DetectError};
pub use generate::generate_legal_moves;
pub use kind::{Move, MoveKind};
