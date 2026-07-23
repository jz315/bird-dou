//! Stable structure-of-arrays buffers intended for PyO3, NumPy, and learner actors.
//!
//! Fixed current-state data and variable public history are deliberately separate. Calling
//! [`crate::BatchEnv::observations_current`] does not repack the complete history on every turn;
//! consumers request [`crate::BatchEnv::public_history_packed`] only when they need a public
//! resynchronization. Raw step deltas are authoritative audit data and are not automatically safe
//! for every player information set during an unresolved doubling round.

mod actions;
mod codes;
mod events;
mod step;
pub mod observation;

pub use actions::{PackedActionData, PackedActions};
pub use codes::{
    ACTION_CALL, ACTION_DOUBLE, ACTION_PLAY, ACTION_REVEAL, ACTION_ROB, DECISION_NO,
    DECISION_YES, DOUBLING_DISABLED, DOUBLING_IN_PROGRESS, DOUBLING_NOT_STARTED,
    DOUBLING_RESOLVED, EVENT_PLAYER, EVENT_SYSTEM, LANDLORD_CALLING, LANDLORD_NOT_STARTED,
    LANDLORD_POST_BID, LANDLORD_RESOLVED, LANDLORD_ROBBING, NO_ACTION, NO_EVENT_CODE,
    NO_RANK, NO_SEAT, NO_U8, PHASE_BOTTOM_REVEAL, PHASE_CALLING, PHASE_CARD_PLAY,
    PHASE_DEALING, PHASE_DOUBLING, PHASE_POST_BOTTOM_REVEAL, PHASE_PRE_DEAL,
    PHASE_ROBBING, PHASE_TERMINAL, REVEAL_AFTER_BOTTOM, REVEAL_BEFORE_DEAL,
    REVEAL_DURING_DEAL, ROLE_FARMER, ROLE_LANDLORD, ROLE_UNASSIGNED, SPRING_FARMER,
    SPRING_LANDLORD, SPRING_NONE, SYSTEM_BOTTOM_REVEALED, SYSTEM_CARD_PLAY_STARTED,
    SYSTEM_DEAL_ROUND, SYSTEM_LANDLORD_RESOLVED, SYSTEM_REDEAL,
};
pub use events::PackedEvents;
pub use observation::PackedObservation;
pub use step::PackedStepResult;
pub(crate) use step::StepPackRow;
