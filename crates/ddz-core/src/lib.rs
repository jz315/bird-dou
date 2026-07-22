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
#![doc = "Domain types for BIRD-Dou. This crate contains no rule engine or policy logic."]

mod action;
mod cards;
mod codec;
mod deal;
mod doubling;
mod event;
mod landlord;
mod moves;
mod observation;
mod phase;
mod reveal;
mod seat;
mod stake;
mod state;

pub use action::{CallAction, DoubleAction, GameAction, RevealAction, RobAction};
pub use cards::{
    CardId, CardIdError, DeckOrder, DeckOrderError, Rank, RankCounts, RankCountsError, BIG_JOKER,
    CARD_COUNT, EMPTY_RANK_COUNTS, PLAYER_COUNT, RANK_COUNT, SMALL_JOKER,
};
pub use codec::{
    decode_observation, decode_state, encode_observation, encode_state, CodecError,
    OBSERVATION_SCHEMA_VERSION, STATE_SCHEMA_VERSION,
};
pub use deal::{DealPlan, DealState, DealStateError, DEAL_ROUNDS};
pub use doubling::{
    DoublingRound, DoublingState, DoublingStateError, PublicDoublingState,
};
pub use event::{GameEvent, GameEventKind, PlayerEvent, SystemEvent};
pub use landlord::{
    CallingState, LandlordSelectionState, LandlordStateError, ResolvedLandlord, RobbingState,
};
pub use moves::{Move, MoveError, MoveKind, PASS_MAIN_RANK};
pub use observation::{Observation, ObservationError};
pub use phase::{Phase, Role};
pub use reveal::{RevealInfo, RevealState, RevealStateError, RevealTiming};
pub use seat::{Seat, SeatError, SeatMap, SeatOrder, SeatOrderError, SeatSet, SeatSetError};
pub use stake::{SpringKind, StakeError, StakeState};
pub use state::{
    CardPlayState, CardPlayStateError, GameOutcome, GameState, GameStateError,
};
