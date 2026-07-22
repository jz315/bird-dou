//! Core state, action, replay, and serialization types for BIRD-Dou.

mod cards;
mod game;
mod moves;
mod state_codec;

pub use cards::{
    card_id_to_rank, cards_to_rank_counts, max_count_for_rank, rank_counts_to_card_ids,
    rank_to_card_ids, validate_rank_counts, CardError, CardId, RankCounts, RankId, Seat,
    BIG_JOKER_CARD, BIG_JOKER_RANK, CARD_COUNT, EMPTY_RANK_COUNTS, RANK_COUNT, SMALL_JOKER_CARD,
    SMALL_JOKER_RANK,
};
pub use game::{
    BidAction, BidEvent, BidState, DoubleAction, GameAction, GameEvent, GameState, Observation,
    Phase, PublicEvent, Role, SpringState, StepResult, OBSERVATION_SCHEMA_VERSION,
};
pub use moves::{Move, MoveError, MoveKind, PASS_MAIN_RANK};
pub use state_codec::{
    deserialize_game_state, serialize_game_state, StateCodecError, GAME_STATE_SCHEMA_VERSION,
};
