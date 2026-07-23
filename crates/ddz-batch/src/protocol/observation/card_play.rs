use ddz_core::{GameState, RankCounts, Seat, RANK_COUNT};

use crate::protocol::codes;
use crate::protocol::codes::{NO_ACTION, NO_RANK, NO_SEAT};
use crate::BatchError;

/// Public card-play state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedCardPlay {
    /// Flattened `[B, 3, 15]` public played-card counts.
    pub played_cards: Vec<u8>,
    /// Whether an active target move exists.
    pub target_valid: Vec<u8>,
    /// Flattened `[B, 15]` target cards.
    pub target_cards: Vec<u8>,
    /// Target move kind or `255`.
    pub target_kind: Vec<u8>,
    /// Target main rank or `255`.
    pub target_main_rank: Vec<u8>,
    /// Target chain length.
    pub target_chain_len: Vec<u8>,
    /// Target total cards.
    pub target_total_cards: Vec<u8>,
    /// Player who produced the target or `-1`.
    pub target_player: Vec<i8>,
    /// Consecutive passes after the target.
    pub consecutive_passes: Vec<u8>,
    /// Flattened `[B, 3]` non-pass play counts.
    pub non_pass_plays: Vec<u16>,
    /// Number of bomb-like plays.
    pub bomb_count: Vec<u16>,
}

impl PackedCardPlay {
    pub(crate) fn with_capacity(batch: usize) -> Self {
        Self {
            played_cards: Vec::with_capacity(batch.saturating_mul(3 * RANK_COUNT)),
            target_valid: Vec::with_capacity(batch),
            target_cards: Vec::with_capacity(batch.saturating_mul(RANK_COUNT)),
            target_kind: Vec::with_capacity(batch),
            target_main_rank: Vec::with_capacity(batch),
            target_chain_len: Vec::with_capacity(batch),
            target_total_cards: Vec::with_capacity(batch),
            target_player: Vec::with_capacity(batch),
            consecutive_passes: Vec::with_capacity(batch),
            non_pass_plays: Vec::with_capacity(batch.saturating_mul(3)),
            bomb_count: Vec::with_capacity(batch),
        }
    }

    pub(crate) fn push(&mut self, state: &GameState) {
        for seat in Seat::ALL {
            self.played_cards
                .extend_from_slice(state.card_play.played_cards[seat].as_array());
            self.non_pass_plays
                .push(state.card_play.non_pass_plays[seat]);
        }
        match state.card_play.last_non_pass {
            Some(movement) => {
                self.target_valid.push(1);
                self.target_cards
                    .extend_from_slice(movement.cards().as_array());
                self.target_kind.push(movement.kind() as u8);
                self.target_main_rank.push(movement.main_rank());
                self.target_chain_len.push(movement.chain_len());
                self.target_total_cards.push(movement.total_cards());
            }
            None => {
                self.target_valid.push(0);
                self.target_cards
                    .extend_from_slice(RankCounts::empty().as_array());
                self.target_kind.push(NO_ACTION);
                self.target_main_rank.push(NO_RANK);
                self.target_chain_len.push(0);
                self.target_total_cards.push(0);
            }
        }
        self.target_player.push(
            state
                .card_play
                .last_non_pass_player
                .map_or(NO_SEAT, |seat| codes::seat(seat)),
        );
        self.consecutive_passes
            .push(state.card_play.consecutive_passes);
        self.bomb_count.push(state.card_play.bomb_count);
    }

    pub(crate) fn validate(&self, batch: usize) -> Result<(), BatchError> {
        if self.played_cards.len() != batch.saturating_mul(3 * RANK_COUNT)
            || self.target_valid.len() != batch
            || self.target_cards.len() != batch.saturating_mul(RANK_COUNT)
            || self.target_kind.len() != batch
            || self.target_main_rank.len() != batch
            || self.target_chain_len.len() != batch
            || self.target_total_cards.len() != batch
            || self.target_player.len() != batch
            || self.consecutive_passes.len() != batch
            || self.non_pass_plays.len() != batch.saturating_mul(3)
            || self.bomb_count.len() != batch
        {
            return Err(BatchError::InternalInvariant(
                "packed card-play columns have inconsistent lengths",
            ));
        }
        Ok(())
    }
}
