use ddz_core::{GameState, Seat};

use crate::protocol::codes::{self, NO_SEAT, NO_U8};
use crate::BatchError;

/// Per-seat reveal timing and multiplier metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedReveal {
    /// Flattened `[B, 3]` reveal-presence mask.
    pub revealed: Vec<u8>,
    /// Flattened `[B, 3]` timing code.
    pub timing: Vec<u8>,
    /// Flattened `[B, 3]` cards received for during-deal reveal; `255` otherwise.
    pub cards_received: Vec<u8>,
    /// Flattened `[B, 3]` reveal factors.
    pub factor: Vec<u32>,
    /// First revealer or `-1`.
    pub first_revealer: Vec<i8>,
    /// Maximum reveal factor currently in force.
    pub maximum_factor: Vec<u32>,
}

impl PackedReveal {
    pub(crate) fn with_capacity(batch: usize) -> Self {
        Self {
            revealed: Vec::with_capacity(batch.saturating_mul(3)),
            timing: Vec::with_capacity(batch.saturating_mul(3)),
            cards_received: Vec::with_capacity(batch.saturating_mul(3)),
            factor: Vec::with_capacity(batch.saturating_mul(3)),
            first_revealer: Vec::with_capacity(batch),
            maximum_factor: Vec::with_capacity(batch),
        }
    }

    pub(crate) fn push(&mut self, state: &GameState) {
        for seat in Seat::ALL {
            match state.reveal.by_seat[seat] {
                Some(info) => {
                    let (timing, cards_received) = codes::reveal_timing(info.timing);
                    self.revealed.push(1);
                    self.timing.push(timing);
                    self.cards_received.push(cards_received);
                    self.factor.push(info.factor);
                }
                None => {
                    self.revealed.push(0);
                    self.timing.push(NO_U8);
                    self.cards_received.push(NO_U8);
                    self.factor.push(1);
                }
            }
        }
        self.first_revealer.push(
            state
                .reveal
                .first_revealer
                .map_or(NO_SEAT, |seat| codes::seat(seat)),
        );
        self.maximum_factor.push(state.reveal.maximum_factor);
    }

    pub(crate) fn validate(&self, batch: usize) -> Result<(), BatchError> {
        let per_seat = batch.saturating_mul(3);
        if self.revealed.len() != per_seat
            || self.timing.len() != per_seat
            || self.cards_received.len() != per_seat
            || self.factor.len() != per_seat
            || self.first_revealer.len() != batch
            || self.maximum_factor.len() != batch
        {
            return Err(BatchError::InternalInvariant(
                "packed reveal columns have inconsistent lengths",
            ));
        }
        Ok(())
    }
}

