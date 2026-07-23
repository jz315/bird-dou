use ddz_core::{RankCounts, Seat, RANK_COUNT};

use crate::BatchError;

use super::ObservationRow;

/// Information-set-dependent cards and public hand sizes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedCards {
    /// Flattened `[B, 15]` observer hands.
    pub own_hand: Vec<u8>,
    /// Flattened `[B, 3, 15]` explicitly revealed opponent hands.
    pub revealed_hands: Vec<u8>,
    /// Flattened `[B, 3]` reveal-presence mask.
    pub revealed_mask: Vec<u8>,
    /// Flattened `[B, 15]` union of truly hidden current and undealt cards.
    pub unknown_pool: Vec<u8>,
    /// Flattened `[B, 3]` cards left.
    pub cards_left: Vec<u8>,
    /// Flattened `[B, 15]` public bottom cards, zero when hidden.
    pub public_bottom: Vec<u8>,
    /// Whether the bottom buffer is meaningful.
    pub bottom_visible: Vec<u8>,
}

impl PackedCards {
    pub(crate) fn with_capacity(batch: usize) -> Self {
        Self {
            own_hand: Vec::with_capacity(batch.saturating_mul(RANK_COUNT)),
            revealed_hands: Vec::with_capacity(batch.saturating_mul(3 * RANK_COUNT)),
            revealed_mask: Vec::with_capacity(batch.saturating_mul(3)),
            unknown_pool: Vec::with_capacity(batch.saturating_mul(RANK_COUNT)),
            cards_left: Vec::with_capacity(batch.saturating_mul(3)),
            public_bottom: Vec::with_capacity(batch.saturating_mul(RANK_COUNT)),
            bottom_visible: Vec::with_capacity(batch),
        }
    }

    pub(crate) fn push(&mut self, row: &ObservationRow<'_>) {
        let empty = RankCounts::empty();
        let own = row.observation.map_or(empty, |value| value.own_hand);
        self.own_hand.extend_from_slice(own.as_array());

        for seat in Seat::ALL {
            let revealed = row
                .observation
                .and_then(|value| value.revealed_hands[seat]);
            self.revealed_mask.push(u8::from(revealed.is_some()));
            self.revealed_hands
                .extend_from_slice(revealed.unwrap_or(empty).as_array());
        }

        let unknown = row.observation.map_or(empty, |value| value.unknown_pool);
        self.unknown_pool.extend_from_slice(unknown.as_array());
        let cards_left = row.state.cards_left();
        for seat in Seat::ALL {
            self.cards_left.push(cards_left[seat]);
        }
        let bottom = row
            .observation
            .and_then(|value| value.public_bottom_cards);
        self.bottom_visible.push(u8::from(bottom.is_some()));
        self.public_bottom
            .extend_from_slice(bottom.unwrap_or(empty).as_array());
    }

    pub(crate) fn validate(&self, batch: usize) -> Result<(), BatchError> {
        let expected_rank = batch.saturating_mul(RANK_COUNT);
        if self.own_hand.len() != expected_rank
            || self.unknown_pool.len() != expected_rank
            || self.public_bottom.len() != expected_rank
            || self.revealed_hands.len() != batch.saturating_mul(3 * RANK_COUNT)
            || self.revealed_mask.len() != batch.saturating_mul(3)
            || self.cards_left.len() != batch.saturating_mul(3)
            || self.bottom_visible.len() != batch
        {
            return Err(BatchError::InternalInvariant(
                "packed observation card buffers have inconsistent lengths",
            ));
        }
        Ok(())
    }
}
