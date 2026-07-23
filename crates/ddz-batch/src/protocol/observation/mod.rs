//! Fixed-width current-state observations. Public history is packed separately.

mod card_play;
mod cards;
mod doubling;
mod landlord;
mod outcome;
mod reveal;
mod stake;
mod status;

pub use card_play::PackedCardPlay;
pub use cards::PackedCards;
pub use doubling::PackedDoubling;
pub use landlord::PackedLandlord;
pub use outcome::PackedOutcome;
pub use reveal::PackedReveal;
pub use stake::PackedStake;
pub use status::PackedStatus;

use ddz_core::{GameState, Observation};

use crate::{BatchError, SlotVersion, BATCH_SCHEMA_VERSION};

/// One source row used while packing a batch.
pub(crate) struct ObservationRow<'a> {
    pub(crate) state: &'a GameState,
    pub(crate) version: SlotVersion,
    pub(crate) observation: Option<&'a Observation>,
}

/// Nested structure-of-arrays representation of current game state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedObservation {
    /// Packed-buffer protocol version.
    pub schema_version: u32,
    /// Number of environment slots.
    pub batch_size: usize,
    /// Phase, observer, version, and match-attempt metadata.
    pub status: PackedStatus,
    /// Information-set-dependent hand and unknown-card fields.
    pub cards: PackedCards,
    /// Reveal metadata.
    pub reveal: PackedReveal,
    /// Calling/robbing state.
    pub landlord: PackedLandlord,
    /// Information-safe doubling state.
    pub doubling: PackedDoubling,
    /// Public multiplier decomposition.
    pub stake: PackedStake,
    /// Public card-play state.
    pub card_play: PackedCardPlay,
    /// Terminal outcome, when present.
    pub outcome: PackedOutcome,
}

impl PackedObservation {
    pub(crate) fn from_rows(rows: &[ObservationRow<'_>]) -> Result<Self, BatchError> {
        let mut result = Self {
            schema_version: BATCH_SCHEMA_VERSION,
            batch_size: rows.len(),
            status: PackedStatus::with_capacity(rows.len()),
            cards: PackedCards::with_capacity(rows.len()),
            reveal: PackedReveal::with_capacity(rows.len()),
            landlord: PackedLandlord::with_capacity(rows.len()),
            doubling: PackedDoubling::with_capacity(rows.len()),
            stake: PackedStake::with_capacity(rows.len()),
            card_play: PackedCardPlay::with_capacity(rows.len()),
            outcome: PackedOutcome::with_capacity(rows.len()),
        };
        for (slot, row) in rows.iter().enumerate() {
            result.status.push(row);
            result.cards.push(row);
            result.reveal.push(row.state);
            result.landlord.push(row.state);
            result.doubling.push(row);
            result.stake.push(slot, row.state)?;
            result.card_play.push(row.state);
            result.outcome.push(row.state);
        }
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<(), BatchError> {
        self.status.validate(self.batch_size)?;
        self.cards.validate(self.batch_size)?;
        self.reveal.validate(self.batch_size)?;
        self.landlord.validate(self.batch_size)?;
        self.doubling.validate(self.batch_size)?;
        self.stake.validate(self.batch_size)?;
        self.card_play.validate(self.batch_size)?;
        self.outcome.validate(self.batch_size)
    }
}
