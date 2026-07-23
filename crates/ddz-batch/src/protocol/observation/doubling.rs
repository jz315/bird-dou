use ddz_core::{Observation, PublicDoublingState};

use crate::protocol::codes::{self, NO_SEAT};
use crate::BatchError;

use super::ObservationRow;

/// Information-safe doubling state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedDoubling {
    /// Disabled/not-started/in-progress/resolved code.
    pub kind: Vec<u8>,
    /// Eligibility mask.
    pub eligible_mask: Vec<u8>,
    /// Public acted mask.
    pub acted_mask: Vec<u8>,
    /// Public doubled mask; hidden until the round resolves.
    pub doubled_mask: Vec<u8>,
    /// Current doubling player or `-1`.
    pub current_player: Vec<i8>,
}

impl PackedDoubling {
    pub(crate) fn with_capacity(batch: usize) -> Self {
        Self {
            kind: Vec::with_capacity(batch),
            eligible_mask: Vec::with_capacity(batch),
            acted_mask: Vec::with_capacity(batch),
            doubled_mask: Vec::with_capacity(batch),
            current_player: Vec::with_capacity(batch),
        }
    }

    pub(crate) fn push(&mut self, row: &ObservationRow<'_>) {
        let value = row.observation.map_or_else(
            || Observation::public_doubling_from_private(&row.state.doubling),
            |observation| observation.doubling.clone(),
        );
        self.kind.push(codes::doubling_state(&value));
        match value {
            PublicDoublingState::Disabled | PublicDoublingState::NotStarted => {
                self.eligible_mask.push(0);
                self.acted_mask.push(0);
                self.doubled_mask.push(0);
                self.current_player.push(NO_SEAT);
            }
            PublicDoublingState::InProgress {
                eligible,
                acted,
                current_player,
            } => {
                self.eligible_mask.push(eligible.bits());
                self.acted_mask.push(acted.bits());
                self.doubled_mask.push(0);
                self.current_player.push(
                    current_player.map_or(NO_SEAT, |seat| codes::seat(seat)),
                );
            }
            PublicDoublingState::Resolved { eligible, doubled } => {
                self.eligible_mask.push(eligible.bits());
                self.acted_mask.push(eligible.bits());
                self.doubled_mask.push(doubled.bits());
                self.current_player.push(NO_SEAT);
            }
        }
    }

    pub(crate) fn validate(&self, expected: usize) -> Result<(), BatchError> {
        let lengths = [
            self.kind.len(),
            self.eligible_mask.len(),
            self.acted_mask.len(),
            self.doubled_mask.len(),
            self.current_player.len(),
        ];
        if lengths.into_iter().any(|length| length != expected) {
            return Err(BatchError::InternalInvariant(
                "packed doubling columns have inconsistent lengths",
            ));
        }
        Ok(())
    }
}
