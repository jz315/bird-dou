use ddz_core::{GameState, Seat};

use crate::protocol::codes::{self, NO_SEAT};
use crate::BatchError;

/// Terminal outcome snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedOutcome {
    /// Whether an outcome exists.
    pub valid: Vec<u8>,
    /// Winning seat or `-1`.
    pub winner: Vec<i8>,
    /// Landlord seat or `-1`.
    pub landlord: Vec<i8>,
    /// Spring code.
    pub spring: Vec<u8>,
    /// Flattened `[B, 3]` raw zero-sum payoff.
    pub payoff: Vec<i64>,
}

impl PackedOutcome {
    pub(crate) fn with_capacity(batch: usize) -> Self {
        Self {
            valid: Vec::with_capacity(batch),
            winner: Vec::with_capacity(batch),
            landlord: Vec::with_capacity(batch),
            spring: Vec::with_capacity(batch),
            payoff: Vec::with_capacity(batch.saturating_mul(3)),
        }
    }

    pub(crate) fn push(&mut self, state: &GameState) {
        match &state.outcome {
            Some(outcome) => {
                self.valid.push(1);
                self.winner.push(codes::seat(outcome.winner));
                self.landlord.push(codes::seat(outcome.landlord));
                self.spring.push(codes::spring(outcome.spring));
                for seat in Seat::ALL {
                    self.payoff.push(outcome.payoff[seat]);
                }
            }
            None => {
                self.valid.push(0);
                self.winner.push(NO_SEAT);
                self.landlord.push(NO_SEAT);
                self.spring.push(codes::spring(ddz_core::SpringKind::None));
                self.payoff.extend_from_slice(&[0; 3]);
            }
        }
    }

    pub(crate) fn validate(&self, batch: usize) -> Result<(), BatchError> {
        if self.valid.len() != batch
            || self.winner.len() != batch
            || self.landlord.len() != batch
            || self.spring.len() != batch
            || self.payoff.len() != batch.saturating_mul(3)
        {
            return Err(BatchError::InternalInvariant(
                "packed outcome columns have inconsistent lengths",
            ));
        }
        Ok(())
    }
}
