use ddz_core::GameState;

use crate::protocol::codes;
use crate::BatchError;

/// Public multiplier decomposition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedStake {
    /// Base room unit.
    pub base_unit: Vec<u32>,
    /// Maximum reveal factor.
    pub reveal_factor: Vec<u32>,
    /// Base-two rob exponent.
    pub rob_exponent: Vec<u8>,
    /// Base-two bomb/rocket exponent.
    pub bomb_exponent: Vec<u8>,
    /// Spring code.
    pub spring: Vec<u8>,
    /// Fully multiplied common stake before per-seat doubling.
    pub common_stake: Vec<u64>,
}

impl PackedStake {
    pub(crate) fn with_capacity(batch: usize) -> Self {
        Self {
            base_unit: Vec::with_capacity(batch),
            reveal_factor: Vec::with_capacity(batch),
            rob_exponent: Vec::with_capacity(batch),
            bomb_exponent: Vec::with_capacity(batch),
            spring: Vec::with_capacity(batch),
            common_stake: Vec::with_capacity(batch),
        }
    }

    pub(crate) fn push(&mut self, slot: usize, state: &GameState) -> Result<(), BatchError> {
        self.base_unit.push(state.stake.base_unit);
        self.reveal_factor.push(state.stake.reveal_factor);
        self.rob_exponent.push(state.stake.rob_exponent);
        self.bomb_exponent.push(state.stake.bomb_exponent);
        self.spring.push(codes::spring(state.stake.spring));
        self.common_stake.push(
            state
                .stake
                .common_stake()
                .map_err(|source| BatchError::PackStake { slot, source })?,
        );
        Ok(())
    }

    pub(crate) fn validate(&self, expected: usize) -> Result<(), BatchError> {
        let lengths = [
            self.base_unit.len(),
            self.reveal_factor.len(),
            self.rob_exponent.len(),
            self.bomb_exponent.len(),
            self.spring.len(),
            self.common_stake.len(),
        ];
        if lengths.into_iter().any(|length| length != expected) {
            return Err(BatchError::InternalInvariant(
                "packed stake columns have inconsistent lengths",
            ));
        }
        Ok(())
    }
}
