//! Slot identity used to reject stale asynchronous inference results.

use std::fmt::{Display, Formatter};

/// Monotonic identity of one environment slot.
///
/// `generation` changes whenever the slot is reset or restored. `revision` changes after
/// every committed player action. A policy response can carry both values back to
/// [`crate::BatchEnv::step_packed_checked`] to prove that it was produced for the current state.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SlotVersion {
    /// Reset/restore generation.
    pub generation: u64,
    /// Number of committed actions since the current generation began.
    pub revision: u64,
}

impl SlotVersion {
    pub(crate) const fn initial(generation: u64) -> Self {
        Self {
            generation,
            revision: 0,
        }
    }

    pub(crate) fn advanced(self) -> Option<Self> {
        self.revision.checked_add(1).map(|revision| Self {
            generation: self.generation,
            revision,
        })
    }
}

impl Display for SlotVersion {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}:{}", self.generation, self.revision)
    }
}
