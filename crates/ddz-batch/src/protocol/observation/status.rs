use crate::protocol::codes::{self, NO_SEAT, NO_U8};
use crate::BatchError;

use super::ObservationRow;

/// Per-slot phase, observer, and stale-response protection metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedStatus {
    /// Whether an information-set observation was requested for this row.
    pub valid: Vec<u8>,
    /// Explicit phase code.
    pub phase: Vec<u8>,
    /// Observation owner or `-1` for an intentionally invalid terminal row.
    pub observer: Vec<i8>,
    /// Role code or `255` when `valid == 0`.
    pub role: Vec<u8>,
    /// Current player or `-1`.
    pub current_player: Vec<i8>,
    /// Resolved landlord or `-1`.
    pub landlord: Vec<i8>,
    /// Terminal flag.
    pub terminal: Vec<u8>,
    /// Current same-match deal-attempt index.
    pub attempt: Vec<u32>,
    /// Number of completed three-card deal rounds.
    pub rounds_dealt: Vec<u8>,
    /// Slot generation.
    pub generation: Vec<u64>,
    /// Slot revision.
    pub revision: Vec<u64>,
}

impl PackedStatus {
    pub(crate) fn with_capacity(batch: usize) -> Self {
        Self {
            valid: Vec::with_capacity(batch),
            phase: Vec::with_capacity(batch),
            observer: Vec::with_capacity(batch),
            role: Vec::with_capacity(batch),
            current_player: Vec::with_capacity(batch),
            landlord: Vec::with_capacity(batch),
            terminal: Vec::with_capacity(batch),
            attempt: Vec::with_capacity(batch),
            rounds_dealt: Vec::with_capacity(batch),
            generation: Vec::with_capacity(batch),
            revision: Vec::with_capacity(batch),
        }
    }

    pub(crate) fn push(&mut self, row: &ObservationRow<'_>) {
        let observer = row.observation.map(|value| value.observer);
        self.valid.push(u8::from(observer.is_some()));
        self.phase.push(codes::phase(row.state.phase));
        self.observer.push(observer.map_or(NO_SEAT, codes::seat));
        self.role.push(
            row.observation
                .map_or(NO_U8, |value| codes::role(value.role)),
        );
        self.current_player
            .push(row.state.current_player.map_or(NO_SEAT, codes::seat));
        self.landlord
            .push(row.state.landlord().map_or(NO_SEAT, codes::seat));
        self.terminal.push(u8::from(row.state.is_terminal()));
        self.attempt.push(row.state.deal.attempt);
        self.rounds_dealt.push(row.state.deal.rounds_dealt);
        self.generation.push(row.version.generation);
        self.revision.push(row.version.revision);
    }

    pub(crate) fn validate(&self, expected: usize) -> Result<(), BatchError> {
        let lengths = [
            self.valid.len(),
            self.phase.len(),
            self.observer.len(),
            self.role.len(),
            self.current_player.len(),
            self.landlord.len(),
            self.terminal.len(),
            self.attempt.len(),
            self.rounds_dealt.len(),
            self.generation.len(),
            self.revision.len(),
        ];
        if lengths.into_iter().any(|length| length != expected) {
            return Err(BatchError::InternalInvariant(
                "packed observation status columns have inconsistent lengths",
            ));
        }
        Ok(())
    }
}


