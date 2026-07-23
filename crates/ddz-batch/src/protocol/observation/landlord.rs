use ddz_core::{GameState, LandlordSelectionState};

use crate::protocol::codes::{self, NO_SEAT};
use crate::BatchError;

/// Calling and robbing state normalized into fixed columns.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedLandlord {
    /// Selection-state code: post-bid/not-started/calling/robbing/resolved.
    pub kind: Vec<u8>,
    /// Original first caller or `-1`.
    pub first_player: Vec<i8>,
    /// Calling seat that first claimed landlord or `-1`.
    pub caller: Vec<i8>,
    /// Current provisional candidate or `-1`.
    pub candidate: Vec<i8>,
    /// Resolved landlord or `-1`.
    pub resolved_landlord: Vec<i8>,
    /// State-local current decision maker or `-1`.
    pub current_player: Vec<i8>,
    /// Calling/robbing acted bit mask.
    pub acted_mask: Vec<u8>,
    /// Calling decline bit mask.
    pub declined_mask: Vec<u8>,
    /// Rob eligibility bit mask.
    pub eligible_mask: Vec<u8>,
    /// Number of successful rob actions.
    pub successful_robs: Vec<u8>,
}

impl PackedLandlord {
    pub(crate) fn with_capacity(batch: usize) -> Self {
        Self {
            kind: Vec::with_capacity(batch),
            first_player: Vec::with_capacity(batch),
            caller: Vec::with_capacity(batch),
            candidate: Vec::with_capacity(batch),
            resolved_landlord: Vec::with_capacity(batch),
            current_player: Vec::with_capacity(batch),
            acted_mask: Vec::with_capacity(batch),
            declined_mask: Vec::with_capacity(batch),
            eligible_mask: Vec::with_capacity(batch),
            successful_robs: Vec::with_capacity(batch),
        }
    }

    pub(crate) fn push(&mut self, state: &GameState) {
        self.kind
            .push(codes::landlord_state(&state.landlord_selection));
        let mut first = NO_SEAT;
        let mut caller = NO_SEAT;
        let mut candidate = NO_SEAT;
        let mut resolved = NO_SEAT;
        let mut current = NO_SEAT;
        let mut acted = 0;
        let mut declined = 0;
        let mut eligible = 0;
        let mut successful = 0;

        match &state.landlord_selection {
            LandlordSelectionState::PostBid { landlord } => {
                resolved = codes::seat(*landlord);
            }
            LandlordSelectionState::NotStarted { first_player } => {
                first = codes::seat(*first_player);
            }
            LandlordSelectionState::Calling(value) => {
                first = codes::seat(value.first_player);
                current = codes::seat(value.current_player);
                acted = value.acted.bits();
                declined = value.declined.bits();
            }
            LandlordSelectionState::Robbing(value) => {
                caller = codes::seat(value.caller);
                candidate = codes::seat(value.candidate);
                current = value.current_player().map_or(NO_SEAT, |seat| codes::seat(seat));
                acted = value.acted().bits();
                eligible = value.eligible().bits();
                successful = value.successful_robs;
            }
            LandlordSelectionState::Resolved(value) => {
                caller = codes::seat(value.caller);
                candidate = codes::seat(value.landlord);
                resolved = codes::seat(value.landlord);
                successful = value.successful_robs;
            }
        }

        self.first_player.push(first);
        self.caller.push(caller);
        self.candidate.push(candidate);
        self.resolved_landlord.push(resolved);
        self.current_player.push(current);
        self.acted_mask.push(acted);
        self.declined_mask.push(declined);
        self.eligible_mask.push(eligible);
        self.successful_robs.push(successful);
    }

    pub(crate) fn validate(&self, expected: usize) -> Result<(), BatchError> {
        let lengths = [
            self.kind.len(),
            self.first_player.len(),
            self.caller.len(),
            self.candidate.len(),
            self.resolved_landlord.len(),
            self.current_player.len(),
            self.acted_mask.len(),
            self.declined_mask.len(),
            self.eligible_mask.len(),
            self.successful_robs.len(),
        ];
        if lengths.into_iter().any(|length| length != expected) {
            return Err(BatchError::InternalInvariant(
                "packed landlord-selection columns have inconsistent lengths",
            ));
        }
        Ok(())
    }
}
