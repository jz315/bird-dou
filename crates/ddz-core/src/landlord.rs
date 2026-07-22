use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::{Seat, SeatOrder, SeatSet};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallingState {
    pub first_player: Seat,
    pub current_player: Seat,
    pub acted: SeatSet,
    pub declined: SeatSet,
}

impl CallingState {
    pub fn validate(&self) -> Result<(), LandlordStateError> {
        if !self.declined.is_subset(self.acted) {
            return Err(LandlordStateError::DeclinedWithoutActing);
        }
        if self.acted.contains(self.current_player) && self.acted != SeatSet::all() {
            return Err(LandlordStateError::CurrentCallerAlreadyActed {
                seat: self.current_player,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobbingState {
    pub caller: Seat,
    pub candidate: Seat,
    pub order: SeatOrder,
    pub cursor: u8,
    pub successful_robs: u8,
}

impl RobbingState {
    pub fn current_player(&self) -> Option<Seat> {
        self.order.get(usize::from(self.cursor))
    }

    pub fn acted(&self) -> SeatSet {
        let mut result = SeatSet::empty();
        for seat in self.order.as_slice().iter().take(usize::from(self.cursor)) {
            result.insert(*seat);
        }
        result
    }

    pub fn eligible(&self) -> SeatSet {
        self.order.as_set()
    }

    pub fn validate(&self) -> Result<(), LandlordStateError> {
        if usize::from(self.cursor) > self.order.len() {
            return Err(LandlordStateError::RobCursorOutOfRange {
                cursor: self.cursor,
                length: self.order.len(),
            });
        }
        if self.order.is_empty() && self.cursor != 0 {
            return Err(LandlordStateError::RobCursorOutOfRange {
                cursor: self.cursor,
                length: 0,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedLandlord {
    pub landlord: Seat,
    pub caller: Seat,
    pub successful_robs: u8,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LandlordSelectionState {
    /// Used by the exact DouZero post-bid profile.
    PostBid { landlord: Seat },
    /// Huanle game before dealing has completed.
    NotStarted { first_player: Seat },
    Calling(CallingState),
    Robbing(RobbingState),
    Resolved(ResolvedLandlord),
}

impl LandlordSelectionState {
    pub fn landlord(&self) -> Option<Seat> {
        match self {
            Self::PostBid { landlord } => Some(*landlord),
            Self::Resolved(resolved) => Some(resolved.landlord),
            Self::NotStarted { .. } | Self::Calling(_) | Self::Robbing(_) => None,
        }
    }

    pub fn successful_robs(&self) -> u8 {
        match self {
            Self::Resolved(resolved) => resolved.successful_robs,
            Self::Robbing(state) => state.successful_robs,
            Self::PostBid { .. } | Self::NotStarted { .. } | Self::Calling(_) => 0,
        }
    }

    pub fn validate(&self) -> Result<(), LandlordStateError> {
        match self {
            Self::Calling(state) => state.validate(),
            Self::Robbing(state) => state.validate(),
            Self::PostBid { .. } | Self::NotStarted { .. } | Self::Resolved(_) => Ok(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LandlordStateError {
    DeclinedWithoutActing,
    CurrentCallerAlreadyActed { seat: Seat },
    RobCursorOutOfRange { cursor: u8, length: usize },
}

impl Display for LandlordStateError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeclinedWithoutActing => {
                write!(formatter, "a player declined calling without having acted")
            }
            Self::CurrentCallerAlreadyActed { seat } => {
                write!(formatter, "current calling seat {seat} has already acted")
            }
            Self::RobCursorOutOfRange { cursor, length } => write!(
                formatter,
                "rob cursor {cursor} is outside turn order length {length}"
            ),
        }
    }
}

impl Error for LandlordStateError {}
