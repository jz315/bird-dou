use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::{Seat, SeatOrder, SeatSet};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DoublingRound {
    pub order: SeatOrder,
    pub cursor: u8,
    pub doubled: SeatSet,
}

impl DoublingRound {
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

    pub fn validate(&self) -> Result<(), DoublingStateError> {
        if usize::from(self.cursor) > self.order.len() {
            return Err(DoublingStateError::CursorOutOfRange {
                cursor: self.cursor,
                length: self.order.len(),
            });
        }
        if !self.doubled.is_subset(self.acted()) {
            return Err(DoublingStateError::DoubleBeforeActing);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DoublingState {
    Disabled,
    NotStarted,
    InProgress(DoublingRound),
    Resolved {
        eligible: SeatSet,
        doubled: SeatSet,
    },
}

impl DoublingState {
    pub fn doubled(&self) -> SeatSet {
        match self {
            Self::InProgress(round) => round.doubled,
            Self::Resolved { doubled, .. } => *doubled,
            Self::Disabled | Self::NotStarted => SeatSet::empty(),
        }
    }

    pub fn validate(&self) -> Result<(), DoublingStateError> {
        match self {
            Self::InProgress(round) => round.validate(),
            Self::Resolved { eligible, doubled } => {
                if doubled.is_subset(*eligible) {
                    Ok(())
                } else {
                    Err(DoublingStateError::IneligibleDouble)
                }
            }
            Self::Disabled | Self::NotStarted => Ok(()),
        }
    }
}

/// Information-safe doubling view. Pending choices reveal only who has acted.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicDoublingState {
    Disabled,
    NotStarted,
    InProgress {
        eligible: SeatSet,
        acted: SeatSet,
        current_player: Option<Seat>,
    },
    Resolved {
        eligible: SeatSet,
        doubled: SeatSet,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DoublingStateError {
    CursorOutOfRange { cursor: u8, length: usize },
    DoubleBeforeActing,
    IneligibleDouble,
}

impl Display for DoublingStateError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CursorOutOfRange { cursor, length } => write!(
                formatter,
                "doubling cursor {cursor} is outside turn order length {length}"
            ),
            Self::DoubleBeforeActing => {
                write!(formatter, "doubling state marks an unacted player as doubled")
            }
            Self::IneligibleDouble => {
                write!(formatter, "resolved doubling contains an ineligible player")
            }
        }
    }
}

impl Error for DoublingStateError {}
