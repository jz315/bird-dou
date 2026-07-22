use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::{Seat, SeatMap};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RevealTiming {
    BeforeDeal,
    DuringDeal { cards_received: u8 },
    AfterBottom,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevealInfo {
    pub timing: RevealTiming,
    pub factor: u32,
    pub sequence: u32,
}

impl RevealInfo {
    pub fn validate(self) -> Result<(), RevealStateError> {
        if self.factor < 2 {
            return Err(RevealStateError::InvalidFactor {
                factor: self.factor,
            });
        }
        if let RevealTiming::DuringDeal { cards_received } = self.timing {
            if cards_received > 17 {
                return Err(RevealStateError::InvalidCardsReceived { cards_received });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevealState {
    pub by_seat: SeatMap<Option<RevealInfo>>,
    pub first_revealer: Option<Seat>,
    pub maximum_factor: u32,
}

impl RevealState {
    pub fn hidden() -> Self {
        Self {
            by_seat: SeatMap::default(),
            first_revealer: None,
            maximum_factor: 1,
        }
    }

    pub fn is_revealed(&self, seat: Seat) -> bool {
        self.by_seat[seat].is_some()
    }

    pub fn validate(&self) -> Result<(), RevealStateError> {
        let mut maximum = 1;
        let mut first: Option<(Seat, u32)> = None;
        for (seat, info) in self.by_seat.iter() {
            if let Some(info) = info {
                info.validate()?;
                maximum = maximum.max(info.factor);
                if first.is_none_or(|(_, sequence)| info.sequence < sequence) {
                    first = Some((seat, info.sequence));
                }
            }
        }

        let expected_first = first.map(|(seat, _)| seat);
        if self.first_revealer != expected_first {
            return Err(RevealStateError::FirstRevealerMismatch {
                declared: self.first_revealer,
                expected: expected_first,
            });
        }
        if self.maximum_factor != maximum {
            return Err(RevealStateError::MaximumFactorMismatch {
                declared: self.maximum_factor,
                expected: maximum,
            });
        }
        Ok(())
    }
}

impl Default for RevealState {
    fn default() -> Self {
        Self::hidden()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RevealStateError {
    InvalidFactor {
        factor: u32,
    },
    InvalidCardsReceived {
        cards_received: u8,
    },
    FirstRevealerMismatch {
        declared: Option<Seat>,
        expected: Option<Seat>,
    },
    MaximumFactorMismatch {
        declared: u32,
        expected: u32,
    },
}

impl Display for RevealStateError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFactor { factor } => {
                write!(formatter, "reveal factor {factor} must be at least two")
            }
            Self::InvalidCardsReceived { cards_received } => write!(
                formatter,
                "during-deal reveal reports {cards_received} received cards; maximum is 17"
            ),
            Self::FirstRevealerMismatch { declared, expected } => write!(
                formatter,
                "first revealer {declared:?} differs from earliest reveal {expected:?}"
            ),
            Self::MaximumFactorMismatch { declared, expected } => write!(
                formatter,
                "maximum reveal factor {declared} differs from computed factor {expected}"
            ),
        }
    }
}

impl Error for RevealStateError {}
