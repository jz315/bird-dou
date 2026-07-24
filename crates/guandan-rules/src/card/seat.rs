use serde::{Deserialize, Serialize};

use super::{SeatError, PLAYER_COUNT};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct Seat(u8);

impl Seat {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);
    pub const TWO: Self = Self(2);
    pub const THREE: Self = Self(3);
    pub const ALL: [Self; 4] = [Self::ZERO, Self::ONE, Self::TWO, Self::THREE];

    pub fn new(value: u8) -> Result<Self, SeatError> {
        if usize::from(value) < PLAYER_COUNT {
            Ok(Self(value))
        } else {
            Err(SeatError::new(value))
        }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[must_use]
    pub const fn next(self) -> Self {
        Self((self.0 + 1) % PLAYER_COUNT as u8)
    }

    #[must_use]
    pub const fn partner(self) -> Self {
        Self((self.0 + 2) % PLAYER_COUNT as u8)
    }

    pub const fn team(self) -> Team {
        if self.0 % 2 == 0 {
            Team::Zero
        } else {
            Team::One
        }
    }
}

impl TryFrom<u8> for Seat {
    type Error = SeatError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<Seat> for u8 {
    fn from(value: Seat) -> Self {
        value.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Team {
    Zero,
    One,
}

impl Team {
    pub const fn index(self) -> usize {
        match self {
            Self::Zero => 0,
            Self::One => 1,
        }
    }
}
