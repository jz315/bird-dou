use std::error::Error;
use std::fmt::{Display, Formatter};

use super::{Card, Rank};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CardError {
    InvalidId(u8),
    InvalidCopy(u8),
    SuitAssignedToJoker(Rank),
    MissingSuit(Rank),
}

impl Display for CardError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidId(id) => write!(formatter, "card id {id} is outside 0..108"),
            Self::InvalidCopy(copy) => write!(formatter, "deck copy {copy} is outside 0..2"),
            Self::SuitAssignedToJoker(rank) => {
                write!(formatter, "joker rank {rank:?} cannot have a suit")
            }
            Self::MissingSuit(rank) => write!(formatter, "standard rank {rank:?} requires a suit"),
        }
    }
}

impl Error for CardError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandError {
    DuplicateCard(Card),
}

impl Display for HandError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateCard(card) => {
                write!(formatter, "physical card {} is duplicated", card.id())
            }
        }
    }
}

impl Error for HandError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SeatError(u8);

impl SeatError {
    pub(super) const fn new(value: u8) -> Self {
        Self(value)
    }
}

impl Display for SeatError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "seat {} is outside 0..4", self.0)
    }
}

impl Error for SeatError {}
