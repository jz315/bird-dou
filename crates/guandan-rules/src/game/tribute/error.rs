use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::{Card, Rank, Seat};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TributeError {
    InvalidLevel(Rank),
    InvalidHandSize { seat: Seat, actual: usize },
    DuplicatePhysicalCard(Card),
    WrongOfferCount { expected: usize, actual: usize },
    WrongReturnCount { expected: usize, actual: usize },
    UnexpectedSeat(Seat),
    DuplicateSeat(Seat),
    CardNotOwned { seat: Seat, card: Card },
    WildcardOffer(Card),
    OfferIsNotHighest { seat: Seat, card: Card },
    EqualOfferChoiceRequired,
    InvalidEqualOfferChoice(Seat),
    ReturnAboveTen(Card),
}

impl Display for TributeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLevel(rank) => write!(formatter, "{rank:?} cannot be a level rank"),
            Self::InvalidHandSize { seat, actual } => write!(
                formatter,
                "seat {} has {actual} cards; tribute requires a fresh 27-card deal",
                seat.index()
            ),
            Self::DuplicatePhysicalCard(card) => {
                write!(formatter, "physical card {} occurs in two hands", card.id())
            }
            Self::WrongOfferCount { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected} tribute offers, received {actual}"
                )
            }
            Self::WrongReturnCount { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected} return cards, received {actual}"
                )
            }
            Self::UnexpectedSeat(seat) => {
                write!(
                    formatter,
                    "seat {} is not part of this tribute step",
                    seat.index()
                )
            }
            Self::DuplicateSeat(seat) => {
                write!(formatter, "seat {} submitted twice", seat.index())
            }
            Self::CardNotOwned { seat, card } => write!(
                formatter,
                "seat {} does not own physical card {}",
                seat.index(),
                card.id()
            ),
            Self::WildcardOffer(card) => write!(
                formatter,
                "heart level card {} cannot be offered as tribute",
                card.id()
            ),
            Self::OfferIsNotHighest { seat, card } => write!(
                formatter,
                "card {} is not seat {}'s highest eligible tribute",
                card.id(),
                seat.index()
            ),
            Self::EqualOfferChoiceRequired => {
                formatter.write_str("equal double tribute requires the winners' head choice")
            }
            Self::InvalidEqualOfferChoice(seat) => write!(
                formatter,
                "seat {} did not submit either tied tribute",
                seat.index()
            ),
            Self::ReturnAboveTen(card) => {
                write!(formatter, "return card {} has a rank above ten", card.id())
            }
        }
    }
}

impl Error for TributeError {}
