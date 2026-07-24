use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::game::RoundOutcome;
use crate::{Card, DetectError, Move, Rank, Seat};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "cards", rename_all = "snake_case")]
pub enum Action {
    Pass,
    Play(Vec<Card>),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StepResult {
    pub played: Option<Move>,
    pub trick_ended: bool,
    pub next_player: Option<Seat>,
    pub round_outcome: Option<RoundOutcome>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GameError {
    InvalidLevel(Rank),
    RoundComplete,
    MatchComplete,
    NotCurrentPlayer {
        expected: Seat,
        actual: Seat,
    },
    MustLead,
    DuplicatePhysicalCard(Card),
    InvalidHandSize {
        seat: Seat,
        expected: usize,
        actual: usize,
    },
    InvalidFinishOrder,
    CardNotOwned(Card),
    InvalidMove(DetectError),
    DoesNotBeat,
}

impl Display for GameError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLevel(rank) => write!(formatter, "{rank:?} cannot be a level rank"),
            Self::RoundComplete => formatter.write_str("the round is already complete"),
            Self::MatchComplete => formatter.write_str("the match already has a winner"),
            Self::NotCurrentPlayer { expected, actual } => write!(
                formatter,
                "seat {} cannot act; seat {} is current",
                actual.index(),
                expected.index()
            ),
            Self::MustLead => formatter.write_str("the trick leader cannot pass"),
            Self::DuplicatePhysicalCard(card) => {
                write!(formatter, "physical card {} occurs in two hands", card.id())
            }
            Self::InvalidHandSize {
                seat,
                expected,
                actual,
            } => write!(
                formatter,
                "seat {} has {actual} cards; a fresh deal requires {expected}",
                seat.index()
            ),
            Self::InvalidFinishOrder => {
                formatter.write_str("finish order must contain every seat exactly once")
            }
            Self::CardNotOwned(card) => {
                write!(formatter, "current player does not own card {}", card.id())
            }
            Self::InvalidMove(error) => write!(formatter, "invalid move: {error}"),
            Self::DoesNotBeat => {
                formatter.write_str("the move does not beat the current trick target")
            }
        }
    }
}

impl Error for GameError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidMove(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DetectError> for GameError {
    fn from(value: DetectError) -> Self {
        Self::InvalidMove(value)
    }
}
