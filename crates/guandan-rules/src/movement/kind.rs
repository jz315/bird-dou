use serde::{Deserialize, Serialize};

use crate::{Card, Rank, Suit};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Move {
    cards: Vec<Card>,
    kind: MoveKind,
}

impl Move {
    pub(crate) fn new(mut cards: Vec<Card>, kind: MoveKind) -> Self {
        cards.sort_unstable();
        Self { cards, kind }
    }

    pub fn cards(&self) -> &[Card] {
        &self.cards
    }

    pub const fn kind(&self) -> &MoveKind {
        &self.kind
    }

    pub fn len(&self) -> usize {
        self.cards.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MoveKind {
    Single {
        rank: Rank,
    },
    Pair {
        rank: Rank,
    },
    Triple {
        rank: Rank,
    },
    FullHouse {
        triple_rank: Rank,
    },
    Straight {
        sequence: u8,
        high: Rank,
    },
    PairStraight {
        sequence: u8,
        high: Rank,
    },
    TripleStraight {
        sequence: u8,
        high: Rank,
    },
    Bomb {
        rank: Rank,
        size: u8,
    },
    StraightFlush {
        suit: Suit,
        sequence: u8,
        high: Rank,
    },
    FourJokers,
}
