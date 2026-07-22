use std::error::Error;
use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut};

use serde::{Deserialize, Serialize};

pub const PLAYER_COUNT: usize = 3;
pub const CARD_COUNT: usize = 54;
pub const RANK_COUNT: usize = 15;
pub const SMALL_JOKER: Rank = Rank::SmallJoker;
pub const BIG_JOKER: Rank = Rank::BigJoker;

/// Ordered ranks `3..A, 2, small joker, big joker`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum Rank {
    Three = 0,
    Four = 1,
    Five = 2,
    Six = 3,
    Seven = 4,
    Eight = 5,
    Nine = 6,
    Ten = 7,
    Jack = 8,
    Queen = 9,
    King = 10,
    Ace = 11,
    Two = 12,
    SmallJoker = 13,
    BigJoker = 14,
}

impl Rank {
    pub const ALL: [Self; RANK_COUNT] = [
        Self::Three,
        Self::Four,
        Self::Five,
        Self::Six,
        Self::Seven,
        Self::Eight,
        Self::Nine,
        Self::Ten,
        Self::Jack,
        Self::Queen,
        Self::King,
        Self::Ace,
        Self::Two,
        Self::SmallJoker,
        Self::BigJoker,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn value(self) -> u8 {
        self as u8
    }

    pub const fn capacity(self) -> u8 {
        match self {
            Self::SmallJoker | Self::BigJoker => 1,
            _ => 4,
        }
    }

    pub const fn is_straight_eligible(self) -> bool {
        self.value() <= Rank::Ace.value()
    }
}

impl TryFrom<u8> for Rank {
    type Error = RankCountsError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Rank::ALL
            .get(usize::from(value))
            .copied()
            .ok_or(RankCountsError::InvalidRank { value })
    }
}

/// Valid physical card ID in the inclusive range `0..=53`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(try_from = "u8", into = "u8")]
pub struct CardId(u8);

impl CardId {
    pub const fn new(value: u8) -> Result<Self, CardIdError> {
        if value < CARD_COUNT as u8 {
            Ok(Self(value))
        } else {
            Err(CardIdError { value })
        }
    }

    pub const fn value(self) -> u8 {
        self.0
    }

    pub const fn rank(self) -> Rank {
        match self.0 {
            0..=51 => Rank::ALL[(self.0 / 4) as usize],
            52 => Rank::SmallJoker,
            53 => Rank::BigJoker,
            _ => unreachable!(),
        }
    }
}

impl TryFrom<u8> for CardId {
    type Error = CardIdError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<CardId> for u8 {
    fn from(value: CardId) -> Self {
        value.value()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CardIdError {
    pub value: u8,
}

impl Display for CardIdError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "card ID {} is outside 0..=53", self.value)
    }
}

impl Error for CardIdError {}

/// Physical-card multiplicity by ordered rank.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "[u8; RANK_COUNT]", into = "[u8; RANK_COUNT]")]
pub struct RankCounts([u8; RANK_COUNT]);

pub const EMPTY_RANK_COUNTS: RankCounts = RankCounts::empty();

impl RankCounts {
    pub const fn empty() -> Self {
        Self([0; RANK_COUNT])
    }

    pub fn new(values: [u8; RANK_COUNT]) -> Result<Self, RankCountsError> {
        for rank in Rank::ALL {
            let count = values[rank.index()];
            let maximum = rank.capacity();
            if count > maximum {
                return Err(RankCountsError::TooManyCards {
                    rank,
                    count,
                    maximum,
                });
            }
        }
        Ok(Self(values))
    }

    pub const fn as_array(&self) -> &[u8; RANK_COUNT] {
        &self.0
    }

    pub fn as_mut_array(&mut self) -> &mut [u8; RANK_COUNT] {
        &mut self.0
    }

    pub const fn get(self, rank: Rank) -> u8 {
        self.0[rank.index()]
    }

    pub fn set(&mut self, rank: Rank, count: u8) -> Result<(), RankCountsError> {
        let maximum = rank.capacity();
        if count > maximum {
            return Err(RankCountsError::TooManyCards {
                rank,
                count,
                maximum,
            });
        }
        self.0[rank.index()] = count;
        Ok(())
    }

    pub fn add_card(&mut self, card: CardId) -> Result<(), RankCountsError> {
        let rank = card.rank();
        let current = self[rank];
        self.set(
            rank,
            current
                .checked_add(1)
                .ok_or(RankCountsError::ArithmeticOverflow { rank })?,
        )
    }

    pub fn remove_card(&mut self, card: CardId) -> Result<(), RankCountsError> {
        let rank = card.rank();
        let current = self[rank];
        if current == 0 {
            return Err(RankCountsError::MissingCard { rank });
        }
        self.0[rank.index()] = current - 1;
        Ok(())
    }

    pub fn card_count(self) -> u16 {
        self.0.iter().map(|value| u16::from(*value)).sum()
    }

    pub const fn is_empty(self) -> bool {
        let mut index = 0;
        while index < RANK_COUNT {
            if self.0[index] != 0 {
                return false;
            }
            index += 1;
        }
        true
    }

    pub fn checked_add(self, other: Self) -> Result<Self, RankCountsError> {
        let mut result = self;
        for rank in Rank::ALL {
            let value = self[rank]
                .checked_add(other[rank])
                .ok_or(RankCountsError::ArithmeticOverflow { rank })?;
            result.set(rank, value)?;
        }
        Ok(result)
    }

    pub fn checked_sub(self, other: Self) -> Result<Self, RankCountsError> {
        let mut result = self;
        for rank in Rank::ALL {
            let available = self[rank];
            let required = other[rank];
            if required > available {
                return Err(RankCountsError::InsufficientCards {
                    rank,
                    available,
                    required,
                });
            }
            result.0[rank.index()] = available - required;
        }
        Ok(result)
    }

    pub fn contains(self, other: Self) -> bool {
        Rank::ALL
            .into_iter()
            .all(|rank| self[rank] >= other[rank])
    }

    pub fn from_cards(cards: impl IntoIterator<Item = CardId>) -> Result<Self, RankCountsError> {
        let mut counts = Self::empty();
        for card in cards {
            counts.add_card(card)?;
        }
        Ok(counts)
    }

    pub fn iter(self) -> impl ExactSizeIterator<Item = (Rank, u8)> {
        Rank::ALL.into_iter().map(move |rank| (rank, self[rank]))
    }
}

impl TryFrom<[u8; RANK_COUNT]> for RankCounts {
    type Error = RankCountsError;

    fn try_from(value: [u8; RANK_COUNT]) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<RankCounts> for [u8; RANK_COUNT] {
    fn from(value: RankCounts) -> Self {
        value.0
    }
}

impl Index<Rank> for RankCounts {
    type Output = u8;

    fn index(&self, rank: Rank) -> &Self::Output {
        &self.0[rank.index()]
    }
}

impl IndexMut<Rank> for RankCounts {
    fn index_mut(&mut self, rank: Rank) -> &mut Self::Output {
        &mut self.0[rank.index()]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RankCountsError {
    InvalidRank {
        value: u8,
    },
    TooManyCards {
        rank: Rank,
        count: u8,
        maximum: u8,
    },
    MissingCard {
        rank: Rank,
    },
    InsufficientCards {
        rank: Rank,
        available: u8,
        required: u8,
    },
    ArithmeticOverflow {
        rank: Rank,
    },
}

impl Display for RankCountsError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRank { value } => write!(formatter, "rank {value} is outside 0..=14"),
            Self::TooManyCards {
                rank,
                count,
                maximum,
            } => write!(
                formatter,
                "rank {rank:?} has {count} cards; physical maximum is {maximum}"
            ),
            Self::MissingCard { rank } => write!(formatter, "hand has no {rank:?} to remove"),
            Self::InsufficientCards {
                rank,
                available,
                required,
            } => write!(
                formatter,
                "rank {rank:?} has {available} cards but {required} are required"
            ),
            Self::ArithmeticOverflow { rank } => {
                write!(formatter, "rank-count arithmetic overflowed at {rank:?}")
            }
        }
    }
}

impl Error for RankCountsError {}

/// Complete permutation of all 54 physical cards.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Vec<CardId>", into = "Vec<CardId>")]
pub struct DeckOrder(Vec<CardId>);

impl DeckOrder {
    pub fn identity() -> Self {
        Self(
            (0_u8..CARD_COUNT as u8)
                .map(|value| CardId::new(value).expect("identity deck uses valid card IDs"))
                .collect(),
        )
    }

    pub fn as_slice(&self) -> &[CardId] {
        &self.0
    }

    pub fn card(&self, index: usize) -> Option<CardId> {
        self.0.get(index).copied()
    }
}

impl TryFrom<Vec<CardId>> for DeckOrder {
    type Error = DeckOrderError;

    fn try_from(value: Vec<CardId>) -> Result<Self, Self::Error> {
        if value.len() != CARD_COUNT {
            return Err(DeckOrderError::WrongLength { actual: value.len() });
        }
        let mut seen = [false; CARD_COUNT];
        for card in &value {
            let index = usize::from(card.value());
            if seen[index] {
                return Err(DeckOrderError::DuplicateCard { card: *card });
            }
            seen[index] = true;
        }
        Ok(Self(value))
    }
}

impl From<DeckOrder> for Vec<CardId> {
    fn from(value: DeckOrder) -> Self {
        value.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeckOrderError {
    WrongLength { actual: usize },
    DuplicateCard { card: CardId },
}

impl Display for DeckOrderError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongLength { actual } => {
                write!(formatter, "deck has {actual} cards; expected {CARD_COUNT}")
            }
            Self::DuplicateCard { card } => {
                write!(formatter, "deck repeats physical card {}", card.value())
            }
        }
    }
}

impl Error for DeckOrderError {}
