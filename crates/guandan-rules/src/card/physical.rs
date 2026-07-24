use serde::{Deserialize, Serialize};

use super::{CardError, Rank, Suit, CARD_COUNT, CARD_FACE_COUNT};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Card(u8);

impl Card {
    pub fn standard(copy: u8, suit: Suit, rank: Rank) -> Result<Self, CardError> {
        if copy >= 2 {
            return Err(CardError::InvalidCopy(copy));
        }
        let rank_index = rank
            .natural_index()
            .ok_or(CardError::SuitAssignedToJoker(rank))?;
        Ok(Self(copy * CARD_FACE_COUNT + rank_index * 4 + suit as u8))
    }

    pub fn joker(copy: u8, rank: Rank) -> Result<Self, CardError> {
        if copy >= 2 {
            return Err(CardError::InvalidCopy(copy));
        }
        let face = match rank {
            Rank::SmallJoker => 52,
            Rank::BigJoker => 53,
            _ => return Err(CardError::MissingSuit(rank)),
        };
        Ok(Self(copy * CARD_FACE_COUNT + face))
    }

    pub fn from_id(id: u8) -> Result<Self, CardError> {
        if usize::from(id) >= CARD_COUNT {
            Err(CardError::InvalidId(id))
        } else {
            Ok(Self(id))
        }
    }

    pub const fn id(self) -> u8 {
        self.0
    }

    pub const fn copy(self) -> u8 {
        self.0 / CARD_FACE_COUNT
    }

    pub const fn rank(self) -> Rank {
        let face = self.0 % CARD_FACE_COUNT;
        if face == 52 {
            Rank::SmallJoker
        } else if face == 53 {
            Rank::BigJoker
        } else {
            Rank::STANDARD[(face / 4) as usize]
        }
    }

    pub const fn suit(self) -> Option<Suit> {
        let face = self.0 % CARD_FACE_COUNT;
        if face >= 52 {
            None
        } else {
            Some(Suit::ALL[(face % 4) as usize])
        }
    }

    pub const fn is_wild(self, level: Rank) -> bool {
        self.rank() as u8 == level as u8 && matches!(self.suit(), Some(Suit::Hearts))
    }
}

impl<'de> Deserialize<'de> for Card {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let id = u8::deserialize(deserializer)?;
        Self::from_id(id).map_err(serde::de::Error::custom)
    }
}

impl From<Card> for u8 {
    fn from(value: Card) -> Self {
        value.id()
    }
}

pub fn all_cards() -> Vec<Card> {
    (0..CARD_COUNT)
        .map(|id| Card::from_id(id as u8).expect("0..108 are valid physical cards"))
        .collect()
}
