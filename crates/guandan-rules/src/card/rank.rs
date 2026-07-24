use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum Suit {
    Clubs = 0,
    Diamonds = 1,
    Hearts = 2,
    Spades = 3,
}

impl Suit {
    pub const ALL: [Self; 4] = [Self::Clubs, Self::Diamonds, Self::Hearts, Self::Spades];
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(u8)]
pub enum Rank {
    #[serde(rename = "2")]
    Two = 0,
    #[serde(rename = "3")]
    Three = 1,
    #[serde(rename = "4")]
    Four = 2,
    #[serde(rename = "5")]
    Five = 3,
    #[serde(rename = "6")]
    Six = 4,
    #[serde(rename = "7")]
    Seven = 5,
    #[serde(rename = "8")]
    Eight = 6,
    #[serde(rename = "9")]
    Nine = 7,
    #[serde(rename = "10")]
    Ten = 8,
    #[serde(rename = "jack")]
    Jack = 9,
    #[serde(rename = "queen")]
    Queen = 10,
    #[serde(rename = "king")]
    King = 11,
    #[serde(rename = "ace")]
    Ace = 12,
    #[serde(rename = "small_joker")]
    SmallJoker = 13,
    #[serde(rename = "big_joker")]
    BigJoker = 14,
}

impl Rank {
    pub const STANDARD: [Self; 13] = [
        Self::Two,
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
    ];

    pub const fn is_standard(self) -> bool {
        (self as u8) <= Self::Ace as u8
    }

    pub const fn natural_index(self) -> Option<u8> {
        if self.is_standard() {
            Some(self as u8)
        } else {
            None
        }
    }

    pub const fn next_level(self) -> Option<Self> {
        match self {
            Self::Two => Some(Self::Three),
            Self::Three => Some(Self::Four),
            Self::Four => Some(Self::Five),
            Self::Five => Some(Self::Six),
            Self::Six => Some(Self::Seven),
            Self::Seven => Some(Self::Eight),
            Self::Eight => Some(Self::Nine),
            Self::Nine => Some(Self::Ten),
            Self::Ten => Some(Self::Jack),
            Self::Jack => Some(Self::Queen),
            Self::Queen => Some(Self::King),
            Self::King => Some(Self::Ace),
            Self::Ace | Self::SmallJoker | Self::BigJoker => None,
        }
    }
}
