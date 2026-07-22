use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::{
    CardId, DeckOrder, RankCounts, RankCountsError, Seat, SeatMap, CARD_COUNT, PLAYER_COUNT,
};

pub const DEAL_ROUNDS: u8 = 17;
const DEALT_CARD_COUNT: usize = 51;

/// Immutable private deal plan. Cards `0..51` are dealt round-robin; the last three are bottom.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DealPlan {
    deck: DeckOrder,
}

impl DealPlan {
    pub const fn new(deck: DeckOrder) -> Self {
        Self { deck }
    }

    pub const fn deck(&self) -> &DeckOrder {
        &self.deck
    }

    pub fn card_for(&self, seat: Seat, round: u8) -> Option<CardId> {
        if round >= DEAL_ROUNDS {
            return None;
        }
        let index = usize::from(round) * PLAYER_COUNT + seat.index();
        self.deck.card(index)
    }

    pub fn bottom_cards(&self) -> [CardId; 3] {
        std::array::from_fn(|offset| {
            self.deck
                .card(DEALT_CARD_COUNT + offset)
                .expect("validated deck always contains three bottom cards")
        })
    }

    pub fn bottom_counts(&self) -> RankCounts {
        RankCounts::from_cards(self.bottom_cards())
            .expect("a validated physical deck always produces valid rank counts")
    }

    pub fn hand_prefix(&self, seat: Seat, rounds: u8) -> Result<RankCounts, DealStateError> {
        if rounds > DEAL_ROUNDS {
            return Err(DealStateError::TooManyRounds { rounds });
        }
        RankCounts::from_cards((0..rounds).map(|round| {
            self.card_for(seat, round)
                .expect("validated round and seat identify a dealt card")
        }))
        .map_err(DealStateError::Counts)
    }

    pub fn final_hands(&self) -> SeatMap<RankCounts> {
        SeatMap::from_fn(|seat| {
            self.hand_prefix(seat, DEAL_ROUNDS)
                .expect("all final deal prefixes are valid")
        })
    }

    pub fn full_deck_counts(&self) -> RankCounts {
        RankCounts::from_cards(self.deck.as_slice().iter().copied())
            .expect("validated deck contains a physically valid pack")
    }

    pub fn validate(&self) -> Result<(), DealStateError> {
        if self.deck.as_slice().len() != CARD_COUNT {
            return Err(DealStateError::InvalidPlan);
        }
        Ok(())
    }
}

/// Current attempt and incremental dealing progress.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DealState {
    pub attempt: u32,
    pub plan: DealPlan,
    pub rounds_dealt: u8,
}

impl DealState {
    pub fn new(attempt: u32, plan: DealPlan) -> Self {
        Self {
            attempt,
            plan,
            rounds_dealt: 0,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.rounds_dealt == DEAL_ROUNDS
    }

    pub fn cards_received(&self, _seat: Seat) -> u8 {
        self.rounds_dealt
    }

    pub fn validate(&self) -> Result<(), DealStateError> {
        self.plan.validate()?;
        if self.rounds_dealt > DEAL_ROUNDS {
            return Err(DealStateError::TooManyRounds {
                rounds: self.rounds_dealt,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DealStateError {
    InvalidPlan,
    TooManyRounds { rounds: u8 },
    Counts(RankCountsError),
}

impl Display for DealStateError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPlan => write!(formatter, "deal plan does not contain a complete deck"),
            Self::TooManyRounds { rounds } => {
                write!(formatter, "deal has completed {rounds} rounds; maximum is {DEAL_ROUNDS}")
            }
            Self::Counts(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for DealStateError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Counts(error) => Some(error),
            Self::InvalidPlan | Self::TooManyRounds { .. } => None,
        }
    }
}
