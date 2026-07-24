mod outcome;
mod transition;

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::game::deal::deal;
use crate::game::{Action, GameError, StepResult};
use crate::{Hand, Move, Rank, Seat, CARDS_PER_PLAYER, PLAYER_COUNT};

pub use outcome::RoundOutcome;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Round {
    pub(super) level: Rank,
    pub(super) hands: [Hand; PLAYER_COUNT],
    pub(super) current_player: Seat,
    pub(super) target: Option<TrickTarget>,
    pub(super) passes_since_play: u8,
    pub(super) finish_order: Vec<Seat>,
    pub(super) outcome: Option<RoundOutcome>,
}

impl Round {
    pub fn new(seed: u64, level: Rank, first_player: Seat) -> Result<Self, GameError> {
        Self::from_deal(level, deal(seed), first_player)
    }

    pub fn from_deal(
        level: Rank,
        hands: [Hand; PLAYER_COUNT],
        first_player: Seat,
    ) -> Result<Self, GameError> {
        if !level.is_standard() {
            return Err(GameError::InvalidLevel(level));
        }
        for seat in Seat::ALL {
            let actual = hands[seat.index()].len();
            if actual != CARDS_PER_PLAYER {
                return Err(GameError::InvalidHandSize {
                    seat,
                    expected: CARDS_PER_PLAYER,
                    actual,
                });
            }
        }
        validate_unique_cards(&hands)?;
        Ok(Self::from_valid_hands(level, hands, first_player))
    }

    pub const fn level(&self) -> Rank {
        self.level
    }

    pub const fn current_player(&self) -> Seat {
        self.current_player
    }

    pub fn hand(&self, seat: Seat) -> &Hand {
        &self.hands[seat.index()]
    }

    pub fn target_move(&self) -> Option<&Move> {
        self.target.as_ref().map(|target| &target.movement)
    }

    pub fn target_player(&self) -> Option<Seat> {
        self.target.as_ref().map(|target| target.player)
    }

    pub fn finish_order(&self) -> &[Seat] {
        &self.finish_order
    }

    pub const fn outcome(&self) -> Option<&RoundOutcome> {
        self.outcome.as_ref()
    }

    pub fn step(&mut self, actor: Seat, action: Action) -> Result<StepResult, GameError> {
        let mut next = self.clone();
        let result = next.apply(actor, action)?;
        *self = next;
        Ok(result)
    }

    fn from_valid_hands(level: Rank, hands: [Hand; PLAYER_COUNT], first_player: Seat) -> Self {
        Self {
            level,
            hands,
            current_player: first_player,
            target: None,
            passes_since_play: 0,
            finish_order: Vec::with_capacity(PLAYER_COUNT),
            outcome: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_hands(
        level: Rank,
        hands: [Hand; PLAYER_COUNT],
        first_player: Seat,
    ) -> Result<Self, GameError> {
        if !level.is_standard() {
            return Err(GameError::InvalidLevel(level));
        }
        validate_unique_cards(&hands)?;
        Ok(Self::from_valid_hands(level, hands, first_player))
    }
}

fn validate_unique_cards(hands: &[Hand; PLAYER_COUNT]) -> Result<(), GameError> {
    let mut seen = BTreeSet::new();
    for hand in hands {
        for card in hand.cards() {
            if !seen.insert(card) {
                return Err(GameError::DuplicatePhysicalCard(card));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(super) struct TrickTarget {
    pub(super) player: Seat,
    pub(super) movement: Move,
}
