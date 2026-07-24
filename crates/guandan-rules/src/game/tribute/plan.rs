use serde::{Deserialize, Serialize};

use crate::game::tribute::validation::{validate_deal, validate_offers};
use crate::game::tribute::{TributeAssignment, TributeError, TributeTransfer};
use crate::game::RoundOutcome;
use crate::movement::strength::rank_strength;
use crate::{Card, Hand, Rank, Seat, PLAYER_COUNT};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TributeMode {
    Single,
    Double,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TributePlan {
    level: Rank,
    hands: [Hand; PLAYER_COUNT],
    mode: TributeMode,
    head: Seat,
    second: Seat,
    givers: Vec<Seat>,
    resisted: bool,
}

impl TributePlan {
    pub fn from_previous_round(
        outcome: &RoundOutcome,
        hands: [Hand; PLAYER_COUNT],
        level: Rank,
    ) -> Result<Self, TributeError> {
        if !level.is_standard() {
            return Err(TributeError::InvalidLevel(level));
        }
        validate_deal(&hands)?;
        let order = outcome.finish_order();
        let mode = if outcome.level_advance() == 3 {
            TributeMode::Double
        } else {
            TributeMode::Single
        };
        let givers = match mode {
            TributeMode::Single => vec![order[3]],
            TributeMode::Double => vec![order[2], order[3]],
        };
        let big_jokers = givers
            .iter()
            .flat_map(|seat| hands[seat.index()].cards())
            .filter(|card| card.rank() == Rank::BigJoker)
            .count();
        Ok(Self {
            level,
            hands,
            mode,
            head: order[0],
            second: order[1],
            givers,
            resisted: big_jokers == 2,
        })
    }

    pub const fn mode(&self) -> TributeMode {
        self.mode
    }

    pub const fn is_resisted(&self) -> bool {
        self.resisted
    }

    pub fn required_givers(&self) -> &[Seat] {
        if self.resisted {
            &[]
        } else {
            &self.givers
        }
    }

    pub fn assign_offers(
        self,
        offers: &[(Seat, Card)],
        equal_offer_to_head: Option<Seat>,
    ) -> Result<TributeAssignment, TributeError> {
        let expected = self.required_givers().len();
        if offers.len() != expected {
            return Err(TributeError::WrongOfferCount {
                expected,
                actual: offers.len(),
            });
        }
        if self.resisted {
            return Ok(TributeAssignment {
                hands: self.hands,
                transfers: Vec::new(),
                opening_player: self.head,
            });
        }
        validate_offers(&self.hands, &self.givers, offers, self.level)?;

        match self.mode {
            TributeMode::Single => Ok(self.assign_single(offers[0])),
            TributeMode::Double => self.assign_double(offers, equal_offer_to_head),
        }
    }

    fn assign_single(self, offer: (Seat, Card)) -> TributeAssignment {
        TributeAssignment {
            hands: self.hands,
            transfers: vec![TributeTransfer {
                from: offer.0,
                to: self.head,
                card: offer.1,
            }],
            opening_player: offer.0,
        }
    }

    fn assign_double(
        self,
        offers: &[(Seat, Card)],
        equal_offer_to_head: Option<Seat>,
    ) -> Result<TributeAssignment, TributeError> {
        let left_strength = rank_strength(offers[0].1.rank(), self.level);
        let right_strength = rank_strength(offers[1].1.rank(), self.level);
        let head_giver = match left_strength.cmp(&right_strength) {
            std::cmp::Ordering::Greater => offers[0].0,
            std::cmp::Ordering::Less => offers[1].0,
            std::cmp::Ordering::Equal => {
                let choice = equal_offer_to_head.ok_or(TributeError::EqualOfferChoiceRequired)?;
                if !self.givers.contains(&choice) {
                    return Err(TributeError::InvalidEqualOfferChoice(choice));
                }
                choice
            }
        };
        let opening_player = if left_strength == right_strength {
            self.head.next()
        } else {
            head_giver
        };
        let transfers = offers
            .iter()
            .map(|(giver, card)| TributeTransfer {
                from: *giver,
                to: if *giver == head_giver {
                    self.head
                } else {
                    self.second
                },
                card: *card,
            })
            .collect();
        Ok(TributeAssignment {
            hands: self.hands,
            transfers,
            opening_player,
        })
    }
}
