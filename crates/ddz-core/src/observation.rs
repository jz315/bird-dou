use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::{
    CardPlayState, DoublingState, GameEvent, GameOutcome, LandlordSelectionState, Phase,
    PublicDoublingState, Rank, RankCounts, RankCountsError, RevealState, Role, Seat, SeatMap,
    StakeState,
};

/// Information-set-safe player view. It intentionally has no seed, deck order or private deal plan.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub phase: Phase,
    pub observer: Seat,
    pub role: Role,
    pub current_player: Option<Seat>,
    pub landlord: Option<Seat>,
    pub own_hand: RankCounts,
    /// Other seats are populated only after those players reveal their hands.
    pub revealed_hands: SeatMap<Option<RankCounts>>,
    /// Union of every truly hidden current hand plus hidden bottom cards, if any.
    pub unknown_pool: RankCounts,
    pub cards_left: SeatMap<u8>,
    pub public_bottom_cards: Option<RankCounts>,
    pub reveal: RevealState,
    pub landlord_selection: LandlordSelectionState,
    pub doubling: PublicDoublingState,
    pub stake: StakeState,
    pub card_play: CardPlayState,
    pub history: Vec<GameEvent>,
    pub outcome: Option<GameOutcome>,
}

impl Observation {
    pub fn validate(&self) -> Result<(), ObservationError> {
        if self.revealed_hands[self.observer].is_some() {
            return Err(ObservationError::OwnHandDuplicated);
        }
        if self.role
            != match self.landlord {
                Some(landlord) if landlord == self.observer => Role::Landlord,
                Some(_) => Role::Farmer,
                None => Role::Unassigned,
            }
        {
            return Err(ObservationError::RoleMismatch);
        }
        if self.phase == Phase::Terminal {
            if self.current_player.is_some() || self.outcome.is_none() {
                return Err(ObservationError::InvalidTerminalContract);
            }
        } else if self.outcome.is_some() {
            return Err(ObservationError::OutcomeBeforeTerminal);
        }

        let visible_total: u16 = self
            .revealed_hands
            .iter()
            .filter_map(|(_, hand)| *hand)
            .map(RankCounts::card_count)
            .sum();
        let expected_hidden = self
            .cards_left
            .iter()
            .filter(|(seat, _)| *seat != self.observer && self.revealed_hands[*seat].is_none())
            .map(|(_, count)| u16::from(*count))
            .sum::<u16>()
            + if self.landlord.is_none() && self.public_bottom_cards.is_none() {
                3
            } else {
                0
            };
        if self.unknown_pool.card_count() != expected_hidden {
            return Err(ObservationError::UnknownPoolSize {
                actual: self.unknown_pool.card_count(),
                expected: expected_hidden,
            });
        }

        if visible_total
            + self.unknown_pool.card_count()
            + self.own_hand.card_count()
            < self.cards_left.iter().map(|(_, count)| u16::from(*count)).sum::<u16>()
        {
            return Err(ObservationError::VisibleCardsUnderflow);
        }

        for rank in Rank::ALL {
            let mut known = self.own_hand[rank];
            for (seat, hand) in self.revealed_hands.iter() {
                if seat != self.observer {
                    known = known
                        .checked_add(hand.map_or(0, |value| value[rank]))
                        .ok_or(ObservationError::Counts(
                            RankCountsError::ArithmeticOverflow { rank },
                        ))?;
                }
            }
            known = known
                .checked_add(self.unknown_pool[rank])
                .ok_or(ObservationError::Counts(
                    RankCountsError::ArithmeticOverflow { rank },
                ))?;
            if let Some(bottom) = self.public_bottom_cards {
                if self.landlord.is_none() {
                    known = known
                        .checked_add(bottom[rank])
                        .ok_or(ObservationError::Counts(
                            RankCountsError::ArithmeticOverflow { rank },
                        ))?;
                }
            }
            if known > rank.capacity() {
                return Err(ObservationError::RankOverexposed {
                    rank,
                    visible: known,
                    capacity: rank.capacity(),
                });
            }
        }
        Ok(())
    }

    pub fn public_doubling_from_private(state: &DoublingState) -> PublicDoublingState {
        match state {
            DoublingState::Disabled => PublicDoublingState::Disabled,
            DoublingState::NotStarted => PublicDoublingState::NotStarted,
            DoublingState::InProgress(round) => PublicDoublingState::InProgress {
                eligible: round.eligible(),
                acted: round.acted(),
                current_player: round.current_player(),
            },
            DoublingState::Resolved { eligible, doubled } => PublicDoublingState::Resolved {
                eligible: *eligible,
                doubled: *doubled,
            },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ObservationError {
    OwnHandDuplicated,
    RoleMismatch,
    InvalidTerminalContract,
    OutcomeBeforeTerminal,
    UnknownPoolSize { actual: u16, expected: u16 },
    VisibleCardsUnderflow,
    RankOverexposed {
        rank: Rank,
        visible: u8,
        capacity: u8,
    },
    Counts(RankCountsError),
}

impl Display for ObservationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OwnHandDuplicated => {
                write!(formatter, "observer hand must not be duplicated in revealed_hands")
            }
            Self::RoleMismatch => write!(formatter, "observation role differs from landlord seat"),
            Self::InvalidTerminalContract => {
                write!(formatter, "terminal observation needs no current player and one outcome")
            }
            Self::OutcomeBeforeTerminal => {
                write!(formatter, "non-terminal observation contains an outcome")
            }
            Self::UnknownPoolSize { actual, expected } => write!(
                formatter,
                "unknown pool contains {actual} cards; public capacities require {expected}"
            ),
            Self::VisibleCardsUnderflow => write!(
                formatter,
                "visible and unknown cards cannot account for public cards-left totals"
            ),
            Self::RankOverexposed {
                rank,
                visible,
                capacity,
            } => write!(
                formatter,
                "observation accounts for {visible} cards of {rank:?}; physical capacity is {capacity}"
            ),
            Self::Counts(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for ObservationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Counts(error) => Some(error),
            _ => None,
        }
    }
}
