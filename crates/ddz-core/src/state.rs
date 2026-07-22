use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::{
    DealState, DealStateError, DoublingState, DoublingStateError, GameEvent,
    LandlordSelectionState, LandlordStateError, Move, MoveKind, Phase, Rank, RankCounts,
    RankCountsError, RevealState, RevealStateError, Role, Seat, SeatMap, SpringKind, StakeError,
    StakeState, EMPTY_RANK_COUNTS,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CardPlayState {
    pub played_cards: SeatMap<RankCounts>,
    pub last_non_pass: Option<Move>,
    pub last_non_pass_player: Option<Seat>,
    pub consecutive_passes: u8,
    pub non_pass_plays: SeatMap<u16>,
    pub bomb_count: u16,
}

impl CardPlayState {
    pub fn empty() -> Self {
        Self {
            played_cards: SeatMap::new([EMPTY_RANK_COUNTS; 3]),
            last_non_pass: None,
            last_non_pass_player: None,
            consecutive_passes: 0,
            non_pass_plays: SeatMap::new([0; 3]),
            bomb_count: 0,
        }
    }

    pub fn validate(&self) -> Result<(), CardPlayStateError> {
        match (self.last_non_pass, self.last_non_pass_player) {
            (None, None) => {
                if self.consecutive_passes != 0 {
                    return Err(CardPlayStateError::PassesWithoutTarget {
                        passes: self.consecutive_passes,
                    });
                }
            }
            (Some(movement), Some(_)) => {
                if movement.kind() == MoveKind::Pass {
                    return Err(CardPlayStateError::PassAsTarget);
                }
                if self.consecutive_passes > 1 {
                    return Err(CardPlayStateError::TooManyPasses {
                        passes: self.consecutive_passes,
                    });
                }
            }
            _ => return Err(CardPlayStateError::TargetPlayerMismatch),
        }

        let counted_bombs: u16 = self
            .non_pass_plays
            .iter()
            .map(|(_, count)| *count)
            .sum();
        if self.bomb_count > counted_bombs {
            return Err(CardPlayStateError::BombCountExceedsPlays {
                bombs: self.bomb_count,
                plays: counted_bombs,
            });
        }
        Ok(())
    }
}

impl Default for CardPlayState {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CardPlayStateError {
    PassesWithoutTarget { passes: u8 },
    PassAsTarget,
    TooManyPasses { passes: u8 },
    TargetPlayerMismatch,
    BombCountExceedsPlays { bombs: u16, plays: u16 },
}

impl Display for CardPlayStateError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PassesWithoutTarget { passes } => {
                write!(formatter, "{passes} consecutive passes exist without an active target")
            }
            Self::PassAsTarget => write!(formatter, "pass cannot be the active target move"),
            Self::TooManyPasses { passes } => write!(
                formatter,
                "active target has {passes} consecutive passes; it should clear after two"
            ),
            Self::TargetPlayerMismatch => {
                write!(formatter, "target move and target player must both be present or absent")
            }
            Self::BombCountExceedsPlays { bombs, plays } => write!(
                formatter,
                "bomb count {bombs} exceeds total non-pass plays {plays}"
            ),
        }
    }
}

impl Error for CardPlayStateError {}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameOutcome {
    pub winner: Seat,
    pub landlord: Seat,
    pub spring: SpringKind,
    pub payoff: SeatMap<i64>,
}

impl GameOutcome {
    pub fn validate(&self) -> Result<(), GameStateError> {
        let total: i128 = self.payoff.iter().map(|(_, value)| i128::from(*value)).sum();
        if total != 0 {
            return Err(GameStateError::NonZeroSumOutcome { total });
        }
        Ok(())
    }
}

/// Authoritative private state. Rule transitions live in `ddz-rules`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameState {
    pub rule_config_id: u32,
    pub phase: Phase,
    pub current_player: Option<Seat>,
    pub deal: DealState,
    pub hands: SeatMap<RankCounts>,
    pub reveal: RevealState,
    pub landlord_selection: LandlordSelectionState,
    pub doubling: DoublingState,
    pub stake: StakeState,
    pub card_play: CardPlayState,
    pub history: Vec<GameEvent>,
    pub outcome: Option<GameOutcome>,
}

impl GameState {
    pub fn landlord(&self) -> Option<Seat> {
        self.landlord_selection.landlord()
    }

    pub fn role_of(&self, seat: Seat) -> Role {
        match self.landlord() {
            Some(landlord) if landlord == seat => Role::Landlord,
            Some(_) => Role::Farmer,
            None => Role::Unassigned,
        }
    }

    pub fn cards_left(&self) -> SeatMap<u8> {
        self.hands.map(|_, hand| {
            u8::try_from(hand.card_count()).expect("one DouDizhu hand always fits in u8")
        })
    }

    pub fn is_terminal(&self) -> bool {
        self.phase == Phase::Terminal
    }

    pub fn validate(&self) -> Result<(), GameStateError> {
        if self.rule_config_id == 0 {
            return Err(GameStateError::ZeroRuleConfigId);
        }
        self.deal.validate().map_err(GameStateError::Deal)?;
        self.reveal.validate().map_err(GameStateError::Reveal)?;
        self.landlord_selection
            .validate()
            .map_err(GameStateError::Landlord)?;
        self.doubling
            .validate()
            .map_err(GameStateError::Doubling)?;
        self.stake.validate().map_err(GameStateError::Stake)?;
        self.card_play
            .validate()
            .map_err(GameStateError::CardPlay)?;

        self.validate_phase_contract()?;
        self.validate_history()?;
        self.validate_cards()?;
        if let Some(outcome) = &self.outcome {
            outcome.validate()?;
        }
        Ok(())
    }

    fn validate_phase_contract(&self) -> Result<(), GameStateError> {
        if self.phase == Phase::Terminal {
            if self.current_player.is_some() || self.outcome.is_none() {
                return Err(GameStateError::InvalidTerminalContract);
            }
        } else if self.outcome.is_some() {
            return Err(GameStateError::OutcomeBeforeTerminal);
        }

        let selection_matches = match (&self.phase, &self.landlord_selection) {
            (Phase::PreDeal | Phase::Dealing, LandlordSelectionState::NotStarted { .. }) => true,
            (Phase::Calling, LandlordSelectionState::Calling(_)) => true,
            (Phase::Robbing, LandlordSelectionState::Robbing(_)) => true,
            (
                Phase::BottomReveal
                | Phase::PostBottomReveal
                | Phase::Doubling
                | Phase::CardPlay
                | Phase::Terminal,
                LandlordSelectionState::Resolved(_) | LandlordSelectionState::PostBid { .. },
            ) => true,
            _ => false,
        };
        if !selection_matches {
            return Err(GameStateError::PhaseLandlordStateMismatch {
                phase: self.phase,
            });
        }

        if matches!(self.phase, Phase::Calling | Phase::Robbing)
            && self.deal.rounds_dealt != crate::DEAL_ROUNDS
        {
            return Err(GameStateError::BiddingBeforeDealComplete {
                rounds: self.deal.rounds_dealt,
            });
        }

        Ok(())
    }

    fn validate_history(&self) -> Result<(), GameStateError> {
        for (index, event) in self.history.iter().enumerate() {
            let expected = u32::try_from(index).map_err(|_| GameStateError::HistoryTooLong)?;
            if event.sequence != expected {
                return Err(GameStateError::HistorySequence {
                    expected,
                    actual: event.sequence,
                });
            }
            if event.attempt > self.deal.attempt {
                return Err(GameStateError::FutureAttemptEvent {
                    event_attempt: event.attempt,
                    current_attempt: self.deal.attempt,
                });
            }
        }
        Ok(())
    }

    fn validate_cards(&self) -> Result<(), GameStateError> {
        let landlord = self.landlord();
        if landlord.is_none() {
            for seat in Seat::ALL {
                let expected = self
                    .deal
                    .plan
                    .hand_prefix(seat, self.deal.rounds_dealt)
                    .map_err(GameStateError::Deal)?;
                if self.hands[seat] != expected {
                    return Err(GameStateError::DealtHandMismatch { seat });
                }
                if !self.card_play.played_cards[seat].is_empty() {
                    return Err(GameStateError::PlayedBeforeLandlord { seat });
                }
            }
            return Ok(());
        }

        let landlord = landlord.expect("checked above");
        if self.deal.rounds_dealt != crate::DEAL_ROUNDS {
            return Err(GameStateError::LandlordBeforeDealComplete {
                rounds: self.deal.rounds_dealt,
            });
        }

        let final_hands = self.deal.plan.final_hands();
        let bottom = self.deal.plan.bottom_counts();
        for seat in Seat::ALL {
            let expected = if seat == landlord {
                final_hands[seat]
                    .checked_add(bottom)
                    .map_err(GameStateError::Counts)?
            } else {
                final_hands[seat]
            };
            let reconstructed = self.hands[seat]
                .checked_add(self.card_play.played_cards[seat])
                .map_err(GameStateError::Counts)?;
            if reconstructed != expected {
                return Err(GameStateError::ResolvedHandMismatch { seat });
            }
        }

        let mut total = RankCounts::empty();
        for seat in Seat::ALL {
            total = total
                .checked_add(self.hands[seat])
                .map_err(GameStateError::Counts)?;
            total = total
                .checked_add(self.card_play.played_cards[seat])
                .map_err(GameStateError::Counts)?;
        }
        for rank in Rank::ALL {
            if total[rank] != rank.capacity() {
                return Err(GameStateError::DeckConservation {
                    rank,
                    actual: total[rank],
                    expected: rank.capacity(),
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GameStateError {
    ZeroRuleConfigId,
    Deal(DealStateError),
    Reveal(RevealStateError),
    Landlord(LandlordStateError),
    Doubling(DoublingStateError),
    Stake(StakeError),
    CardPlay(CardPlayStateError),
    Counts(RankCountsError),
    InvalidTerminalContract,
    OutcomeBeforeTerminal,
    PhaseLandlordStateMismatch {
        phase: Phase,
    },
    BiddingBeforeDealComplete {
        rounds: u8,
    },
    HistoryTooLong,
    HistorySequence {
        expected: u32,
        actual: u32,
    },
    FutureAttemptEvent {
        event_attempt: u32,
        current_attempt: u32,
    },
    DealtHandMismatch {
        seat: Seat,
    },
    PlayedBeforeLandlord {
        seat: Seat,
    },
    LandlordBeforeDealComplete {
        rounds: u8,
    },
    ResolvedHandMismatch {
        seat: Seat,
    },
    DeckConservation {
        rank: Rank,
        actual: u8,
        expected: u8,
    },
    NonZeroSumOutcome {
        total: i128,
    },
}

impl Display for GameStateError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroRuleConfigId => write!(formatter, "rule_config_id must be non-zero"),
            Self::Deal(error) => Display::fmt(error, formatter),
            Self::Reveal(error) => Display::fmt(error, formatter),
            Self::Landlord(error) => Display::fmt(error, formatter),
            Self::Doubling(error) => Display::fmt(error, formatter),
            Self::Stake(error) => Display::fmt(error, formatter),
            Self::CardPlay(error) => Display::fmt(error, formatter),
            Self::Counts(error) => Display::fmt(error, formatter),
            Self::InvalidTerminalContract => {
                write!(formatter, "terminal state needs no current player and one outcome")
            }
            Self::OutcomeBeforeTerminal => write!(formatter, "non-terminal state contains an outcome"),
            Self::PhaseLandlordStateMismatch { phase } => write!(
                formatter,
                "landlord-selection state is incompatible with phase {phase:?}"
            ),
            Self::BiddingBeforeDealComplete { rounds } => write!(
                formatter,
                "bidding started after {rounds} deal rounds; expected 17"
            ),
            Self::HistoryTooLong => write!(formatter, "event history exceeds u32 sequence range"),
            Self::HistorySequence { expected, actual } => write!(
                formatter,
                "event sequence {actual} is discontinuous; expected {expected}"
            ),
            Self::FutureAttemptEvent {
                event_attempt,
                current_attempt,
            } => write!(
                formatter,
                "event belongs to future attempt {event_attempt}; current attempt is {current_attempt}"
            ),
            Self::DealtHandMismatch { seat } => {
                write!(formatter, "seat {seat} hand differs from dealt prefix")
            }
            Self::PlayedBeforeLandlord { seat } => {
                write!(formatter, "seat {seat} has played cards before landlord resolution")
            }
            Self::LandlordBeforeDealComplete { rounds } => write!(
                formatter,
                "landlord exists after only {rounds} deal rounds"
            ),
            Self::ResolvedHandMismatch { seat } => write!(
                formatter,
                "seat {seat} current plus played cards differs from its resolved initial hand"
            ),
            Self::DeckConservation {
                rank,
                actual,
                expected,
            } => write!(
                formatter,
                "resolved state has {actual} cards of {rank:?}; expected {expected}"
            ),
            Self::NonZeroSumOutcome { total } => {
                write!(formatter, "terminal payoff sums to {total}, not zero")
            }
        }
    }
}

impl Error for GameStateError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Deal(error) => Some(error),
            Self::Reveal(error) => Some(error),
            Self::Landlord(error) => Some(error),
            Self::Doubling(error) => Some(error),
            Self::Stake(error) => Some(error),
            Self::CardPlay(error) => Some(error),
            Self::Counts(error) => Some(error),
            _ => None,
        }
    }
}
