use std::error::Error;
use std::fmt::{Display, Formatter};

use ddz_core::{
    GameAction, GameStateError, ObservationError, Phase, RankCountsError, Seat, SeatOrderError,
    StakeError,
};

use super::invariants::RuleStateError;
use crate::{
    DealError, DetectMoveError, GenerateMovesError, RuleConfigError, RuleProfile, SettlementError,
};

#[derive(Debug)]
pub enum GameError {
    RuleConfig(RuleConfigError),
    Deal(DealError),
    State(GameStateError),
    RuleState(RuleStateError),
    Observation(ObservationError),
    GenerateMoves(GenerateMovesError),
    DetectMove(DetectMoveError),
    Settlement(SettlementError),
    RankCounts(RankCountsError),
    SeatOrder(SeatOrderError),
    Stake(StakeError),
    WrongProfile { expected: RuleProfile, actual: RuleProfile },
    Terminal,
    AutomaticPhase { phase: Phase },
    NoCurrentPlayer { phase: Phase },
    NotCurrentPlayer { expected: Seat, actual: Seat },
    WrongActionForPhase { phase: Phase, action: GameAction },
    IllegalAction { actor: Seat, action: GameAction },
    AttemptOverflow,
    HistoryTooLong,
    StakeExponentOverflow,
    AutomaticTransitionLimit,
    UndoTokenMismatch,
    InvalidInternalState(&'static str),
}

impl Display for GameError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuleConfig(error) => Display::fmt(error, formatter),
            Self::Deal(error) => Display::fmt(error, formatter),
            Self::State(error) => Display::fmt(error, formatter),
            Self::RuleState(error) => Display::fmt(error, formatter),
            Self::Observation(error) => Display::fmt(error, formatter),
            Self::GenerateMoves(error) => Display::fmt(error, formatter),
            Self::DetectMove(error) => Display::fmt(error, formatter),
            Self::Settlement(error) => Display::fmt(error, formatter),
            Self::RankCounts(error) => Display::fmt(error, formatter),
            Self::SeatOrder(error) => Display::fmt(error, formatter),
            Self::Stake(error) => Display::fmt(error, formatter),
            Self::WrongProfile { expected, actual } => write!(
                formatter,
                "game constructor requires profile {expected:?}, received {actual:?}"
            ),
            Self::Terminal => formatter.write_str("terminal games accept no actions"),
            Self::AutomaticPhase { phase } => {
                write!(formatter, "phase {phase:?} has no player decision")
            }
            Self::NoCurrentPlayer { phase } => {
                write!(formatter, "phase {phase:?} has no current player")
            }
            Self::NotCurrentPlayer { expected, actual } => write!(
                formatter,
                "seat {actual} attempted to act; current player is seat {expected}"
            ),
            Self::WrongActionForPhase { phase, action } => {
                write!(formatter, "action {action:?} is invalid during phase {phase:?}")
            }
            Self::IllegalAction { actor, action } => {
                write!(formatter, "seat {actor} selected illegal action {action:?}")
            }
            Self::AttemptOverflow => formatter.write_str("redeal attempt counter overflowed"),
            Self::HistoryTooLong => {
                formatter.write_str("public event sequence no longer fits in u32")
            }
            Self::StakeExponentOverflow => formatter.write_str("stake exponent overflowed u8"),
            Self::AutomaticTransitionLimit => formatter.write_str(
                "automatic transition loop exceeded its deterministic safety limit",
            ),
            Self::UndoTokenMismatch => formatter.write_str(
                "undo token belongs to a different rule configuration or match seed",
            ),
            Self::InvalidInternalState(message) => formatter.write_str(message),
        }
    }
}

impl Error for GameError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RuleConfig(error) => Some(error),
            Self::Deal(error) => Some(error),
            Self::State(error) => Some(error),
            Self::RuleState(error) => Some(error),
            Self::Observation(error) => Some(error),
            Self::GenerateMoves(error) => Some(error),
            Self::DetectMove(error) => Some(error),
            Self::Settlement(error) => Some(error),
            Self::RankCounts(error) => Some(error),
            Self::SeatOrder(error) => Some(error),
            Self::Stake(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum GameRestoreError {
    RuleConfig(RuleConfigError),
    Deal(DealError),
    State(GameStateError),
    RuleState(RuleStateError),
    RuleConfigIdMismatch { expected: u32, actual: u32 },
    DealPlanMismatch { attempt: u32 },
    ProfileStateMismatch { profile: RuleProfile },
    BaseUnitMismatch { expected: u32, actual: u32 },
    Replay(GameError),
    ReplayMissingLandlord,
    ReplayMismatch,
}

impl Display for GameRestoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuleConfig(error) => Display::fmt(error, formatter),
            Self::Deal(error) => Display::fmt(error, formatter),
            Self::State(error) => Display::fmt(error, formatter),
            Self::RuleState(error) => Display::fmt(error, formatter),
            Self::RuleConfigIdMismatch { expected, actual } => write!(
                formatter,
                "state rule_config_id {actual} differs from supplied rules {expected}"
            ),
            Self::DealPlanMismatch { attempt } => write!(
                formatter,
                "state deal plan for attempt {attempt} does not match the supplied match seed"
            ),
            Self::ProfileStateMismatch { profile } => write!(
                formatter,
                "state flow is incompatible with rule profile {profile:?}"
            ),
            Self::BaseUnitMismatch { expected, actual } => write!(
                formatter,
                "state base unit {actual} differs from rule configuration {expected}"
            ),
            Self::Replay(error) => write!(formatter, "state replay failed: {error}"),
            Self::ReplayMissingLandlord => {
                formatter.write_str("post-bid state replay requires a landlord")
            }
            Self::ReplayMismatch => {
                formatter.write_str("serialized state differs from deterministic event replay")
            }
        }
    }
}

impl Error for GameRestoreError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RuleConfig(error) => Some(error),
            Self::Deal(error) => Some(error),
            Self::State(error) => Some(error),
            Self::RuleState(error) => Some(error),
            Self::Replay(error) => Some(error),
            Self::RuleConfigIdMismatch { .. }
            | Self::DealPlanMismatch { .. }
            | Self::ProfileStateMismatch { .. }
            | Self::BaseUnitMismatch { .. }
            | Self::ReplayMissingLandlord
            | Self::ReplayMismatch => None,
        }
    }
}
