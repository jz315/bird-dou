//! Terminal spring detection, pairwise transfer calculation, and training rewards.

use std::error::Error;
use std::fmt::{Display, Formatter};

use ddz_core::{
    DoublingState, GameOutcome, GameState, GameStateError, Phase, Seat, SeatMap, SpringKind,
    StakeError,
};

use crate::{RewardMode, RuleConfig, RuleConfigError};

pub fn settle_game(
    state: &mut GameState,
    winner: Seat,
    rules: &RuleConfig,
) -> Result<(), SettlementError> {
    rules.validate().map_err(SettlementError::RuleConfig)?;
    if state.phase != Phase::CardPlay {
        return Err(SettlementError::WrongPhase { phase: state.phase });
    }
    let outcome = expected_outcome(state, winner, rules)?;
    state.stake.spring = outcome.spring;
    state.phase = Phase::Terminal;
    state.current_player = None;
    state.outcome = Some(outcome);
    state.validate().map_err(SettlementError::State)
}

pub fn terminal_reward(
    state: &GameState,
    seat: Seat,
    rules: &RuleConfig,
) -> Result<i64, SettlementError> {
    rules.validate().map_err(SettlementError::RuleConfig)?;
    let outcome = state.outcome.as_ref().ok_or(SettlementError::OutcomeMissing)?;
    let won = if seat == outcome.landlord {
        outcome.winner == outcome.landlord
    } else {
        outcome.winner != outcome.landlord
    };
    let sign = if won { 1_i64 } else { -1_i64 };
    match rules.reward_mode {
        RewardMode::WinPercentage => Ok(sign),
        RewardMode::AverageDifferencePoints => {
            let magnitude = 1_i64
                .checked_shl(u32::from(state.card_play.bomb_count))
                .ok_or(SettlementError::RewardOverflow)?;
            sign.checked_mul(magnitude)
                .ok_or(SettlementError::RewardOverflow)
        }
        RewardMode::LogAverageDifferencePoints => sign
            .checked_mul(i64::from(state.card_play.bomb_count) + 1)
            .ok_or(SettlementError::RewardOverflow),
        RewardMode::RawScore => Ok(outcome.payoff[seat]),
    }
}

pub(crate) fn validate_terminal_outcome(
    state: &GameState,
    rules: &RuleConfig,
) -> Result<(), SettlementError> {
    let actual = state.outcome.as_ref().ok_or(SettlementError::OutcomeMissing)?;
    let expected = expected_outcome(state, actual.winner, rules)?;
    if *actual != expected {
        return Err(SettlementError::OutcomeMismatch);
    }
    if state.stake.spring != actual.spring {
        return Err(SettlementError::SpringMismatch {
            expected: actual.spring,
            actual: state.stake.spring,
        });
    }
    Ok(())
}

fn expected_outcome(
    state: &GameState,
    winner: Seat,
    rules: &RuleConfig,
) -> Result<GameOutcome, SettlementError> {
    if !state.hands[winner].is_empty() {
        return Err(SettlementError::WinnerStillHasCards {
            winner,
            cards: state.hands[winner].card_count(),
        });
    }
    if !matches!(
        &state.doubling,
        DoublingState::Disabled | DoublingState::Resolved { .. }
    ) {
        return Err(SettlementError::UnresolvedDoubling);
    }

    let landlord = state.landlord().ok_or(SettlementError::LandlordMissing)?;
    let spring = determine_spring(state, winner, landlord, rules);
    let mut stake = state.stake;
    stake.spring = spring;
    let common = stake.common_stake().map_err(SettlementError::Stake)?;
    let doubled = state.doubling.doubled();
    let landlord_factor = if doubled.contains(landlord) { 2_u64 } else { 1 };

    let mut payoff = SeatMap::new([0_i64; 3]);
    let landlord_won = winner == landlord;
    for farmer in Seat::ALL.into_iter().filter(|seat| *seat != landlord) {
        let farmer_factor = if doubled.contains(farmer) { 2_u64 } else { 1 };
        let uncapped = common
            .checked_mul(landlord_factor)
            .and_then(|value| value.checked_mul(farmer_factor))
            .ok_or(SettlementError::ScoreOverflow)?;
        let transfer = rules
            .settlement
            .pair_score_cap
            .map_or(uncapped, |cap| uncapped.min(cap));
        let transfer = i64::try_from(transfer).map_err(|_| SettlementError::ScoreOverflow)?;
        if landlord_won {
            payoff[farmer] = -transfer;
            payoff[landlord] = payoff[landlord]
                .checked_add(transfer)
                .ok_or(SettlementError::ScoreOverflow)?;
        } else {
            payoff[farmer] = transfer;
            payoff[landlord] = payoff[landlord]
                .checked_sub(transfer)
                .ok_or(SettlementError::ScoreOverflow)?;
        }
    }

    Ok(GameOutcome {
        winner,
        landlord,
        spring,
        payoff,
    })
}

fn determine_spring(
    state: &GameState,
    winner: Seat,
    landlord: Seat,
    rules: &RuleConfig,
) -> SpringKind {
    let spring = rules.settlement.spring;
    if winner == landlord
        && spring.landlord_spring_enabled
        && Seat::ALL
            .into_iter()
            .filter(|seat| *seat != landlord)
            .all(|farmer| state.card_play.non_pass_plays[farmer] == 0)
    {
        SpringKind::LandlordSpring
    } else if winner != landlord
        && spring.farmer_spring_enabled
        && state.card_play.non_pass_plays[landlord] == 1
    {
        SpringKind::FarmerSpring
    } else {
        SpringKind::None
    }
}

#[derive(Debug)]
pub enum SettlementError {
    RuleConfig(RuleConfigError),
    State(GameStateError),
    WrongPhase { phase: Phase },
    WinnerStillHasCards { winner: Seat, cards: u16 },
    UnresolvedDoubling,
    LandlordMissing,
    OutcomeMissing,
    OutcomeMismatch,
    SpringMismatch { expected: SpringKind, actual: SpringKind },
    Stake(StakeError),
    ScoreOverflow,
    RewardOverflow,
}

impl Display for SettlementError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuleConfig(error) => Display::fmt(error, formatter),
            Self::State(error) => Display::fmt(error, formatter),
            Self::WrongPhase { phase } => {
                write!(formatter, "settlement requires card_play, received {phase:?}")
            }
            Self::WinnerStillHasCards { winner, cards } => write!(
                formatter,
                "seat {winner} was declared winner with {cards} cards remaining"
            ),
            Self::UnresolvedDoubling => {
                formatter.write_str("settlement requires disabled or resolved doubling")
            }
            Self::LandlordMissing => {
                formatter.write_str("cannot settle before landlord resolution")
            }
            Self::OutcomeMissing => formatter.write_str("terminal reward requires an outcome"),
            Self::OutcomeMismatch => {
                formatter.write_str("stored terminal outcome differs from deterministic settlement")
            }
            Self::SpringMismatch { expected, actual } => write!(
                formatter,
                "stake spring {actual:?} differs from terminal outcome {expected:?}"
            ),
            Self::Stake(error) => Display::fmt(error, formatter),
            Self::ScoreOverflow => formatter.write_str("pairwise terminal score overflowed"),
            Self::RewardOverflow => formatter.write_str("learner reward overflowed"),
        }
    }
}

impl Error for SettlementError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RuleConfig(error) => Some(error),
            Self::State(error) => Some(error),
            Self::Stake(error) => Some(error),
            _ => None,
        }
    }
}
