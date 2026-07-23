use std::error::Error;
use std::fmt::{Display, Formatter};

use ddz_core::{
    DoublingState, GameState, LandlordSelectionState, Phase, RevealState, SpringKind,
};

use crate::settlement::{self, SettlementError};
use crate::{RuleConfig, RuleProfile};

pub(crate) fn validate(state: &GameState, rules: &RuleConfig) -> Result<(), RuleStateError> {
    if state.rule_config_id != rules.rule_config_id {
        return Err(RuleStateError::RuleConfigIdMismatch {
            expected: rules.rule_config_id,
            actual: state.rule_config_id,
        });
    }
    if state.stake.base_unit != rules.settlement.base_unit {
        return Err(RuleStateError::BaseUnitMismatch {
            expected: rules.settlement.base_unit,
            actual: state.stake.base_unit,
        });
    }
    if state.stake.reveal_factor != state.reveal.maximum_factor {
        return Err(RuleStateError::RevealFactorMismatch {
            expected: state.reveal.maximum_factor,
            actual: state.stake.reveal_factor,
        });
    }
    let expected_robs = state.landlord_selection.successful_robs();
    if state.stake.rob_exponent != expected_robs {
        return Err(RuleStateError::RobExponentMismatch {
            expected: expected_robs,
            actual: state.stake.rob_exponent,
        });
    }
    let expected_bombs = u8::try_from(state.card_play.bomb_count)
        .map_err(|_| RuleStateError::BombCountTooLarge {
            count: state.card_play.bomb_count,
        })?;
    if state.stake.bomb_exponent != expected_bombs {
        return Err(RuleStateError::BombExponentMismatch {
            expected: expected_bombs,
            actual: state.stake.bomb_exponent,
        });
    }
    if state.phase != Phase::Terminal && state.stake.spring != SpringKind::None {
        return Err(RuleStateError::SpringBeforeTerminal);
    }

    match rules.profile {
        RuleProfile::DouzeroPostBid => validate_douzero(state)?,
        RuleProfile::HuanleClassic => validate_huanle(state)?,
    }

    if state.phase == Phase::Terminal {
        settlement::validate_terminal_outcome(state, rules)
            .map_err(RuleStateError::Settlement)?;
    }
    Ok(())
}

fn validate_douzero(state: &GameState) -> Result<(), RuleStateError> {
    if !matches!(state.phase, Phase::CardPlay | Phase::Terminal) {
        return Err(RuleStateError::ProfilePhaseMismatch {
            profile: RuleProfile::DouzeroPostBid,
            phase: state.phase,
        });
    }
    if !matches!(&state.landlord_selection, LandlordSelectionState::PostBid { .. }) {
        return Err(RuleStateError::PostBidLandlordStateRequired);
    }
    if state.reveal != RevealState::hidden() {
        return Err(RuleStateError::RevealMustBeDisabled);
    }
    if state.doubling != DoublingState::Disabled {
        return Err(RuleStateError::DoublingStateMismatch {
            phase: state.phase,
        });
    }
    Ok(())
}

fn validate_huanle(state: &GameState) -> Result<(), RuleStateError> {
    if matches!(&state.landlord_selection, LandlordSelectionState::PostBid { .. }) {
        return Err(RuleStateError::PostBidStateInHuanle);
    }
    let doubling_matches = match state.phase {
        Phase::PreDeal
        | Phase::Dealing
        | Phase::Calling
        | Phase::Robbing
        | Phase::BottomReveal
        | Phase::PostBottomReveal => state.doubling == DoublingState::NotStarted,
        Phase::Doubling => matches!(&state.doubling, DoublingState::InProgress(_)),
        Phase::CardPlay | Phase::Terminal => {
            matches!(&state.doubling, DoublingState::Resolved { .. })
        }
    };
    if !doubling_matches {
        return Err(RuleStateError::DoublingStateMismatch {
            phase: state.phase,
        });
    }
    Ok(())
}

#[derive(Debug)]
pub enum RuleStateError {
    RuleConfigIdMismatch { expected: u32, actual: u32 },
    BaseUnitMismatch { expected: u32, actual: u32 },
    RevealFactorMismatch { expected: u32, actual: u32 },
    RobExponentMismatch { expected: u8, actual: u8 },
    BombCountTooLarge { count: u16 },
    BombExponentMismatch { expected: u8, actual: u8 },
    SpringBeforeTerminal,
    ProfilePhaseMismatch { profile: RuleProfile, phase: Phase },
    PostBidLandlordStateRequired,
    PostBidStateInHuanle,
    RevealMustBeDisabled,
    DoublingStateMismatch { phase: Phase },
    Settlement(SettlementError),
}

impl Display for RuleStateError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuleConfigIdMismatch { expected, actual } => write!(
                formatter,
                "state rule_config_id {actual} differs from supplied rules {expected}"
            ),
            Self::BaseUnitMismatch { expected, actual } => write!(
                formatter,
                "state base unit {actual} differs from supplied rules {expected}"
            ),
            Self::RevealFactorMismatch { expected, actual } => write!(
                formatter,
                "stake reveal factor {actual} differs from reveal state {expected}"
            ),
            Self::RobExponentMismatch { expected, actual } => write!(
                formatter,
                "stake rob exponent {actual} differs from landlord state {expected}"
            ),
            Self::BombCountTooLarge { count } => {
                write!(formatter, "bomb count {count} no longer fits in the stake exponent")
            }
            Self::BombExponentMismatch { expected, actual } => write!(
                formatter,
                "stake bomb exponent {actual} differs from card-play bomb count {expected}"
            ),
            Self::SpringBeforeTerminal => {
                formatter.write_str("spring factor is set before terminal settlement")
            }
            Self::ProfilePhaseMismatch { profile, phase } => write!(
                formatter,
                "phase {phase:?} is incompatible with rule profile {profile:?}"
            ),
            Self::PostBidLandlordStateRequired => {
                formatter.write_str("DouZero profile requires a post-bid landlord state")
            }
            Self::PostBidStateInHuanle => {
                formatter.write_str("Huanle profile cannot contain a post-bid landlord state")
            }
            Self::RevealMustBeDisabled => {
                formatter.write_str("DouZero profile requires an untouched reveal state")
            }
            Self::DoublingStateMismatch { phase } => write!(
                formatter,
                "doubling state is inconsistent with phase {phase:?}"
            ),
            Self::Settlement(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for RuleStateError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Settlement(error) => Some(error),
            _ => None,
        }
    }
}
