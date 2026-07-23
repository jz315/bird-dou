mod patterns;

use std::error::Error;
use std::fmt::{Display, Formatter};

use ddz_core::{Move, MoveError, MoveKind, Rank, RankCounts};

use crate::{AttachmentMultiplicity, RuleConfig, RuleConfigError};

pub use patterns::detect_move;

pub fn detect_move_with_rules(
    cards: RankCounts,
    rules: &RuleConfig,
) -> Result<Move, DetectMoveError> {
    rules.validate().map_err(DetectMoveError::RuleConfig)?;
    let movement = detect_move(cards)?;
    validate_move_for_rules(movement, rules)?;
    Ok(movement)
}

pub fn validate_move_for_rules(
    movement: Move,
    rules: &RuleConfig,
) -> Result<(), DetectMoveError> {
    rules.validate().map_err(DetectMoveError::RuleConfig)?;
    let canonical = detect_move(movement.cards())?;
    if canonical != movement {
        return Err(DetectMoveError::NonCanonical {
            supplied: movement,
            canonical,
        });
    }

    match movement.kind() {
        MoveKind::FourWithTwoSingles => {
            if !rules.moves.four_with_two.two_singles_enabled {
                return Err(DetectMoveError::MoveDisabled {
                    kind: movement.kind(),
                });
            }
            validate_attachment_multiplicity(
                movement,
                rules.moves.four_with_two.single_attachments,
                false,
            )?;
        }
        MoveKind::FourWithTwoPairs => {
            if !rules.moves.four_with_two.two_pairs_enabled {
                return Err(DetectMoveError::MoveDisabled {
                    kind: movement.kind(),
                });
            }
            validate_attachment_multiplicity(
                movement,
                rules.moves.four_with_two.pair_attachments,
                true,
            )?;
        }
        MoveKind::AirplaneWithSingles => validate_attachment_multiplicity(
            movement,
            rules.moves.airplane.single_attachments,
            false,
        )?,
        MoveKind::AirplaneWithPairs => validate_attachment_multiplicity(
            movement,
            rules.moves.airplane.pair_attachments,
            true,
        )?,
        MoveKind::Pass
        | MoveKind::Single
        | MoveKind::Pair
        | MoveKind::Triple
        | MoveKind::TripleWithSingle
        | MoveKind::TripleWithPair
        | MoveKind::Straight
        | MoveKind::PairStraight
        | MoveKind::TripleStraight
        | MoveKind::Bomb
        | MoveKind::Rocket => {}
    }
    Ok(())
}

fn validate_attachment_multiplicity(
    movement: Move,
    multiplicity: AttachmentMultiplicity,
    pairs: bool,
) -> Result<(), DetectMoveError> {
    if multiplicity == AttachmentMultiplicity::MayShareRank {
        return Ok(());
    }
    let body_start = usize::from(movement.main_rank());
    let body_end = body_start + usize::from(movement.chain_len());
    let maximum = if pairs { 2 } else { 1 };
    for (rank, count) in movement.cards().iter() {
        if !(body_start..body_end).contains(&rank.index()) && count > maximum {
            return Err(DetectMoveError::AttachmentRanksMustBeDistinct {
                kind: movement.kind(),
                rank,
            });
        }
    }
    Ok(())
}

#[derive(Debug)]
pub enum DetectMoveError {
    TooManyCards,
    Unrecognized { cards: RankCounts },
    Move(MoveError),
    RuleConfig(RuleConfigError),
    NonCanonical { supplied: Move, canonical: Move },
    MoveDisabled { kind: MoveKind },
    AttachmentRanksMustBeDistinct { kind: MoveKind, rank: Rank },
}

impl Display for DetectMoveError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyCards => write!(formatter, "move card count does not fit in u8"),
            Self::Unrecognized { cards } => write!(
                formatter,
                "rank counts {:?} do not form a supported move",
                cards.as_array()
            ),
            Self::Move(error) => Display::fmt(error, formatter),
            Self::RuleConfig(error) => Display::fmt(error, formatter),
            Self::NonCanonical { supplied, canonical } => write!(
                formatter,
                "supplied move {supplied:?} is not canonical; canonical value is {canonical:?}"
            ),
            Self::MoveDisabled { kind } => write!(formatter, "move kind {kind:?} is disabled"),
            Self::AttachmentRanksMustBeDistinct { kind, rank } => write!(
                formatter,
                "{kind:?} reuses attachment rank {rank:?} but this profile requires distinct ranks"
            ),
        }
    }
}

impl Error for DetectMoveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Move(error) => Some(error),
            Self::RuleConfig(error) => Some(error),
            _ => None,
        }
    }
}
