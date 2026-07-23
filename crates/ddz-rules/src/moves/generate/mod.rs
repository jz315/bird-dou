mod chains;
mod four;
mod groups;

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use ddz_core::{Move, MoveKind, RankCounts, RankCountsError};

use super::{detect_move_with_rules, move_beats, validate_move_for_rules, DetectMoveError};
use crate::{RuleConfig, RuleConfigError};

pub fn generate_lead_moves(
    hand: RankCounts,
    rules: &RuleConfig,
) -> Result<Vec<Move>, GenerateMovesError> {
    rules.validate().map_err(GenerateMovesError::RuleConfig)?;
    generate_filtered(hand, rules, GenerationFilter::ANY)
}

pub fn generate_follow_moves(
    hand: RankCounts,
    target: Move,
    rules: &RuleConfig,
) -> Result<Vec<Move>, GenerateMovesError> {
    rules.validate().map_err(GenerateMovesError::RuleConfig)?;
    validate_move_for_rules(target, rules).map_err(GenerateMovesError::Target)?;
    if target.is_pass() {
        return Err(GenerateMovesError::TargetIsPass);
    }

    let mut result = BTreeSet::from([Move::pass()]);
    if target.kind() == MoveKind::Rocket {
        return Ok(result.into_iter().collect());
    }

    if target.kind() == MoveKind::Bomb {
        extend_filtered(
            hand,
            rules,
            GenerationFilter::kind(MoveKind::Bomb)
                .with_chain_length(1)
                .above(target.main_rank()),
            &mut result,
        )?;
    } else {
        extend_filtered(
            hand,
            rules,
            GenerationFilter::kind(target.kind())
                .with_chain_length(target.chain_len())
                .above(target.main_rank()),
            &mut result,
        )?;
        extend_filtered(
            hand,
            rules,
            GenerationFilter::kind(MoveKind::Bomb).with_chain_length(1),
            &mut result,
        )?;
    }
    extend_filtered(
        hand,
        rules,
        GenerationFilter::kind(MoveKind::Rocket).with_chain_length(1),
        &mut result,
    )?;

    result.retain(|movement| movement.is_pass() || move_beats(*movement, target));
    Ok(result.into_iter().collect())
}

fn generate_filtered(
    hand: RankCounts,
    rules: &RuleConfig,
    filter: GenerationFilter,
) -> Result<Vec<Move>, GenerateMovesError> {
    let mut result = BTreeSet::new();
    extend_filtered(hand, rules, filter, &mut result)?;
    Ok(result.into_iter().collect())
}

fn extend_filtered(
    hand: RankCounts,
    rules: &RuleConfig,
    filter: GenerationFilter,
    result: &mut BTreeSet<Move>,
) -> Result<(), GenerateMovesError> {
    groups::generate(hand, rules, filter, result)?;
    chains::generate(hand, rules, filter, result)?;
    four::generate(hand, rules, filter, result)
}

pub(super) fn insert_detected(
    cards: RankCounts,
    rules: &RuleConfig,
    filter: GenerationFilter,
    result: &mut BTreeSet<Move>,
) -> Result<(), GenerateMovesError> {
    let movement = detect_move_with_rules(cards, rules).map_err(|source| {
        GenerateMovesError::GeneratedMove {
            cards,
            source: Box::new(source),
        }
    })?;
    if !movement.is_pass()
        && filter.accepts_meta(
            movement.kind(),
            movement.chain_len(),
            movement.main_rank(),
        )
    {
        result.insert(movement);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub(super) struct GenerationFilter {
    kind: Option<MoveKind>,
    chain_length: Option<u8>,
    minimum_main_exclusive: Option<u8>,
}

impl GenerationFilter {
    const ANY: Self = Self {
        kind: None,
        chain_length: None,
        minimum_main_exclusive: None,
    };

    const fn kind(kind: MoveKind) -> Self {
        Self {
            kind: Some(kind),
            chain_length: None,
            minimum_main_exclusive: None,
        }
    }

    const fn with_chain_length(mut self, chain_length: u8) -> Self {
        self.chain_length = Some(chain_length);
        self
    }

    const fn above(mut self, main_rank: u8) -> Self {
        self.minimum_main_exclusive = Some(main_rank);
        self
    }

    pub(super) fn accepts_meta(
        self,
        kind: MoveKind,
        chain_length: u8,
        main_rank: u8,
    ) -> bool {
        (self.kind.is_none() || self.kind == Some(kind))
            && (self.chain_length.is_none() || self.chain_length == Some(chain_length))
            && match self.minimum_main_exclusive {
                Some(minimum) => main_rank > minimum,
                None => true,
            }
    }

    pub(super) fn wants(self, kind: MoveKind) -> bool {
        self.kind.is_none() || self.kind == Some(kind)
    }

    pub(super) const fn requested_chain_length(self) -> Option<u8> {
        self.chain_length
    }

    pub(super) const fn minimum_main_exclusive(self) -> Option<u8> {
        self.minimum_main_exclusive
    }
}

#[derive(Debug)]
pub enum GenerateMovesError {
    RuleConfig(RuleConfigError),
    Counts(RankCountsError),
    Target(DetectMoveError),
    TargetIsPass,
    GeneratedMove {
        cards: RankCounts,
        source: Box<DetectMoveError>,
    },
}

impl Display for GenerateMovesError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuleConfig(error) => Display::fmt(error, formatter),
            Self::Counts(error) => Display::fmt(error, formatter),
            Self::Target(error) => write!(formatter, "invalid follow target: {error}"),
            Self::TargetIsPass => write!(formatter, "follow generation requires a non-pass target"),
            Self::GeneratedMove { cards, source } => write!(
                formatter,
                "generated rank counts {:?} failed move detection: {source}",
                cards.as_array()
            ),
        }
    }
}

impl Error for GenerateMovesError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RuleConfig(error) => Some(error),
            Self::Counts(error) => Some(error),
            Self::Target(error) => Some(error),
            Self::GeneratedMove { source, .. } => Some(source.as_ref()),
            Self::TargetIsPass => None,
        }
    }
}
