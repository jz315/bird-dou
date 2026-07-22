//! Classification of rank counts into canonical moves.

use std::error::Error;
use std::fmt::{Display, Formatter};

use ddz_core::{
    validate_rank_counts, CardError, Move, MoveError, MoveKind, RankCounts, RankId, BIG_JOKER_RANK,
    SMALL_JOKER_RANK,
};

use crate::{AttachmentMultiplicity, RuleConfig, RuleConfigError, RuleProfile};

const CHAIN_RANK_COUNT: usize = 12;

/// Detect a structural move without silently choosing a platform profile.
///
/// This accepts the union of attachment forms expressible by [`RuleConfig`]. Use
/// [`detect_move_with_rules`] when checking whether a play is legal in a concrete
/// environment profile.
///
/// # Errors
///
/// Returns [`DetectMoveError::Cards`] for impossible rank counts or
/// [`DetectMoveError::Unrecognized`] when no move shape matches.
pub fn detect_move(cards: RankCounts) -> Result<Move, DetectMoveError> {
    validate_rank_counts(&cards).map_err(DetectMoveError::Cards)?;
    let total: u8 = cards.iter().sum();

    if total == 0 {
        return Ok(Move::pass());
    }
    if let Some(detected) = detect_simple_group(&cards, total)? {
        return Ok(detected);
    }
    if let Some(detected) = detect_triple_attachment(&cards, total)? {
        return Ok(detected);
    }
    if let Some(detected) = detect_uniform_chain(&cards, 1, 5, MoveKind::Straight)? {
        return Ok(detected);
    }
    if let Some(detected) = detect_four_with_two(&cards, total)? {
        return Ok(detected);
    }
    if let Some(detected) = detect_uniform_chain(&cards, 2, 3, MoveKind::PairStraight)? {
        return Ok(detected);
    }
    if let Some(detected) = detect_uniform_chain(&cards, 3, 2, MoveKind::TripleStraight)? {
        return Ok(detected);
    }
    if let Some(detected) = detect_airplane(&cards, total, false)? {
        return Ok(detected);
    }
    if let Some(detected) = detect_airplane(&cards, total, true)? {
        return Ok(detected);
    }

    Err(DetectMoveError::Unrecognized { cards })
}

/// Detect a move and enforce every attachment switch in a concrete rule profile.
///
/// # Errors
///
/// Returns any structural error from [`detect_move`], a configuration validation
/// error, or a profile-policy error when the shape is disabled by `rules`.
pub fn detect_move_with_rules(
    cards: RankCounts,
    rules: &RuleConfig,
) -> Result<Move, DetectMoveError> {
    rules.validate().map_err(DetectMoveError::RuleConfig)?;
    let detected = detect_move(cards)?;
    validate_for_rules(&detected, rules)?;
    Ok(detected)
}

fn detect_simple_group(cards: &RankCounts, total: u8) -> Result<Option<Move>, DetectMoveError> {
    if total == 2
        && cards[usize::from(SMALL_JOKER_RANK)] == 1
        && cards[usize::from(BIG_JOKER_RANK)] == 1
    {
        return make_move(MoveKind::Rocket, *cards, BIG_JOKER_RANK, 1).map(Some);
    }

    let (kind, multiplicity) = match total {
        1 => (MoveKind::Single, 1),
        2 => (MoveKind::Pair, 2),
        3 => (MoveKind::Triple, 3),
        4 => (MoveKind::Bomb, 4),
        _ => return Ok(None),
    };
    let Some(main_rank) = sole_nonzero_rank(cards) else {
        return Ok(None);
    };
    if cards[usize::from(main_rank)] != multiplicity {
        return Ok(None);
    }
    make_move(kind, *cards, main_rank, 1).map(Some)
}

fn detect_triple_attachment(
    cards: &RankCounts,
    total: u8,
) -> Result<Option<Move>, DetectMoveError> {
    let kind = match total {
        4 => MoveKind::TripleWithSingle,
        5 => MoveKind::TripleWithPair,
        _ => return Ok(None),
    };
    let Some(main_rank) = unique_rank_with_count(cards, 3) else {
        return Ok(None);
    };

    if kind == MoveKind::TripleWithPair
        && cards
            .iter()
            .enumerate()
            .any(|(rank, &count)| rank != usize::from(main_rank) && count % 2 != 0)
    {
        return Ok(None);
    }
    make_move(kind, *cards, main_rank, 1).map(Some)
}

fn detect_uniform_chain(
    cards: &RankCounts,
    multiplicity: u8,
    minimum_len: u8,
    kind: MoveKind,
) -> Result<Option<Move>, DetectMoveError> {
    let ranks: Vec<usize> = cards
        .iter()
        .enumerate()
        .filter_map(|(rank, &count)| (count != 0).then_some(rank))
        .collect();
    if ranks.len() < usize::from(minimum_len)
        || ranks.iter().any(|&rank| rank >= CHAIN_RANK_COUNT)
        || ranks.iter().any(|&rank| cards[rank] != multiplicity)
        || !ranks.windows(2).all(|pair| pair[1] == pair[0] + 1)
    {
        return Ok(None);
    }

    let main_rank =
        RankId::try_from(ranks[0]).map_err(|_| DetectMoveError::Unrecognized { cards: *cards })?;
    let chain_len =
        u8::try_from(ranks.len()).map_err(|_| DetectMoveError::Unrecognized { cards: *cards })?;
    make_move(kind, *cards, main_rank, chain_len).map(Some)
}

fn detect_four_with_two(cards: &RankCounts, total: u8) -> Result<Option<Move>, DetectMoveError> {
    let kind = match total {
        6 => MoveKind::FourWithTwoSingles,
        8 => MoveKind::FourWithTwoPairs,
        _ => return Ok(None),
    };
    let Some(main_rank) = ranks_with_count(cards, 4).into_iter().max() else {
        return Ok(None);
    };
    if kind == MoveKind::FourWithTwoPairs
        && cards
            .iter()
            .enumerate()
            .any(|(rank, &count)| rank != usize::from(main_rank) && count % 2 != 0)
    {
        return Ok(None);
    }
    make_move(kind, *cards, main_rank, 1).map(Some)
}

fn detect_airplane(
    cards: &RankCounts,
    total: u8,
    pair_wings: bool,
) -> Result<Option<Move>, DetectMoveError> {
    let cards_per_body_rank = if pair_wings { 5 } else { 4 };
    if total % cards_per_body_rank != 0 {
        return Ok(None);
    }
    let chain_len = total / cards_per_body_rank;
    if chain_len < 2 || usize::from(chain_len) > CHAIN_RANK_COUNT {
        return Ok(None);
    }

    let maximum_start = CHAIN_RANK_COUNT - usize::from(chain_len);
    let mut candidates = Vec::new();
    for start in 0..=maximum_start {
        let end = start + usize::from(chain_len);
        if cards[start..end].iter().all(|&count| count == 3)
            && attachments_match(cards, start, end, pair_wings)
        {
            candidates.push(start);
        }
    }
    let Some(main_rank) = candidates.into_iter().max() else {
        return Ok(None);
    };
    let main_rank =
        RankId::try_from(main_rank).map_err(|_| DetectMoveError::Unrecognized { cards: *cards })?;
    let kind = if pair_wings {
        MoveKind::AirplaneWithPairs
    } else {
        MoveKind::AirplaneWithSingles
    };
    make_move(kind, *cards, main_rank, chain_len).map(Some)
}

fn attachments_match(cards: &RankCounts, body_start: usize, body_end: usize, pairs: bool) -> bool {
    cards
        .iter()
        .enumerate()
        .all(|(rank, &count)| (body_start..body_end).contains(&rank) || !pairs || count % 2 == 0)
}

fn validate_for_rules(detected: &Move, rules: &RuleConfig) -> Result<(), DetectMoveError> {
    match detected.kind() {
        MoveKind::FourWithTwoSingles => {
            if !rules.four_with_two.two_singles_enabled {
                return Err(disabled(detected.kind(), rules.profile));
            }
            validate_attachment_multiplicity(
                detected,
                rules.four_with_two.single_attachments,
                false,
            )?;
        }
        MoveKind::FourWithTwoPairs => {
            if !rules.four_with_two.two_pairs_enabled {
                return Err(disabled(detected.kind(), rules.profile));
            }
            validate_attachment_multiplicity(detected, rules.four_with_two.pair_attachments, true)?;
        }
        MoveKind::AirplaneWithSingles => {
            validate_attachment_multiplicity(detected, rules.airplane.single_attachments, false)?;
        }
        MoveKind::AirplaneWithPairs => {
            validate_attachment_multiplicity(detected, rules.airplane.pair_attachments, true)?;
        }
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
    detected: &Move,
    multiplicity: AttachmentMultiplicity,
    pair_attachments: bool,
) -> Result<(), DetectMoveError> {
    if multiplicity == AttachmentMultiplicity::MayShareRank {
        return Ok(());
    }

    let body_start = usize::from(detected.main_rank());
    let body_end = body_start + usize::from(detected.chain_len());
    let maximum = if pair_attachments { 2 } else { 1 };
    for (rank_id, &count) in (0_u8..).zip(detected.cards().iter()) {
        let is_body = (body_start..body_end).contains(&usize::from(rank_id));
        if !is_body && count > maximum {
            return Err(DetectMoveError::AttachmentRanksMustBeDistinct {
                kind: detected.kind(),
                rank_id,
            });
        }
    }
    Ok(())
}

fn disabled(kind: MoveKind, profile: RuleProfile) -> DetectMoveError {
    DetectMoveError::MoveDisabled { kind, profile }
}

fn sole_nonzero_rank(cards: &RankCounts) -> Option<RankId> {
    let mut ranks = cards
        .iter()
        .enumerate()
        .filter_map(|(rank, &count)| (count != 0).then_some(rank));
    let rank = ranks.next()?;
    if ranks.next().is_some() {
        return None;
    }
    RankId::try_from(rank).ok()
}

fn unique_rank_with_count(cards: &RankCounts, expected: u8) -> Option<RankId> {
    let ranks = ranks_with_count(cards, expected);
    if ranks.len() == 1 {
        Some(ranks[0])
    } else {
        None
    }
}

fn ranks_with_count(cards: &RankCounts, expected: u8) -> Vec<RankId> {
    cards
        .iter()
        .enumerate()
        .filter_map(|(rank, &count)| {
            if count == expected {
                RankId::try_from(rank).ok()
            } else {
                None
            }
        })
        .collect()
}

fn make_move(
    kind: MoveKind,
    cards: RankCounts,
    main_rank: RankId,
    chain_len: u8,
) -> Result<Move, DetectMoveError> {
    Move::try_new(kind, cards, main_rank, chain_len).map_err(DetectMoveError::Move)
}

/// Errors returned by structural or profile-aware move detection.
#[derive(Debug)]
pub enum DetectMoveError {
    /// Rank counts exceed physical deck capacity.
    Cards(CardError),
    /// Counts do not match any supported move shape.
    Unrecognized {
        /// Rejected rank counts.
        cards: RankCounts,
    },
    /// A detected shape violated the canonical move constructor.
    Move(MoveError),
    /// The supplied rule configuration was invalid.
    RuleConfig(RuleConfigError),
    /// The concrete profile disabled this move form.
    MoveDisabled {
        /// Detected kind.
        kind: MoveKind,
        /// Profile that rejected it.
        profile: RuleProfile,
    },
    /// A profile requires distinct attachment ranks but a rank was reused.
    AttachmentRanksMustBeDistinct {
        /// Detected kind.
        kind: MoveKind,
        /// Reused attachment rank.
        rank_id: RankId,
    },
}

impl Display for DetectMoveError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cards(error) => Display::fmt(error, formatter),
            Self::Unrecognized { cards } => {
                write!(formatter, "rank counts do not form a recognized move: {cards:?}")
            }
            Self::Move(error) => Display::fmt(error, formatter),
            Self::RuleConfig(error) => Display::fmt(error, formatter),
            Self::MoveDisabled { kind, profile } => {
                write!(formatter, "move {kind:?} is disabled by profile {profile:?}")
            }
            Self::AttachmentRanksMustBeDistinct { kind, rank_id } => write!(
                formatter,
                "move {kind:?} reuses attachment rank {rank_id}, but the profile requires distinct ranks"
            ),
        }
    }
}

impl Error for DetectMoveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Cards(error) => Some(error),
            Self::Move(error) => Some(error),
            Self::RuleConfig(error) => Some(error),
            Self::Unrecognized { .. }
            | Self::MoveDisabled { .. }
            | Self::AttachmentRanksMustBeDistinct { .. } => None,
        }
    }
}
