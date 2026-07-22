//! Template-based legal move generation for free leads.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use ddz_core::{
    validate_rank_counts, CardError, Move, MoveKind, RankCounts, BIG_JOKER_RANK, EMPTY_RANK_COUNTS,
    RANK_COUNT, SMALL_JOKER_RANK,
};

use crate::{
    detect_move_with_rules, AttachmentMultiplicity, DetectMoveError, RuleConfig, RuleConfigError,
};

const CHAIN_RANK_COUNT: usize = 12;

/// Generate every legal non-pass move that can lead a trick from `hand`.
///
/// The implementation enumerates rank templates and capacity-constrained
/// attachment allocations. It never enumerates arbitrary subsets of physical
/// cards. Returned moves are deduplicated and follow [`Move`]'s stable total order.
///
/// # Errors
///
/// Returns [`GenerateMovesError::Cards`] for an impossible hand,
/// [`GenerateMovesError::RuleConfig`] for invalid rules, or
/// [`GenerateMovesError::GeneratedMove`] if an internal template fails canonical
/// rule-aware detection.
pub fn generate_lead_moves(
    hand: &RankCounts,
    rules: &RuleConfig,
) -> Result<Vec<Move>, GenerateMovesError> {
    validate_generation_inputs(hand, rules)?;
    generate_lead_moves_validated(hand, rules)
}

/// Generate every legal response to a non-pass target move.
///
/// A normal response must have the same kind and chain length as `target`, with
/// a strictly higher main rank. Any bomb beats a non-bomb, higher bombs beat
/// lower bombs, and the rocket beats every move except another rocket. Pass is
/// always included and follows [`Move`]'s stable total order.
///
/// # Errors
///
/// Returns [`GenerateMovesError::Cards`] for an impossible hand,
/// [`GenerateMovesError::RuleConfig`] for invalid rules,
/// [`GenerateMovesError::Target`] when the target is illegal under `rules`,
/// [`GenerateMovesError::TargetIsPass`] when called without an active target, or
/// [`GenerateMovesError::NonCanonicalTarget`] when target metadata differs from
/// rule-aware canonical detection.
pub fn generate_follow_moves(
    hand: &RankCounts,
    target: &Move,
    rules: &RuleConfig,
) -> Result<Vec<Move>, GenerateMovesError> {
    validate_generation_inputs(hand, rules)?;
    validate_follow_target(target, rules)?;

    if target.kind() == MoveKind::Rocket {
        return Ok(vec![Move::pass()]);
    }

    let mut responses = BTreeSet::from([Move::pass()]);
    for candidate in generate_lead_moves_validated(hand, rules)? {
        if move_beats(&candidate, target) {
            responses.insert(candidate);
        }
    }
    Ok(responses.into_iter().collect())
}

fn validate_generation_inputs(
    hand: &RankCounts,
    rules: &RuleConfig,
) -> Result<(), GenerateMovesError> {
    validate_rank_counts(hand).map_err(GenerateMovesError::Cards)?;
    rules.validate().map_err(GenerateMovesError::RuleConfig)
}

fn validate_follow_target(target: &Move, rules: &RuleConfig) -> Result<(), GenerateMovesError> {
    let canonical =
        detect_move_with_rules(*target.cards(), rules).map_err(GenerateMovesError::Target)?;
    if canonical.kind() == MoveKind::Pass {
        return Err(GenerateMovesError::TargetIsPass);
    }
    if canonical != *target {
        return Err(GenerateMovesError::NonCanonicalTarget {
            supplied: *target,
            canonical,
        });
    }
    Ok(())
}

pub(crate) fn move_beats(candidate: &Move, target: &Move) -> bool {
    match (candidate.kind(), target.kind()) {
        (_, MoveKind::Rocket) => false,
        (MoveKind::Bomb, MoveKind::Bomb) => candidate.main_rank() > target.main_rank(),
        (MoveKind::Rocket | MoveKind::Bomb, _) => true,
        (_, MoveKind::Bomb) => false,
        _ => {
            candidate.kind() == target.kind()
                && candidate.chain_len() == target.chain_len()
                && candidate.main_rank() > target.main_rank()
        }
    }
}

fn generate_lead_moves_validated(
    hand: &RankCounts,
    rules: &RuleConfig,
) -> Result<Vec<Move>, GenerateMovesError> {
    let mut moves = BTreeSet::new();
    generate_groups(hand, rules, &mut moves)?;
    generate_uniform_chains(hand, rules, &mut moves)?;
    generate_airplanes(hand, rules, &mut moves)?;
    generate_four_with_two(hand, rules, &mut moves)?;

    Ok(moves.into_iter().collect())
}

fn generate_groups(
    hand: &RankCounts,
    rules: &RuleConfig,
    moves: &mut BTreeSet<Move>,
) -> Result<(), GenerateMovesError> {
    for (rank_id, &available) in (0_u8..).zip(hand.iter()) {
        for required in [1, 2, 3] {
            if available >= required {
                let mut cards = EMPTY_RANK_COUNTS;
                cards[usize::from(rank_id)] = required;
                insert_candidate(cards, rules, moves)?;
            }
        }

        if available == 4 {
            let mut cards = EMPTY_RANK_COUNTS;
            cards[usize::from(rank_id)] = 4;
            insert_candidate(cards, rules, moves)?;
        }

        if available >= 3 {
            generate_triple_attachments(hand, rank_id, rules, moves)?;
        }
    }

    if hand[usize::from(SMALL_JOKER_RANK)] == 1 && hand[usize::from(BIG_JOKER_RANK)] == 1 {
        let mut cards = EMPTY_RANK_COUNTS;
        cards[usize::from(SMALL_JOKER_RANK)] = 1;
        cards[usize::from(BIG_JOKER_RANK)] = 1;
        insert_candidate(cards, rules, moves)?;
    }
    Ok(())
}

fn generate_triple_attachments(
    hand: &RankCounts,
    body_rank: u8,
    rules: &RuleConfig,
    moves: &mut BTreeSet<Move>,
) -> Result<(), GenerateMovesError> {
    for (attachment_rank, &available) in (0_u8..).zip(hand.iter()) {
        if attachment_rank == body_rank {
            continue;
        }
        if available >= 1 {
            let mut cards = EMPTY_RANK_COUNTS;
            cards[usize::from(body_rank)] = 3;
            cards[usize::from(attachment_rank)] = 1;
            insert_candidate(cards, rules, moves)?;
        }
        if available >= 2 {
            let mut cards = EMPTY_RANK_COUNTS;
            cards[usize::from(body_rank)] = 3;
            cards[usize::from(attachment_rank)] = 2;
            insert_candidate(cards, rules, moves)?;
        }
    }
    Ok(())
}

fn generate_uniform_chains(
    hand: &RankCounts,
    rules: &RuleConfig,
    moves: &mut BTreeSet<Move>,
) -> Result<(), GenerateMovesError> {
    for (multiplicity, minimum_len) in [(1, 5), (2, 3), (3, 2)] {
        for (start, chain_len) in chain_intervals(hand, multiplicity, minimum_len) {
            let cards = chain_body(start, chain_len, multiplicity);
            insert_candidate(cards, rules, moves)?;
        }
    }
    Ok(())
}

fn generate_airplanes(
    hand: &RankCounts,
    rules: &RuleConfig,
    moves: &mut BTreeSet<Move>,
) -> Result<(), GenerateMovesError> {
    for (start, chain_len) in chain_intervals(hand, 3, 2) {
        let body = chain_body(start, chain_len, 3);
        let capacities = attachment_capacities(hand, &body);

        for attachments in
            attachment_allocations(&capacities, chain_len, 1, rules.airplane.single_attachments)
        {
            insert_candidate(combine(&body, &attachments), rules, moves)?;
        }
        for attachments in
            attachment_allocations(&capacities, chain_len, 2, rules.airplane.pair_attachments)
        {
            insert_candidate(combine(&body, &attachments), rules, moves)?;
        }
    }
    Ok(())
}

fn generate_four_with_two(
    hand: &RankCounts,
    rules: &RuleConfig,
    moves: &mut BTreeSet<Move>,
) -> Result<(), GenerateMovesError> {
    for (body_rank, &available) in (0_u8..).zip(hand.iter()) {
        if available != 4 {
            continue;
        }
        let mut body = EMPTY_RANK_COUNTS;
        body[usize::from(body_rank)] = 4;
        let capacities = attachment_capacities(hand, &body);

        if rules.four_with_two.two_singles_enabled {
            for attachments in
                attachment_allocations(&capacities, 2, 1, rules.four_with_two.single_attachments)
            {
                insert_candidate(combine(&body, &attachments), rules, moves)?;
            }
        }
        if rules.four_with_two.two_pairs_enabled {
            for attachments in
                attachment_allocations(&capacities, 2, 2, rules.four_with_two.pair_attachments)
            {
                insert_candidate(combine(&body, &attachments), rules, moves)?;
            }
        }
    }
    Ok(())
}

fn chain_intervals(hand: &RankCounts, multiplicity: u8, minimum_len: u8) -> Vec<(u8, u8)> {
    let mut intervals = Vec::new();
    for start in 0..CHAIN_RANK_COUNT {
        if hand[start] < multiplicity {
            continue;
        }
        let mut end = start;
        while end < CHAIN_RANK_COUNT && hand[end] >= multiplicity {
            end += 1;
        }
        let run_len = end - start;
        for length in usize::from(minimum_len)..=run_len {
            let start_rank = u8::try_from(start).expect("chain rank index fits in u8");
            let chain_len = u8::try_from(length).expect("chain length fits in u8");
            intervals.push((start_rank, chain_len));
        }
    }
    intervals
}

fn chain_body(start: u8, chain_len: u8, multiplicity: u8) -> RankCounts {
    let mut cards = EMPTY_RANK_COUNTS;
    for rank in start..start + chain_len {
        cards[usize::from(rank)] = multiplicity;
    }
    cards
}

fn attachment_capacities(hand: &RankCounts, body: &RankCounts) -> RankCounts {
    let mut capacities = EMPTY_RANK_COUNTS;
    for rank in 0..RANK_COUNT {
        capacities[rank] = hand[rank] - body[rank];
        if body[rank] != 0 {
            capacities[rank] = 0;
        }
    }
    capacities
}

fn attachment_allocations(
    capacities: &RankCounts,
    required_units: u8,
    cards_per_unit: u8,
    multiplicity: AttachmentMultiplicity,
) -> Vec<RankCounts> {
    let mut unit_capacities = EMPTY_RANK_COUNTS;
    for rank in 0..RANK_COUNT {
        let available_units = capacities[rank] / cards_per_unit;
        unit_capacities[rank] = match multiplicity {
            AttachmentMultiplicity::DistinctRanks => available_units.min(1),
            AttachmentMultiplicity::MayShareRank => available_units,
        };
    }

    let mut allocations = Vec::new();
    let mut current = EMPTY_RANK_COUNTS;
    enumerate_allocations(
        &unit_capacities,
        required_units,
        cards_per_unit,
        0,
        &mut current,
        &mut allocations,
    );
    allocations
}

fn enumerate_allocations(
    unit_capacities: &RankCounts,
    units_left: u8,
    cards_per_unit: u8,
    rank: usize,
    current: &mut RankCounts,
    allocations: &mut Vec<RankCounts>,
) {
    if units_left == 0 {
        allocations.push(*current);
        return;
    }
    if rank == RANK_COUNT {
        return;
    }

    let remaining_capacity: u8 = unit_capacities[rank..].iter().sum();
    if remaining_capacity < units_left {
        return;
    }

    let maximum = unit_capacities[rank].min(units_left);
    for units in 0..=maximum {
        current[rank] = units * cards_per_unit;
        enumerate_allocations(
            unit_capacities,
            units_left - units,
            cards_per_unit,
            rank + 1,
            current,
            allocations,
        );
    }
    current[rank] = 0;
}

fn combine(body: &RankCounts, attachments: &RankCounts) -> RankCounts {
    let mut cards = EMPTY_RANK_COUNTS;
    for rank in 0..RANK_COUNT {
        cards[rank] = body[rank] + attachments[rank];
    }
    cards
}

fn insert_candidate(
    cards: RankCounts,
    rules: &RuleConfig,
    moves: &mut BTreeSet<Move>,
) -> Result<(), GenerateMovesError> {
    let detected = detect_move_with_rules(cards, rules).map_err(|source| {
        GenerateMovesError::GeneratedMove {
            cards,
            source: Box::new(source),
        }
    })?;
    moves.insert(detected);
    Ok(())
}

/// Errors returned by free-lead move generation.
#[derive(Debug)]
pub enum GenerateMovesError {
    /// The hand exceeds physical deck capacity.
    Cards(CardError),
    /// The supplied rule configuration is invalid.
    RuleConfig(RuleConfigError),
    /// The response target is structurally invalid or forbidden by the profile.
    Target(DetectMoveError),
    /// Response generation requires a non-pass target.
    TargetIsPass,
    /// Target metadata did not use the detector's canonical interpretation.
    NonCanonicalTarget {
        /// Target supplied by the caller.
        supplied: Move,
        /// Canonical target inferred from its cards.
        canonical: Move,
    },
    /// A generated template failed canonical rule-aware detection.
    GeneratedMove {
        /// Candidate rank counts.
        cards: RankCounts,
        /// Detector failure preserving its original context.
        source: Box<DetectMoveError>,
    },
}

impl Display for GenerateMovesError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cards(error) => Display::fmt(error, formatter),
            Self::RuleConfig(error) => Display::fmt(error, formatter),
            Self::Target(error) => write!(formatter, "invalid response target: {error}"),
            Self::TargetIsPass => write!(
                formatter,
                "response generation requires the active non-pass target move"
            ),
            Self::NonCanonicalTarget {
                supplied,
                canonical,
            } => write!(
                formatter,
                "response target {supplied:?} is not canonical; detected {canonical:?}"
            ),
            Self::GeneratedMove { cards, source } => write!(
                formatter,
                "generated candidate {cards:?} failed canonical detection: {source}"
            ),
        }
    }
}

impl Error for GenerateMovesError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Cards(error) => Some(error),
            Self::RuleConfig(error) => Some(error),
            Self::Target(error) => Some(error),
            Self::GeneratedMove { source, .. } => Some(source.as_ref()),
            Self::TargetIsPass | Self::NonCanonicalTarget { .. } => None,
        }
    }
}
