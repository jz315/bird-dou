use std::collections::BTreeSet;

use ddz_core::{Move, MoveKind, Rank, RankCounts};

use super::{insert_detected, GenerateMovesError, GenerationFilter};
use crate::moves::attachments::{allocations, attachment_capacities, combine};
use crate::RuleConfig;

const CHAIN_RANK_COUNT: usize = 12;

pub(super) fn generate(
    hand: RankCounts,
    rules: &RuleConfig,
    filter: GenerationFilter,
    result: &mut BTreeSet<Move>,
) -> Result<(), GenerateMovesError> {
    generate_uniform(hand, rules, filter, result)?;
    generate_airplanes(hand, rules, filter, result)
}

fn generate_uniform(
    hand: RankCounts,
    rules: &RuleConfig,
    filter: GenerationFilter,
    result: &mut BTreeSet<Move>,
) -> Result<(), GenerateMovesError> {
    for (kind, multiplicity, minimum_length) in [
        (MoveKind::Straight, 1, 5),
        (MoveKind::PairStraight, 2, 3),
        (MoveKind::TripleStraight, 3, 2),
    ] {
        if !filter.wants(kind) {
            continue;
        }
        for (start, length) in intervals(hand, multiplicity, minimum_length, filter) {
            insert_detected(body(start, length, multiplicity)?, rules, filter, result)?;
        }
    }
    Ok(())
}

fn generate_airplanes(
    hand: RankCounts,
    rules: &RuleConfig,
    filter: GenerationFilter,
    result: &mut BTreeSet<Move>,
) -> Result<(), GenerateMovesError> {
    for (kind, pairs, multiplicity) in [
        (
            MoveKind::AirplaneWithSingles,
            false,
            rules.moves.airplane.single_attachments,
        ),
        (
            MoveKind::AirplaneWithPairs,
            true,
            rules.moves.airplane.pair_attachments,
        ),
    ] {
        if !filter.wants(kind) {
            continue;
        }
        for (start, length) in intervals(hand, 3, 2, filter) {
            if !filter.accepts_meta(kind, length, start.value()) {
                continue;
            }
            let body = body(start, length, 3)?;
            let capacities = attachment_capacities(hand, body);
            for attachment in allocations(
                capacities,
                length,
                if pairs { 2 } else { 1 },
                multiplicity,
            ) {
                insert_detected(combine(body, attachment), rules, filter, result)?;
            }
        }
    }
    Ok(())
}

fn intervals(
    hand: RankCounts,
    multiplicity: u8,
    minimum_length: u8,
    filter: GenerationFilter,
) -> Vec<(Rank, u8)> {
    let lengths: Vec<u8> = match filter.requested_chain_length() {
        Some(length) => vec![length],
        None => (minimum_length..=12).collect(),
    };
    let mut result = Vec::new();
    for length in lengths {
        if length < minimum_length || usize::from(length) > CHAIN_RANK_COUNT {
            continue;
        }
        let maximum_start = CHAIN_RANK_COUNT - usize::from(length);
        for start_index in 0..=maximum_start {
            let start = Rank::ALL[start_index];
            if filter
                .minimum_main_exclusive()
                .is_some_and(|minimum| start.value() <= minimum)
            {
                continue;
            }
            if (start_index..start_index + usize::from(length))
                .all(|index| hand[Rank::ALL[index]] >= multiplicity)
            {
                result.push((start, length));
            }
        }
    }
    result
}

fn body(
    start: Rank,
    length: u8,
    multiplicity: u8,
) -> Result<RankCounts, GenerateMovesError> {
    let mut body = RankCounts::empty();
    for index in start.index()..start.index() + usize::from(length) {
        body.set(Rank::ALL[index], multiplicity)
            .map_err(GenerateMovesError::Counts)?;
    }
    Ok(body)
}
