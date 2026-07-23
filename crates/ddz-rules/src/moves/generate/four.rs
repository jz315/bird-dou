use std::collections::BTreeSet;

use ddz_core::{Move, MoveKind, Rank, RankCounts};

use super::{insert_detected, GenerateMovesError, GenerationFilter};
use crate::moves::attachments::{allocations, attachment_capacities, combine};
use crate::RuleConfig;

pub(super) fn generate(
    hand: RankCounts,
    rules: &RuleConfig,
    filter: GenerationFilter,
    result: &mut BTreeSet<Move>,
) -> Result<(), GenerateMovesError> {
    for four in Rank::ALL {
        if hand[four] != 4 {
            continue;
        }
        let mut body = RankCounts::empty();
        body.set(four, 4).map_err(GenerateMovesError::Counts)?;
        let capacities = attachment_capacities(hand, body);

        if rules.moves.four_with_two.two_singles_enabled
            && filter.accepts_meta(MoveKind::FourWithTwoSingles, 1, four.value())
        {
            for attachment in allocations(
                capacities,
                2,
                1,
                rules.moves.four_with_two.single_attachments,
            ) {
                insert_detected(combine(body, attachment), rules, filter, result)?;
            }
        }
        if rules.moves.four_with_two.two_pairs_enabled
            && filter.accepts_meta(MoveKind::FourWithTwoPairs, 1, four.value())
        {
            for attachment in allocations(
                capacities,
                2,
                2,
                rules.moves.four_with_two.pair_attachments,
            ) {
                insert_detected(combine(body, attachment), rules, filter, result)?;
            }
        }
    }
    Ok(())
}
