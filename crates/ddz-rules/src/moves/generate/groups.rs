use std::collections::BTreeSet;

use ddz_core::{Move, MoveKind, Rank, RankCounts};

use super::{insert_detected, GenerateMovesError, GenerationFilter};
use crate::RuleConfig;

pub(super) fn generate(
    hand: RankCounts,
    rules: &RuleConfig,
    filter: GenerationFilter,
    result: &mut BTreeSet<Move>,
) -> Result<(), GenerateMovesError> {
    for rank in Rank::ALL {
        let available = hand[rank];
        for (kind, required) in [
            (MoveKind::Single, 1),
            (MoveKind::Pair, 2),
            (MoveKind::Triple, 3),
            (MoveKind::Bomb, 4),
        ] {
            if available >= required && filter.accepts_meta(kind, 1, rank.value()) {
                let mut cards = RankCounts::empty();
                cards.set(rank, required).map_err(GenerateMovesError::Counts)?;
                insert_detected(cards, rules, filter, result)?;
            }
        }

        if available >= 3 && filter.wants(MoveKind::TripleWithSingle) {
            generate_triple_attachment(hand, rank, false, rules, filter, result)?;
        }
        if available >= 3 && filter.wants(MoveKind::TripleWithPair) {
            generate_triple_attachment(hand, rank, true, rules, filter, result)?;
        }
    }

    if filter.accepts_meta(MoveKind::Rocket, 1, Rank::BigJoker.value())
        && hand[Rank::SmallJoker] == 1
        && hand[Rank::BigJoker] == 1
    {
        let mut cards = RankCounts::empty();
        cards
            .set(Rank::SmallJoker, 1)
            .map_err(GenerateMovesError::Counts)?;
        cards
            .set(Rank::BigJoker, 1)
            .map_err(GenerateMovesError::Counts)?;
        insert_detected(cards, rules, filter, result)?;
    }
    Ok(())
}

fn generate_triple_attachment(
    hand: RankCounts,
    body_rank: Rank,
    pairs: bool,
    rules: &RuleConfig,
    filter: GenerationFilter,
    result: &mut BTreeSet<Move>,
) -> Result<(), GenerateMovesError> {
    let kind = if pairs {
        MoveKind::TripleWithPair
    } else {
        MoveKind::TripleWithSingle
    };
    if !filter.accepts_meta(kind, 1, body_rank.value()) {
        return Ok(());
    }
    let required = if pairs { 2 } else { 1 };
    for attachment_rank in Rank::ALL {
        if attachment_rank == body_rank || hand[attachment_rank] < required {
            continue;
        }
        let mut cards = RankCounts::empty();
        cards
            .set(body_rank, 3)
            .map_err(GenerateMovesError::Counts)?;
        cards
            .set(attachment_rank, required)
            .map_err(GenerateMovesError::Counts)?;
        insert_detected(cards, rules, filter, result)?;
    }
    Ok(())
}
