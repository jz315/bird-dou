use ddz_core::{Move, MoveKind, Rank, RankCounts};

use super::DetectMoveError;

const CHAIN_RANK_COUNT: usize = 12;

pub fn detect_move(cards: RankCounts) -> Result<Move, DetectMoveError> {
    let total = u8::try_from(cards.card_count()).map_err(|_| DetectMoveError::TooManyCards)?;
    if total == 0 {
        return Ok(Move::pass());
    }
    if let Some(movement) = detect_rocket(cards)? {
        return Ok(movement);
    }
    if let Some(movement) = detect_simple_group(cards, total)? {
        return Ok(movement);
    }
    if let Some(movement) = detect_triple_attachment(cards, total)? {
        return Ok(movement);
    }
    for (multiplicity, minimum, kind) in [
        (1, 5, MoveKind::Straight),
        (2, 3, MoveKind::PairStraight),
        (3, 2, MoveKind::TripleStraight),
    ] {
        if let Some(movement) = detect_uniform_chain(cards, multiplicity, minimum, kind)? {
            return Ok(movement);
        }
    }
    if let Some(movement) = detect_four_with_two(cards, total)? {
        return Ok(movement);
    }
    if let Some(movement) = detect_airplane(cards, total, false)? {
        return Ok(movement);
    }
    if let Some(movement) = detect_airplane(cards, total, true)? {
        return Ok(movement);
    }
    Err(DetectMoveError::Unrecognized { cards })
}

fn detect_rocket(cards: RankCounts) -> Result<Option<Move>, DetectMoveError> {
    if cards.card_count() == 2
        && cards[Rank::SmallJoker] == 1
        && cards[Rank::BigJoker] == 1
    {
        return make_move(MoveKind::Rocket, cards, Rank::BigJoker, 1).map(Some);
    }
    Ok(None)
}

fn detect_simple_group(cards: RankCounts, total: u8) -> Result<Option<Move>, DetectMoveError> {
    let (kind, required) = match total {
        1 => (MoveKind::Single, 1),
        2 => (MoveKind::Pair, 2),
        3 => (MoveKind::Triple, 3),
        4 => (MoveKind::Bomb, 4),
        _ => return Ok(None),
    };
    let Some(rank) = sole_nonzero_rank(cards) else {
        return Ok(None);
    };
    if cards[rank] != required {
        return Ok(None);
    }
    make_move(kind, cards, rank, 1).map(Some)
}

fn detect_triple_attachment(
    cards: RankCounts,
    total: u8,
) -> Result<Option<Move>, DetectMoveError> {
    let kind = match total {
        4 => MoveKind::TripleWithSingle,
        5 => MoveKind::TripleWithPair,
        _ => return Ok(None),
    };
    let triples = ranks_with_count(cards, 3);
    if triples.len() != 1 {
        return Ok(None);
    }
    let triple = triples[0];
    if kind == MoveKind::TripleWithPair
        && cards
            .iter()
            .any(|(rank, count)| rank != triple && count % 2 != 0)
    {
        return Ok(None);
    }
    make_move(kind, cards, triple, 1).map(Some)
}

fn detect_uniform_chain(
    cards: RankCounts,
    multiplicity: u8,
    minimum_length: u8,
    kind: MoveKind,
) -> Result<Option<Move>, DetectMoveError> {
    let ranks = cards
        .iter()
        .filter_map(|(rank, count)| (count != 0).then_some((rank, count)))
        .collect::<Vec<_>>();
    if ranks.len() < usize::from(minimum_length)
        || ranks.iter().any(|(rank, _)| !rank.is_straight_eligible())
        || ranks.iter().any(|(_, count)| *count != multiplicity)
        || !ranks
            .windows(2)
            .all(|pair| pair[1].0.value() == pair[0].0.value() + 1)
    {
        return Ok(None);
    }
    let length = u8::try_from(ranks.len()).map_err(|_| DetectMoveError::TooManyCards)?;
    make_move(kind, cards, ranks[0].0, length).map(Some)
}

fn detect_four_with_two(
    cards: RankCounts,
    total: u8,
) -> Result<Option<Move>, DetectMoveError> {
    let kind = match total {
        6 => MoveKind::FourWithTwoSingles,
        8 => MoveKind::FourWithTwoPairs,
        _ => return Ok(None),
    };
    for four in ranks_with_count(cards, 4).into_iter().rev() {
        if kind == MoveKind::FourWithTwoPairs
            && cards
                .iter()
                .any(|(rank, count)| rank != four && count % 2 != 0)
        {
            continue;
        }
        return make_move(kind, cards, four, 1).map(Some);
    }
    Ok(None)
}

fn detect_airplane(
    cards: RankCounts,
    total: u8,
    pair_wings: bool,
) -> Result<Option<Move>, DetectMoveError> {
    let per_body_rank: u8 = if pair_wings { 5 } else { 4 };
    if total % per_body_rank != 0 {
        return Ok(None);
    }
    let length = total / per_body_rank;
    if length < 2 || usize::from(length) > CHAIN_RANK_COUNT {
        return Ok(None);
    }
    let maximum_start = CHAIN_RANK_COUNT - usize::from(length);
    for start_index in (0..=maximum_start).rev() {
        let end = start_index + usize::from(length);
        if !(start_index..end).all(|index| cards[Rank::ALL[index]] == 3) {
            continue;
        }
        if pair_wings
            && cards.iter().any(|(rank, count)| {
                !(start_index..end).contains(&rank.index()) && count % 2 != 0
            })
        {
            continue;
        }
        let kind = if pair_wings {
            MoveKind::AirplaneWithPairs
        } else {
            MoveKind::AirplaneWithSingles
        };
        return make_move(kind, cards, Rank::ALL[start_index], length).map(Some);
    }
    Ok(None)
}

fn sole_nonzero_rank(cards: RankCounts) -> Option<Rank> {
    let mut ranks = cards
        .iter()
        .filter_map(|(rank, count)| (count != 0).then_some(rank));
    let rank = ranks.next()?;
    ranks.next().is_none().then_some(rank)
}

fn ranks_with_count(cards: RankCounts, expected: u8) -> Vec<Rank> {
    cards
        .iter()
        .filter_map(|(rank, count)| (count == expected).then_some(rank))
        .collect()
}

fn make_move(
    kind: MoveKind,
    cards: RankCounts,
    main_rank: Rank,
    chain_length: u8,
) -> Result<Move, DetectMoveError> {
    Move::new(kind, cards, main_rank.value(), chain_length).map_err(DetectMoveError::Move)
}
