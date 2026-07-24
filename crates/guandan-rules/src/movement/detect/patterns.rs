use crate::movement::detect::SplitCards;
use crate::movement::sequence::{high_rank, windows};
use crate::movement::strength::rank_strength;
use crate::movement::MoveKind;
use crate::{Rank, Suit};

pub(super) fn detect_four_jokers(split: &SplitCards) -> Option<MoveKind> {
    (split.wild_count == 0
        && split.natural.len() == 4
        && split
            .natural
            .iter()
            .all(|card| matches!(card.rank(), Rank::SmallJoker | Rank::BigJoker)))
    .then_some(MoveKind::FourJokers)
}

pub(super) fn detect_bomb(split: &SplitCards, card_count: usize, level: Rank) -> Option<MoveKind> {
    if !(4..=10).contains(&card_count) || split.counts.len() > 1 {
        return None;
    }
    let rank = split.counts.keys().next().copied().unwrap_or(level);
    (rank.is_standard() && split.natural.len() + split.wild_count == card_count).then_some(
        MoveKind::Bomb {
            rank,
            size: u8::try_from(card_count).expect("bomb size is at most ten"),
        },
    )
}

pub(super) fn detect_straight_flush(split: &SplitCards, card_count: usize) -> Option<MoveKind> {
    if card_count != 5 {
        return None;
    }
    Suit::ALL.iter().rev().find_map(|suit| {
        fits_sequence(split, 1, Some(*suit), 5).map(|(sequence, high)| MoveKind::StraightFlush {
            suit: *suit,
            sequence,
            high,
        })
    })
}

pub(super) fn detect_normal(
    split: &SplitCards,
    card_count: usize,
    level: Rank,
) -> Option<MoveKind> {
    match card_count {
        1 => detect_group(split, 1, level).map(|rank| MoveKind::Single { rank }),
        2 => detect_group(split, 2, level).map(|rank| MoveKind::Pair { rank }),
        3 => detect_group(split, 3, level).map(|rank| MoveKind::Triple { rank }),
        5 => detect_full_house(split, level).or_else(|| {
            fits_sequence(split, 1, None, 5)
                .map(|(sequence, high)| MoveKind::Straight { sequence, high })
        }),
        6 => fits_sequence(split, 2, None, 3)
            .map(|(sequence, high)| MoveKind::PairStraight { sequence, high })
            .or_else(|| {
                fits_sequence(split, 3, None, 2)
                    .map(|(sequence, high)| MoveKind::TripleStraight { sequence, high })
            }),
        _ => None,
    }
}

fn detect_group(split: &SplitCards, size: usize, level: Rank) -> Option<Rank> {
    Rank::STANDARD
        .into_iter()
        .chain([Rank::SmallJoker, Rank::BigJoker])
        .filter(|rank| {
            let natural = split.counts.get(rank).copied().unwrap_or_default();
            let usable_wilds = if rank.is_standard() {
                split.wild_count
            } else {
                0
            };
            split.counts.keys().all(|present| present == rank) && natural + usable_wilds == size
        })
        .max_by_key(|rank| rank_strength(*rank, level))
}

fn detect_full_house(split: &SplitCards, level: Rank) -> Option<MoveKind> {
    Rank::STANDARD
        .into_iter()
        .filter_map(|triple_rank| {
            Rank::STANDARD
                .into_iter()
                .chain([Rank::SmallJoker, Rank::BigJoker])
                .filter(|pair_rank| *pair_rank != triple_rank)
                .find_map(|pair_rank| {
                    fits_rank_counts(split, &[(triple_rank, 3), (pair_rank, 2)])
                        .then_some(MoveKind::FullHouse { triple_rank })
                })
        })
        .max_by_key(|kind| match kind {
            MoveKind::FullHouse { triple_rank } => rank_strength(*triple_rank, level),
            _ => unreachable!("only full houses are produced"),
        })
}

fn fits_rank_counts(split: &SplitCards, required: &[(Rank, usize)]) -> bool {
    if split.counts.keys().any(|rank| {
        !required
            .iter()
            .any(|(required_rank, _)| required_rank == rank)
    }) {
        return false;
    }
    let mut missing = 0;
    for (rank, target) in required {
        let actual = split.counts.get(rank).copied().unwrap_or_default();
        if actual > *target {
            return false;
        }
        let deficit = target - actual;
        if !rank.is_standard() && deficit > 0 {
            return false;
        }
        missing += deficit;
    }
    missing == split.wild_count
}

fn fits_sequence(
    split: &SplitCards,
    copies_per_rank: usize,
    suit: Option<Suit>,
    width: usize,
) -> Option<(u8, Rank)> {
    if split
        .natural
        .iter()
        .any(|card| !card.rank().is_standard() || suit.is_some_and(|s| card.suit() != Some(s)))
    {
        return None;
    }
    windows(width).rev().find_map(|(index, ranks)| {
        let required: Vec<_> = ranks
            .iter()
            .copied()
            .map(|rank| (rank, copies_per_rank))
            .collect();
        fits_rank_counts(split, &required).then_some((index, high_rank(ranks)))
    })
}
