use std::cmp::Ordering;

use crate::movement::strength::rank_strength;
use crate::movement::{Move, MoveKind};
use crate::Rank;

pub fn beats(candidate: &Move, target: &Move, level: Rank) -> bool {
    let candidate_kind = *candidate.kind();
    let target_kind = *target.kind();
    let candidate_tier = tier(candidate_kind);
    let target_tier = tier(target_kind);
    match candidate_tier.cmp(&target_tier) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => compare_same_tier(candidate_kind, target_kind, level),
    }
}

fn tier(kind: MoveKind) -> u8 {
    match kind {
        MoveKind::FourJokers => 5,
        MoveKind::Bomb { size, .. } if size >= 6 => 4,
        MoveKind::StraightFlush { .. } => 3,
        MoveKind::Bomb { size: 5, .. } => 2,
        MoveKind::Bomb { size: 4, .. } => 1,
        _ => 0,
    }
}

fn compare_same_tier(candidate: MoveKind, target: MoveKind, level: Rank) -> bool {
    match (candidate, target) {
        (MoveKind::FourJokers, MoveKind::FourJokers) => false,
        (
            MoveKind::Bomb {
                rank: candidate_rank,
                size: candidate_size,
            },
            MoveKind::Bomb {
                rank: target_rank,
                size: target_size,
            },
        ) => {
            candidate_size > target_size
                || (candidate_size == target_size
                    && rank_strength(candidate_rank, level) > rank_strength(target_rank, level))
        }
        (
            MoveKind::StraightFlush {
                sequence: candidate_sequence,
                ..
            },
            MoveKind::StraightFlush {
                sequence: target_sequence,
                ..
            },
        ) => candidate_sequence > target_sequence,
        _ if tier(candidate) == 0 && tier(target) == 0 => compare_normal(candidate, target, level),
        _ => false,
    }
}

fn compare_normal(candidate: MoveKind, target: MoveKind, level: Rank) -> bool {
    match (candidate, target) {
        (MoveKind::Single { rank: left }, MoveKind::Single { rank: right })
        | (MoveKind::Pair { rank: left }, MoveKind::Pair { rank: right })
        | (MoveKind::Triple { rank: left }, MoveKind::Triple { rank: right })
        | (MoveKind::FullHouse { triple_rank: left }, MoveKind::FullHouse { triple_rank: right }) => {
            rank_strength(left, level) > rank_strength(right, level)
        }
        (
            MoveKind::Straight { sequence: left, .. },
            MoveKind::Straight {
                sequence: right, ..
            },
        )
        | (
            MoveKind::PairStraight { sequence: left, .. },
            MoveKind::PairStraight {
                sequence: right, ..
            },
        )
        | (
            MoveKind::TripleStraight { sequence: left, .. },
            MoveKind::TripleStraight {
                sequence: right, ..
            },
        ) => left > right,
        _ => false,
    }
}
