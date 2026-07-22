//! Physical-card and rank-count representations.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// Number of physical cards in the standard `DouDizhu` deck.
pub const CARD_COUNT: usize = 54;
/// Number of ordered ranks: `3..A, 2, small joker, big joker`.
pub const RANK_COUNT: usize = 15;
/// Rank ID assigned to the small joker.
pub const SMALL_JOKER_RANK: RankId = 13;
/// Rank ID assigned to the big joker.
pub const BIG_JOKER_RANK: RankId = 14;
/// Physical card ID assigned to the small joker.
pub const SMALL_JOKER_CARD: CardId = 52;
/// Physical card ID assigned to the big joker.
pub const BIG_JOKER_CARD: CardId = 53;

/// Physical card identifier in the inclusive range `0..=53`.
pub type CardId = u8;
/// Ordered rank identifier in the inclusive range `0..=14`.
pub type RankId = u8;
/// Player seat identifier. Valid seats are defined with game state in E007.
pub type Seat = u8;
/// Per-rank card multiplicities ordered as `3..A, 2, small joker, big joker`.
pub type RankCounts = [u8; RANK_COUNT];
/// Rank counts for an empty hand.
pub const EMPTY_RANK_COUNTS: RankCounts = [0; RANK_COUNT];

/// Errors produced while validating or converting card representations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CardError {
    /// A physical card ID fell outside `0..=53`.
    InvalidCardId {
        /// Rejected card ID.
        card_id: CardId,
    },
    /// A rank ID fell outside `0..=14`.
    InvalidRankId {
        /// Rejected rank ID.
        rank_id: RankId,
    },
    /// The same physical card appeared more than once in a card list.
    DuplicateCardId {
        /// Repeated card ID.
        card_id: CardId,
    },
    /// A rank count exceeded the number of physical cards available for that rank.
    TooManyCardsForRank {
        /// Rank containing the invalid count.
        rank_id: RankId,
        /// Rejected count.
        count: u8,
        /// Maximum physical count for the rank.
        maximum: u8,
    },
}

impl Display for CardError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCardId { card_id } => {
                write!(formatter, "card ID {card_id} is outside 0..=53")
            }
            Self::InvalidRankId { rank_id } => {
                write!(formatter, "rank ID {rank_id} is outside 0..=14")
            }
            Self::DuplicateCardId { card_id } => {
                write!(
                    formatter,
                    "physical card ID {card_id} appears more than once"
                )
            }
            Self::TooManyCardsForRank {
                rank_id,
                count,
                maximum,
            } => write!(
                formatter,
                "rank ID {rank_id} has count {count}, but its maximum is {maximum}"
            ),
        }
    }
}

impl Error for CardError {}

/// Convert a physical card ID to its ordered rank ID.
///
/// IDs `0..=51` are rank-major with four physical cards per rank. IDs 52 and
/// 53 are the small and big jokers respectively.
///
/// # Errors
///
/// Returns [`CardError::InvalidCardId`] when `card_id` is greater than 53.
pub const fn card_id_to_rank(card_id: CardId) -> Result<RankId, CardError> {
    match card_id {
        0..=51 => Ok(card_id / 4),
        SMALL_JOKER_CARD => Ok(SMALL_JOKER_RANK),
        BIG_JOKER_CARD => Ok(BIG_JOKER_RANK),
        _ => Err(CardError::InvalidCardId { card_id }),
    }
}

/// Return the physical-card capacity of a rank.
///
/// # Errors
///
/// Returns [`CardError::InvalidRankId`] when `rank_id` is greater than 14.
pub const fn max_count_for_rank(rank_id: RankId) -> Result<u8, CardError> {
    match rank_id {
        0..=12 => Ok(4),
        SMALL_JOKER_RANK | BIG_JOKER_RANK => Ok(1),
        _ => Err(CardError::InvalidRankId { rank_id }),
    }
}

/// Return all physical card IDs belonging to one rank in stable ascending order.
///
/// Standard ranks contain four IDs; joker ranks contain one.
///
/// # Errors
///
/// Returns [`CardError::InvalidRankId`] when `rank_id` is greater than 14.
pub fn rank_to_card_ids(rank_id: RankId) -> Result<Vec<CardId>, CardError> {
    match rank_id {
        0..=12 => {
            let first = rank_id * 4;
            Ok(vec![first, first + 1, first + 2, first + 3])
        }
        SMALL_JOKER_RANK => Ok(vec![SMALL_JOKER_CARD]),
        BIG_JOKER_RANK => Ok(vec![BIG_JOKER_CARD]),
        _ => Err(CardError::InvalidRankId { rank_id }),
    }
}

/// Collapse unique physical cards into per-rank counts.
///
/// # Errors
///
/// Returns [`CardError::InvalidCardId`] for an out-of-range ID or
/// [`CardError::DuplicateCardId`] if a physical card is repeated.
pub fn cards_to_rank_counts(cards: &[CardId]) -> Result<RankCounts, CardError> {
    let mut counts = EMPTY_RANK_COUNTS;
    let mut seen = [false; CARD_COUNT];

    for &card_id in cards {
        let rank_id = card_id_to_rank(card_id)?;
        let card_index = usize::from(card_id);
        if seen[card_index] {
            return Err(CardError::DuplicateCardId { card_id });
        }
        seen[card_index] = true;
        counts[usize::from(rank_id)] += 1;
    }

    Ok(counts)
}

/// Validate that every count can be represented by the physical deck.
///
/// # Errors
///
/// Returns [`CardError::TooManyCardsForRank`] when a standard-rank count exceeds
/// four or a joker count exceeds one.
pub fn validate_rank_counts(rank_counts: &RankCounts) -> Result<(), CardError> {
    for (rank_id, &count) in (0_u8..).zip(rank_counts.iter()) {
        let maximum = max_count_for_rank(rank_id)?;
        if count > maximum {
            return Err(CardError::TooManyCardsForRank {
                rank_id,
                count,
                maximum,
            });
        }
    }

    Ok(())
}

/// Expand per-rank counts to a canonical set of unique physical card IDs.
///
/// When fewer than four cards of a standard rank are requested, the lowest
/// physical IDs for that rank are selected. The result is rank-major and stable.
///
/// # Errors
///
/// Returns [`CardError::TooManyCardsForRank`] when a standard-rank count exceeds
/// four or a joker count exceeds one.
pub fn rank_counts_to_card_ids(rank_counts: &RankCounts) -> Result<Vec<CardId>, CardError> {
    let mut cards = Vec::with_capacity(rank_counts.iter().map(|&count| usize::from(count)).sum());
    validate_rank_counts(rank_counts)?;

    for (rank_id, &count) in (0_u8..).zip(rank_counts.iter()) {
        cards.extend(
            rank_to_card_ids(rank_id)?
                .into_iter()
                .take(usize::from(count)),
        );
    }

    Ok(cards)
}
