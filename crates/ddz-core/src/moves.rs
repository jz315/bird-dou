//! Canonical move representation and stable ordering.

use std::cmp::Ordering;
use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};

use crate::{
    validate_rank_counts, CardError, RankCounts, RankId, BIG_JOKER_RANK, EMPTY_RANK_COUNTS,
    RANK_COUNT,
};

/// Sentinel used as the main rank of the canonical pass move.
pub const PASS_MAIN_RANK: RankId = 15;
const STRAIGHT_RANK_COUNT: usize = 12;

/// Canonical `DouDizhu` move categories in stable sort order.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum MoveKind {
    /// Play no cards when responding to another player's move.
    Pass = 0,
    /// One card.
    Single = 1,
    /// Two cards of one rank.
    Pair = 2,
    /// Three cards of one rank.
    Triple = 3,
    /// A triple carrying one individual card.
    TripleWithSingle = 4,
    /// A triple carrying one pair.
    TripleWithPair = 5,
    /// At least five consecutive individual ranks.
    Straight = 6,
    /// At least three consecutive pairs.
    PairStraight = 7,
    /// At least two consecutive triples without wings.
    TripleStraight = 8,
    /// Consecutive triples carrying the same number of individual cards.
    AirplaneWithSingles = 9,
    /// Consecutive triples carrying the same number of pairs.
    AirplaneWithPairs = 10,
    /// Four cards of one rank carrying two individual cards.
    FourWithTwoSingles = 11,
    /// Four cards of one rank carrying two pairs.
    FourWithTwoPairs = 12,
    /// Four cards of one rank.
    Bomb = 13,
    /// Small joker and big joker together.
    Rocket = 14,
}

impl From<MoveKind> for u8 {
    fn from(kind: MoveKind) -> Self {
        kind as Self
    }
}

impl TryFrom<u8> for MoveKind {
    type Error = MoveError;

    fn try_from(tag: u8) -> Result<Self, Self::Error> {
        match tag {
            0 => Ok(Self::Pass),
            1 => Ok(Self::Single),
            2 => Ok(Self::Pair),
            3 => Ok(Self::Triple),
            4 => Ok(Self::TripleWithSingle),
            5 => Ok(Self::TripleWithPair),
            6 => Ok(Self::Straight),
            7 => Ok(Self::PairStraight),
            8 => Ok(Self::TripleStraight),
            9 => Ok(Self::AirplaneWithSingles),
            10 => Ok(Self::AirplaneWithPairs),
            11 => Ok(Self::FourWithTwoSingles),
            12 => Ok(Self::FourWithTwoPairs),
            13 => Ok(Self::Bomb),
            14 => Ok(Self::Rocket),
            _ => Err(MoveError::UnknownKindTag { tag }),
        }
    }
}

/// A normalized move with internally consistent metadata.
///
/// `main_rank` is the repeated body rank for groups and attachments, the lowest
/// body rank for chains, [`BIG_JOKER_RANK`] for a rocket, and
/// [`PASS_MAIN_RANK`] for a pass. `chain_len` counts body ranks and is one for
/// non-chain moves.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
pub struct Move {
    kind: MoveKind,
    cards: RankCounts,
    main_rank: RankId,
    chain_len: u8,
    total_cards: u8,
}

impl Move {
    /// Construct the unique canonical pass representation.
    #[must_use]
    pub const fn pass() -> Self {
        Self {
            kind: MoveKind::Pass,
            cards: EMPTY_RANK_COUNTS,
            main_rank: PASS_MAIN_RANK,
            chain_len: 0,
            total_cards: 0,
        }
    }

    /// Construct a move from an already-classified kind and rank counts.
    ///
    /// This validates physical capacities, body metadata, chain bounds, and the
    /// kind's card-count equation. Inferring `kind` from arbitrary counts belongs
    /// to the E004 detector.
    ///
    /// # Errors
    ///
    /// Returns [`MoveError`] when any supplied component is not canonical for the
    /// declared move kind.
    pub fn try_new(
        kind: MoveKind,
        cards: RankCounts,
        main_rank: RankId,
        chain_len: u8,
    ) -> Result<Self, MoveError> {
        if kind == MoveKind::Pass {
            if cards == EMPTY_RANK_COUNTS && main_rank == PASS_MAIN_RANK && chain_len == 0 {
                return Ok(Self::pass());
            }
            return Err(MoveError::NonCanonicalPass);
        }

        validate_rank_counts(&cards).map_err(MoveError::Cards)?;
        if usize::from(main_rank) >= RANK_COUNT {
            return Err(MoveError::InvalidMainRank { main_rank });
        }

        let (minimum_chain_len, maximum_chain_len) = kind.chain_bounds();
        if !(minimum_chain_len..=maximum_chain_len).contains(&chain_len) {
            return Err(MoveError::InvalidChainLength {
                kind,
                chain_len,
                minimum: minimum_chain_len,
                maximum: maximum_chain_len,
            });
        }

        if kind.is_chain() && usize::from(main_rank) + usize::from(chain_len) > STRAIGHT_RANK_COUNT
        {
            return Err(MoveError::InvalidChainRange {
                main_rank,
                chain_len,
            });
        }

        let actual_total: u8 = cards.iter().sum();
        let expected_total = kind.expected_total(chain_len);
        if u16::from(actual_total) != expected_total {
            return Err(MoveError::InvalidTotalCards {
                kind,
                actual: actual_total,
                expected: expected_total,
            });
        }

        kind.validate_body(&cards, main_rank, chain_len)?;

        Ok(Self {
            kind,
            cards,
            main_rank,
            chain_len,
            total_cards: actual_total,
        })
    }

    /// Move category.
    #[must_use]
    pub const fn kind(&self) -> MoveKind {
        self.kind
    }

    /// Per-rank cards consumed by this move.
    #[must_use]
    pub const fn cards(&self) -> &RankCounts {
        &self.cards
    }

    /// Body rank, lowest chain rank, rocket sentinel, or pass sentinel.
    #[must_use]
    pub const fn main_rank(&self) -> RankId {
        self.main_rank
    }

    /// Number of ranks in the move body.
    #[must_use]
    pub const fn chain_len(&self) -> u8 {
        self.chain_len
    }

    /// Number of physical cards consumed by the move.
    #[must_use]
    pub const fn total_cards(&self) -> u8 {
        self.total_cards
    }
}

impl MoveKind {
    const fn chain_bounds(self) -> (u8, u8) {
        match self {
            Self::Pass => (0, 0),
            Self::Straight => (5, 12),
            Self::PairStraight => (3, 12),
            Self::TripleStraight | Self::AirplaneWithSingles | Self::AirplaneWithPairs => (2, 12),
            Self::Single
            | Self::Pair
            | Self::Triple
            | Self::TripleWithSingle
            | Self::TripleWithPair
            | Self::FourWithTwoSingles
            | Self::FourWithTwoPairs
            | Self::Bomb
            | Self::Rocket => (1, 1),
        }
    }

    fn expected_total(self, chain_len: u8) -> u16 {
        let chain_len = u16::from(chain_len);
        match self {
            Self::Pass => 0,
            Self::Single => 1,
            Self::Pair | Self::Rocket => 2,
            Self::Triple => 3,
            Self::TripleWithSingle | Self::Bomb => 4,
            Self::TripleWithPair => 5,
            Self::Straight => chain_len,
            Self::PairStraight => 2 * chain_len,
            Self::TripleStraight => 3 * chain_len,
            Self::AirplaneWithSingles => 4 * chain_len,
            Self::AirplaneWithPairs => 5 * chain_len,
            Self::FourWithTwoSingles => 6,
            Self::FourWithTwoPairs => 8,
        }
    }

    const fn is_chain(self) -> bool {
        matches!(
            self,
            Self::Straight
                | Self::PairStraight
                | Self::TripleStraight
                | Self::AirplaneWithSingles
                | Self::AirplaneWithPairs
        )
    }

    fn validate_body(
        self,
        cards: &RankCounts,
        main_rank: RankId,
        chain_len: u8,
    ) -> Result<(), MoveError> {
        if self == Self::Rocket {
            if main_rank != BIG_JOKER_RANK {
                return Err(MoveError::UnexpectedMainRank {
                    kind: self,
                    actual: main_rank,
                    expected: BIG_JOKER_RANK,
                });
            }
            if cards[13] != 1 || cards[14] != 1 {
                return Err(MoveError::InvalidRocket);
            }
            return Ok(());
        }

        let expected_body_count = match self {
            Self::Single | Self::Straight => 1,
            Self::Pair | Self::PairStraight => 2,
            Self::Triple
            | Self::TripleWithSingle
            | Self::TripleWithPair
            | Self::TripleStraight
            | Self::AirplaneWithSingles
            | Self::AirplaneWithPairs => 3,
            Self::FourWithTwoSingles | Self::FourWithTwoPairs | Self::Bomb => 4,
            Self::Pass | Self::Rocket => unreachable!("handled before body validation"),
        };

        for rank_id in main_rank..main_rank + chain_len {
            let actual = cards[usize::from(rank_id)];
            if actual != expected_body_count {
                return Err(MoveError::BodyCountMismatch {
                    kind: self,
                    rank_id,
                    actual,
                    expected: expected_body_count,
                });
            }
        }

        if matches!(
            self,
            Self::TripleWithPair | Self::AirplaneWithPairs | Self::FourWithTwoPairs
        ) {
            let body_end = main_rank + chain_len;
            for (rank_id, &count) in (0_u8..).zip(cards.iter()) {
                let is_body = (main_rank..body_end).contains(&rank_id);
                if !is_body && count % 2 != 0 {
                    return Err(MoveError::InvalidPairAttachment {
                        kind: self,
                        rank_id,
                        count,
                    });
                }
            }
        }
        Ok(())
    }
}

impl Ord for Move {
    fn cmp(&self, other: &Self) -> Ordering {
        self.kind
            .cmp(&other.kind)
            .then_with(|| self.total_cards.cmp(&other.total_cards))
            .then_with(|| self.chain_len.cmp(&other.chain_len))
            .then_with(|| self.main_rank.cmp(&other.main_rank))
            .then_with(|| self.cards.cmp(&other.cards))
    }
}

impl PartialOrd for Move {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MoveWire {
    kind: MoveKind,
    cards: RankCounts,
    main_rank: RankId,
    chain_len: u8,
    total_cards: u8,
}

impl<'de> Deserialize<'de> for Move {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MoveWire::deserialize(deserializer)?;
        let value = Self::try_new(wire.kind, wire.cards, wire.main_rank, wire.chain_len)
            .map_err(D::Error::custom)?;
        if value.total_cards != wire.total_cards {
            return Err(D::Error::custom(MoveError::SerializedTotalMismatch {
                encoded: wire.total_cards,
                computed: value.total_cards,
            }));
        }
        Ok(value)
    }
}

/// Errors produced while constructing or decoding a normalized move.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MoveError {
    /// Rank counts cannot be represented by the physical deck.
    Cards(CardError),
    /// A numeric move-kind tag is not assigned in schema version 1.
    UnknownKindTag {
        /// Rejected numeric tag.
        tag: u8,
    },
    /// A pass used cards or non-sentinel metadata.
    NonCanonicalPass,
    /// A non-pass main rank fell outside `0..=14`.
    InvalidMainRank {
        /// Rejected main rank.
        main_rank: RankId,
    },
    /// A kind used a chain length outside its structural bounds.
    InvalidChainLength {
        /// Declared kind.
        kind: MoveKind,
        /// Rejected chain length.
        chain_len: u8,
        /// Inclusive minimum.
        minimum: u8,
        /// Inclusive maximum before applying rank-range limits.
        maximum: u8,
    },
    /// A chain extended beyond A into 2 or jokers.
    InvalidChainRange {
        /// Lowest body rank.
        main_rank: RankId,
        /// Number of body ranks.
        chain_len: u8,
    },
    /// Rank counts did not contain the required number of cards.
    InvalidTotalCards {
        /// Declared kind.
        kind: MoveKind,
        /// Count present in `cards`.
        actual: u8,
        /// Count required by the kind and chain length.
        expected: u16,
    },
    /// A body rank did not carry the multiplicity required by the kind.
    BodyCountMismatch {
        /// Declared kind.
        kind: MoveKind,
        /// Body rank containing the mismatch.
        rank_id: RankId,
        /// Actual multiplicity.
        actual: u8,
        /// Required multiplicity.
        expected: u8,
    },
    /// A move requiring pair attachments contained an unmatched card.
    InvalidPairAttachment {
        /// Declared kind.
        kind: MoveKind,
        /// Attachment rank with odd multiplicity.
        rank_id: RankId,
        /// Rejected multiplicity.
        count: u8,
    },
    /// A kind requires a fixed sentinel main rank.
    UnexpectedMainRank {
        /// Declared kind.
        kind: MoveKind,
        /// Supplied main rank.
        actual: RankId,
        /// Required main rank.
        expected: RankId,
    },
    /// A rocket did not contain exactly one of each joker.
    InvalidRocket,
    /// Serialized redundant total did not match the rank counts.
    SerializedTotalMismatch {
        /// Total carried by serialized data.
        encoded: u8,
        /// Total recomputed from rank counts.
        computed: u8,
    },
}

impl From<CardError> for MoveError {
    fn from(error: CardError) -> Self {
        Self::Cards(error)
    }
}

impl Display for MoveError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cards(error) => Display::fmt(error, formatter),
            Self::UnknownKindTag { tag } => write!(formatter, "unknown move-kind tag {tag}"),
            Self::NonCanonicalPass => write!(
                formatter,
                "pass must use zero cards, rank sentinel 15, and chain length 0"
            ),
            Self::InvalidMainRank { main_rank } => {
                write!(
                    formatter,
                    "non-pass main rank {main_rank} is outside 0..=14"
                )
            }
            Self::InvalidChainLength {
                kind,
                chain_len,
                minimum,
                maximum,
            } => write!(
                formatter,
                "move {kind:?} has chain length {chain_len}; expected {minimum}..={maximum}"
            ),
            Self::InvalidChainRange {
                main_rank,
                chain_len,
            } => write!(
                formatter,
                "chain starting at rank {main_rank} with length {chain_len} extends beyond A"
            ),
            Self::InvalidTotalCards {
                kind,
                actual,
                expected,
            } => write!(
                formatter,
                "move {kind:?} has {actual} cards; expected {expected}"
            ),
            Self::BodyCountMismatch {
                kind,
                rank_id,
                actual,
                expected,
            } => write!(
                formatter,
                "move {kind:?} body rank {rank_id} has count {actual}; expected {expected}"
            ),
            Self::InvalidPairAttachment {
                kind,
                rank_id,
                count,
            } => write!(
                formatter,
                "move {kind:?} attachment rank {rank_id} has odd count {count}"
            ),
            Self::UnexpectedMainRank {
                kind,
                actual,
                expected,
            } => write!(
                formatter,
                "move {kind:?} uses main rank {actual}; expected {expected}"
            ),
            Self::InvalidRocket => write!(formatter, "rocket must contain exactly both jokers"),
            Self::SerializedTotalMismatch { encoded, computed } => write!(
                formatter,
                "serialized total_cards is {encoded}, but rank counts contain {computed}"
            ),
        }
    }
}

impl Error for MoveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Cards(error) => Some(error),
            Self::UnknownKindTag { .. }
            | Self::NonCanonicalPass
            | Self::InvalidMainRank { .. }
            | Self::InvalidChainLength { .. }
            | Self::InvalidChainRange { .. }
            | Self::InvalidTotalCards { .. }
            | Self::BodyCountMismatch { .. }
            | Self::InvalidPairAttachment { .. }
            | Self::UnexpectedMainRank { .. }
            | Self::InvalidRocket
            | Self::SerializedTotalMismatch { .. } => None,
        }
    }
}
