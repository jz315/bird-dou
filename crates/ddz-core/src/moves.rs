use std::cmp::Ordering;
use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};

use crate::{Rank, RankCounts, RankCountsError, BIG_JOKER, EMPTY_RANK_COUNTS, SMALL_JOKER};

pub const PASS_MAIN_RANK: u8 = 15;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum MoveKind {
    Pass = 0,
    Single = 1,
    Pair = 2,
    Triple = 3,
    TripleWithSingle = 4,
    TripleWithPair = 5,
    Straight = 6,
    PairStraight = 7,
    TripleStraight = 8,
    AirplaneWithSingles = 9,
    AirplaneWithPairs = 10,
    FourWithTwoSingles = 11,
    FourWithTwoPairs = 12,
    Bomb = 13,
    Rocket = 14,
}

/// Canonical structural move. Platform-specific attachment restrictions belong in `ddz-rules`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
pub struct Move {
    kind: MoveKind,
    cards: RankCounts,
    main_rank: u8,
    chain_len: u8,
    total_cards: u8,
}

impl Move {
    pub const fn pass() -> Self {
        Self {
            kind: MoveKind::Pass,
            cards: EMPTY_RANK_COUNTS,
            main_rank: PASS_MAIN_RANK,
            chain_len: 0,
            total_cards: 0,
        }
    }

    pub fn new(
        kind: MoveKind,
        cards: RankCounts,
        main_rank: u8,
        chain_len: u8,
    ) -> Result<Self, MoveError> {
        if kind == MoveKind::Pass {
            if cards.is_empty() && main_rank == PASS_MAIN_RANK && chain_len == 0 {
                return Ok(Self::pass());
            }
            return Err(MoveError::NonCanonicalPass);
        }
        if main_rank >= crate::RANK_COUNT as u8 {
            return Err(MoveError::InvalidMainRank { main_rank });
        }

        let (minimum, maximum) = kind.chain_bounds();
        if !(minimum..=maximum).contains(&chain_len) {
            return Err(MoveError::InvalidChainLength {
                kind,
                chain_len,
                minimum,
                maximum,
            });
        }
        let chain_end = main_rank
            .checked_add(chain_len)
            .ok_or(MoveError::InvalidChainRange {
                main_rank,
                chain_len,
            })?;
        if kind.is_chain() && chain_end > Rank::Two.value() {
            return Err(MoveError::InvalidChainRange {
                main_rank,
                chain_len,
            });
        }

        let actual = u8::try_from(cards.card_count())
            .map_err(|_| MoveError::TooManyCards { count: cards.card_count() })?;
        let expected = kind.expected_total(chain_len);
        if actual != expected {
            return Err(MoveError::InvalidTotal {
                kind,
                actual,
                expected,
            });
        }

        validate_body(kind, cards, main_rank, chain_len)?;

        Ok(Self {
            kind,
            cards,
            main_rank,
            chain_len,
            total_cards: actual,
        })
    }

    pub const fn kind(self) -> MoveKind {
        self.kind
    }

    pub const fn cards(self) -> RankCounts {
        self.cards
    }

    pub const fn main_rank(self) -> u8 {
        self.main_rank
    }

    pub const fn chain_len(self) -> u8 {
        self.chain_len
    }

    pub const fn total_cards(self) -> u8 {
        self.total_cards
    }

    pub const fn is_pass(self) -> bool {
        matches!(self.kind, MoveKind::Pass)
    }

    pub const fn is_bomb_like(self) -> bool {
        matches!(self.kind, MoveKind::Bomb | MoveKind::Rocket)
    }
}

impl MoveKind {
    const fn chain_bounds(self) -> (u8, u8) {
        match self {
            Self::Pass => (0, 0),
            Self::Straight => (5, 12),
            Self::PairStraight => (3, 12),
            Self::TripleStraight | Self::AirplaneWithSingles | Self::AirplaneWithPairs => (2, 12),
            _ => (1, 1),
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

    const fn expected_total(self, chain_len: u8) -> u8 {
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
}

fn validate_body(
    kind: MoveKind,
    cards: RankCounts,
    main_rank: u8,
    chain_len: u8,
) -> Result<(), MoveError> {
    if kind == MoveKind::Rocket {
        if main_rank != BIG_JOKER.value()
            || cards[SMALL_JOKER] != 1
            || cards[BIG_JOKER] != 1
        {
            return Err(MoveError::InvalidRocket);
        }
        return Ok(());
    }

    let body_count = match kind {
        MoveKind::Single | MoveKind::Straight => 1,
        MoveKind::Pair | MoveKind::PairStraight => 2,
        MoveKind::Triple
        | MoveKind::TripleWithSingle
        | MoveKind::TripleWithPair
        | MoveKind::TripleStraight
        | MoveKind::AirplaneWithSingles
        | MoveKind::AirplaneWithPairs => 3,
        MoveKind::FourWithTwoSingles | MoveKind::FourWithTwoPairs | MoveKind::Bomb => 4,
        MoveKind::Pass | MoveKind::Rocket => unreachable!(),
    };

    for rank_value in main_rank..chain_end(main_rank, chain_len)? {
        let rank = Rank::try_from(rank_value).map_err(MoveError::Counts)?;
        let actual = cards[rank];
        if actual != body_count {
            return Err(MoveError::BodyCount {
                rank,
                actual,
                expected: body_count,
            });
        }
    }

    if matches!(
        kind,
        MoveKind::TripleWithPair | MoveKind::AirplaneWithPairs | MoveKind::FourWithTwoPairs
    ) {
        let body_end = main_rank + chain_len;
        for (rank, count) in cards.iter() {
            let in_body = (main_rank..body_end).contains(&rank.value());
            if !in_body && count % 2 != 0 {
                return Err(MoveError::OddPairAttachment { rank, count });
            }
        }
    }

    Ok(())
}

fn chain_end(main_rank: u8, chain_len: u8) -> Result<u8, MoveError> {
    main_rank
        .checked_add(chain_len)
        .ok_or(MoveError::InvalidChainRange {
            main_rank,
            chain_len,
        })
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
    main_rank: u8,
    chain_len: u8,
    total_cards: u8,
}

impl<'de> Deserialize<'de> for Move {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MoveWire::deserialize(deserializer)?;
        let value = Self::new(wire.kind, wire.cards, wire.main_rank, wire.chain_len)
            .map_err(D::Error::custom)?;
        if value.total_cards != wire.total_cards {
            return Err(D::Error::custom(MoveError::SerializedTotal {
                encoded: wire.total_cards,
                computed: value.total_cards,
            }));
        }
        Ok(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MoveError {
    Counts(RankCountsError),
    NonCanonicalPass,
    InvalidMainRank {
        main_rank: u8,
    },
    InvalidChainLength {
        kind: MoveKind,
        chain_len: u8,
        minimum: u8,
        maximum: u8,
    },
    InvalidChainRange {
        main_rank: u8,
        chain_len: u8,
    },
    TooManyCards {
        count: u16,
    },
    InvalidTotal {
        kind: MoveKind,
        actual: u8,
        expected: u8,
    },
    BodyCount {
        rank: Rank,
        actual: u8,
        expected: u8,
    },
    OddPairAttachment {
        rank: Rank,
        count: u8,
    },
    InvalidRocket,
    SerializedTotal {
        encoded: u8,
        computed: u8,
    },
}

impl Display for MoveError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Counts(error) => Display::fmt(error, formatter),
            Self::NonCanonicalPass => write!(formatter, "pass move contains cards or metadata"),
            Self::InvalidMainRank { main_rank } => {
                write!(formatter, "move main rank {main_rank} is outside 0..=14")
            }
            Self::InvalidChainLength {
                kind,
                chain_len,
                minimum,
                maximum,
            } => write!(
                formatter,
                "{kind:?} chain length {chain_len} is outside {minimum}..={maximum}"
            ),
            Self::InvalidChainRange {
                main_rank,
                chain_len,
            } => write!(
                formatter,
                "chain from rank {main_rank} with length {chain_len} reaches rank 2 or joker"
            ),
            Self::TooManyCards { count } => write!(formatter, "move contains {count} cards"),
            Self::InvalidTotal {
                kind,
                actual,
                expected,
            } => write!(
                formatter,
                "{kind:?} contains {actual} cards; canonical total is {expected}"
            ),
            Self::BodyCount {
                rank,
                actual,
                expected,
            } => write!(
                formatter,
                "body rank {rank:?} has {actual} cards; expected {expected}"
            ),
            Self::OddPairAttachment { rank, count } => write!(
                formatter,
                "pair-wing move has odd attachment count {count} at {rank:?}"
            ),
            Self::InvalidRocket => write!(formatter, "rocket must contain both jokers exactly once"),
            Self::SerializedTotal { encoded, computed } => write!(
                formatter,
                "serialized move total {encoded} differs from computed total {computed}"
            ),
        }
    }
}

impl Error for MoveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Counts(error) => Some(error),
            _ => None,
        }
    }
}
