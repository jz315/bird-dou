mod patterns;

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::movement::Move;
use crate::{Card, Rank};

use patterns::{detect_bomb, detect_four_jokers, detect_normal, detect_straight_flush};

pub fn detect_move(cards: &[Card], level: Rank) -> Result<Move, DetectError> {
    validate_input(cards, level)?;
    let split = SplitCards::new(cards, level);
    let kind = detect_four_jokers(&split)
        .or_else(|| detect_bomb(&split, cards.len(), level))
        .or_else(|| detect_straight_flush(&split, cards.len()))
        .or_else(|| detect_normal(&split, cards.len(), level))
        .ok_or(DetectError::UnsupportedPattern {
            card_count: cards.len(),
        })?;
    Ok(Move::new(cards.to_vec(), kind))
}

fn validate_input(cards: &[Card], level: Rank) -> Result<(), DetectError> {
    if cards.is_empty() {
        return Err(DetectError::Empty);
    }
    if !level.is_standard() {
        return Err(DetectError::InvalidLevel(level));
    }
    let mut unique = BTreeSet::new();
    for card in cards {
        if !unique.insert(*card) {
            return Err(DetectError::DuplicateCard(*card));
        }
    }
    Ok(())
}

pub(super) struct SplitCards {
    pub(super) natural: Vec<Card>,
    pub(super) wild_count: usize,
    pub(super) counts: BTreeMap<Rank, usize>,
}

impl SplitCards {
    fn new(cards: &[Card], level: Rank) -> Self {
        let mut natural = Vec::new();
        let mut wild_count = 0;
        let mut counts = BTreeMap::new();
        for card in cards {
            if card.is_wild(level) {
                wild_count += 1;
            } else {
                natural.push(*card);
                *counts.entry(card.rank()).or_default() += 1;
            }
        }
        Self {
            natural,
            wild_count,
            counts,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetectError {
    Empty,
    InvalidLevel(Rank),
    DuplicateCard(Card),
    UnsupportedPattern { card_count: usize },
}

impl Display for DetectError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => formatter.write_str("a move must contain at least one card"),
            Self::InvalidLevel(rank) => write!(formatter, "{rank:?} cannot be a level rank"),
            Self::DuplicateCard(card) => {
                write!(
                    formatter,
                    "physical card {} appears more than once",
                    card.id()
                )
            }
            Self::UnsupportedPattern { card_count } => {
                write!(
                    formatter,
                    "{card_count} cards do not form a legal Guandan move"
                )
            }
        }
    }
}

impl Error for DetectError {}
