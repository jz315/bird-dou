mod groups;
mod sequences;

use std::collections::BTreeSet;

use crate::movement::{beats, detect_move, DetectError, Move};
use crate::{Card, Hand, Rank};

pub fn generate_legal_moves(
    hand: &Hand,
    target: Option<&Move>,
    level: Rank,
) -> Result<Vec<Move>, DetectError> {
    if !level.is_standard() {
        return Err(DetectError::InvalidLevel(level));
    }
    let mut candidates = CandidateCards::default();
    groups::collect(hand, level, &mut candidates);
    sequences::collect(hand, level, &mut candidates);

    let mut moves: Vec<_> = candidates
        .values
        .into_iter()
        .filter_map(|cards| detect_move(&cards, level).ok())
        .filter(|movement| target.is_none_or(|current| beats(movement, current, level)))
        .collect();
    moves.sort_by_key(|movement| {
        (
            movement.len(),
            movement
                .cards()
                .iter()
                .map(|card| card.id())
                .collect::<Vec<_>>(),
        )
    });
    Ok(moves)
}

#[derive(Default)]
pub(super) struct CandidateCards {
    values: BTreeSet<Vec<Card>>,
}

impl CandidateCards {
    pub(super) fn add(&mut self, mut cards: Vec<Card>) {
        cards.sort_unstable();
        self.values.insert(cards);
    }
}

pub(super) fn natural_cards(hand: &Hand, rank: Rank, level: Rank) -> Vec<Card> {
    hand.cards()
        .filter(|card| card.rank() == rank && !card.is_wild(level))
        .collect()
}

pub(super) fn wild_cards(hand: &Hand, level: Rank) -> Vec<Card> {
    hand.cards().filter(|card| card.is_wild(level)).collect()
}
