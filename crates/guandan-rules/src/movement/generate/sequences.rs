use crate::movement::generate::{natural_cards, wild_cards, CandidateCards};
use crate::movement::sequence::windows;
use crate::{Card, Hand, Rank, Suit};

pub(super) fn collect(hand: &Hand, level: Rank, output: &mut CandidateCards) {
    let wilds = wild_cards(hand, level);
    for (_, ranks) in windows(5) {
        collect_sequence(hand, level, ranks, 1, None, &wilds, output);
        for suit in Suit::ALL {
            collect_sequence(hand, level, ranks, 1, Some(suit), &wilds, output);
        }
    }
    for (_, ranks) in windows(3) {
        collect_sequence(hand, level, ranks, 2, None, &wilds, output);
    }
    for (_, ranks) in windows(2) {
        collect_sequence(hand, level, ranks, 3, None, &wilds, output);
    }
}

fn collect_sequence(
    hand: &Hand,
    level: Rank,
    ranks: &[Rank],
    copies: usize,
    suit: Option<Suit>,
    wilds: &[Card],
    output: &mut CandidateCards,
) {
    let mut cards = Vec::with_capacity(ranks.len() * copies);
    let mut missing = 0;
    for rank in ranks {
        let natural: Vec<Card> = natural_cards(hand, *rank, level)
            .into_iter()
            .filter(|card| suit.is_none_or(|required| card.suit() == Some(required)))
            .collect();
        let selected = natural.len().min(copies);
        cards.extend_from_slice(&natural[..selected]);
        missing += copies - selected;
    }
    if missing <= wilds.len() {
        cards.extend_from_slice(&wilds[..missing]);
        output.add(cards);
    }
}
