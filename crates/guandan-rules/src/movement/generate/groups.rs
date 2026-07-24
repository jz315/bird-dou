use crate::movement::generate::{natural_cards, wild_cards, CandidateCards};
use crate::{Card, Hand, Rank};

pub(super) fn collect(hand: &Hand, level: Rank, output: &mut CandidateCards) {
    for card in hand.cards() {
        output.add(vec![card]);
    }
    let wilds = wild_cards(hand, level);
    for rank in Rank::STANDARD {
        let natural = natural_cards(hand, rank, level);
        collect_rank_groups(&natural, &wilds, output);
    }
    for rank in [Rank::SmallJoker, Rank::BigJoker] {
        let natural = natural_cards(hand, rank, level);
        for size in 2..=3 {
            if natural.len() >= size {
                output.add(natural[..size].to_vec());
            }
        }
    }
    collect_full_houses(hand, level, &wilds, output);
    collect_four_jokers(hand, output);
}

fn collect_rank_groups(natural: &[Card], wilds: &[Card], output: &mut CandidateCards) {
    for size in 2..=10 {
        for wild_count in 0..=wilds.len().min(size) {
            let natural_count = size - wild_count;
            if natural_count <= natural.len() {
                let mut cards = natural[..natural_count].to_vec();
                cards.extend_from_slice(&wilds[..wild_count]);
                output.add(cards);
            }
        }
    }
}

fn collect_full_houses(hand: &Hand, level: Rank, wilds: &[Card], output: &mut CandidateCards) {
    let pair_ranks = Rank::STANDARD
        .into_iter()
        .chain([Rank::SmallJoker, Rank::BigJoker]);
    for triple_rank in Rank::STANDARD {
        let triple_natural = natural_cards(hand, triple_rank, level);
        for pair_rank in pair_ranks.clone().filter(|rank| *rank != triple_rank) {
            let pair_natural = natural_cards(hand, pair_rank, level);
            for triple_wilds in 0..=wilds.len().min(3) {
                for pair_wilds in 0..=wilds.len().saturating_sub(triple_wilds).min(2) {
                    if !pair_rank.is_standard() && pair_wilds > 0 {
                        continue;
                    }
                    let triple_count = 3 - triple_wilds;
                    let pair_count = 2 - pair_wilds;
                    if triple_count <= triple_natural.len() && pair_count <= pair_natural.len() {
                        let mut cards = triple_natural[..triple_count].to_vec();
                        cards.extend_from_slice(&pair_natural[..pair_count]);
                        cards.extend_from_slice(&wilds[..triple_wilds + pair_wilds]);
                        output.add(cards);
                    }
                }
            }
        }
    }
}

fn collect_four_jokers(hand: &Hand, output: &mut CandidateCards) {
    let jokers: Vec<_> = hand
        .cards()
        .filter(|card| matches!(card.rank(), Rank::SmallJoker | Rank::BigJoker))
        .collect();
    if jokers.len() == 4 {
        output.add(jokers);
    }
}
