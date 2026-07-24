use std::collections::BTreeSet;

use guandan_rules::{all_cards, Card, Rank, Suit, CARD_COUNT};

#[test]
fn two_decks_have_108_unique_physical_cards() {
    let cards = all_cards();
    let ids: BTreeSet<_> = cards.iter().map(|card| card.id()).collect();

    assert_eq!(cards.len(), CARD_COUNT);
    assert_eq!(ids.len(), CARD_COUNT);
}

#[test]
fn both_heart_level_cards_are_wild() {
    for copy in 0..2 {
        let card = Card::standard(copy, Suit::Hearts, Rank::Ten).unwrap();
        assert!(card.is_wild(Rank::Ten));
        assert!(!card.is_wild(Rank::Nine));
    }
}
