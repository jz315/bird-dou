use ddz_core::{
    card_id_to_rank, cards_to_rank_counts, max_count_for_rank, rank_counts_to_card_ids,
    rank_to_card_ids, CardError, BIG_JOKER_CARD, BIG_JOKER_RANK, CARD_COUNT, EMPTY_RANK_COUNTS,
    SMALL_JOKER_CARD, SMALL_JOKER_RANK,
};

#[test]
fn every_physical_card_maps_to_the_expected_rank() {
    for card_id in 0_u8..52 {
        assert_eq!(card_id_to_rank(card_id), Ok(card_id / 4));
    }
    assert_eq!(card_id_to_rank(SMALL_JOKER_CARD), Ok(SMALL_JOKER_RANK));
    assert_eq!(card_id_to_rank(BIG_JOKER_CARD), Ok(BIG_JOKER_RANK));
    assert_eq!(
        card_id_to_rank(54),
        Err(CardError::InvalidCardId { card_id: 54 })
    );
    assert_eq!(
        cards_to_rank_counts(&[54]),
        Err(CardError::InvalidCardId { card_id: 54 })
    );
}

#[test]
fn rank_to_cards_is_stable_and_validated() {
    assert_eq!(rank_to_card_ids(0), Ok(vec![0, 1, 2, 3]));
    assert_eq!(rank_to_card_ids(12), Ok(vec![48, 49, 50, 51]));
    assert_eq!(rank_to_card_ids(13), Ok(vec![52]));
    assert_eq!(rank_to_card_ids(14), Ok(vec![53]));
    assert_eq!(
        rank_to_card_ids(15),
        Err(CardError::InvalidRankId { rank_id: 15 })
    );
    assert_eq!(max_count_for_rank(12), Ok(4));
    assert_eq!(max_count_for_rank(13), Ok(1));
}

#[test]
fn the_complete_deck_round_trips_without_loss() {
    let deck: Vec<u8> = (0..u8::try_from(CARD_COUNT).expect("54 fits in u8")).collect();
    let counts = cards_to_rank_counts(&deck).expect("the complete deck is valid");

    assert_eq!(counts, [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1]);
    assert_eq!(rank_counts_to_card_ids(&counts), Ok(deck));
}

#[test]
fn a_partial_hand_has_a_deterministic_canonical_expansion() {
    let cards = [3, 0, 9, 52, 48, 51];
    let counts = cards_to_rank_counts(&cards).expect("the physical cards are unique");

    assert_eq!(
        rank_counts_to_card_ids(&counts),
        Ok(vec![0, 1, 8, 48, 49, 52])
    );
}

#[test]
fn every_legal_single_rank_count_round_trips() {
    for rank_id in 0_u8..15 {
        let maximum = max_count_for_rank(rank_id).expect("all enumerated ranks are valid");
        for count in 0..=maximum {
            let mut expected = EMPTY_RANK_COUNTS;
            expected[usize::from(rank_id)] = count;
            let cards = rank_counts_to_card_ids(&expected).expect("count is within capacity");

            assert_eq!(cards_to_rank_counts(&cards), Ok(expected));
        }
    }
}

#[test]
fn duplicate_physical_cards_are_rejected() {
    assert_eq!(
        cards_to_rank_counts(&[7, 7]),
        Err(CardError::DuplicateCardId { card_id: 7 })
    );
}

#[test]
fn impossible_rank_counts_are_rejected() {
    let mut too_many_threes = EMPTY_RANK_COUNTS;
    too_many_threes[0] = 5;
    assert_eq!(
        rank_counts_to_card_ids(&too_many_threes),
        Err(CardError::TooManyCardsForRank {
            rank_id: 0,
            count: 5,
            maximum: 4,
        })
    );

    let mut duplicate_joker = EMPTY_RANK_COUNTS;
    duplicate_joker[usize::from(SMALL_JOKER_RANK)] = 2;
    assert_eq!(
        rank_counts_to_card_ids(&duplicate_joker),
        Err(CardError::TooManyCardsForRank {
            rank_id: SMALL_JOKER_RANK,
            count: 2,
            maximum: 1,
        })
    );
}
