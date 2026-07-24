use std::collections::BTreeSet;

use guandan_rules::{
    all_cards, Card, Hand, Rank, Round, RoundOutcome, Seat, TributeError, TributeMode, TributePlan,
    PLAYER_COUNT,
};

fn deal_with(required: [Vec<Card>; PLAYER_COUNT]) -> [Hand; PLAYER_COUNT] {
    let reserved: BTreeSet<_> = required.iter().flatten().copied().collect();
    let mut cards = required;
    for card in all_cards() {
        if reserved.contains(&card) {
            continue;
        }
        let seat = cards
            .iter()
            .position(|hand| hand.len() < 27)
            .expect("four hands have room for the full deck");
        cards[seat].push(card);
    }
    std::array::from_fn(|seat| Hand::from_cards(cards[seat].iter().copied()).unwrap())
}

fn return_card(hand: &Hand) -> Card {
    hand.cards()
        .find(|card| {
            card.rank()
                .natural_index()
                .is_some_and(|rank| rank <= Rank::Ten.natural_index().unwrap())
        })
        .expect("a test winner has a returnable card")
}

#[test]
fn single_tribute_exchanges_cards_and_giver_opens() {
    let big_zero = Card::joker(0, Rank::BigJoker).unwrap();
    let big_one = Card::joker(1, Rank::BigJoker).unwrap();
    let hands = deal_with([vec![big_one], vec![], vec![big_zero], vec![]]);
    let outcome =
        RoundOutcome::from_finish_order([Seat::ZERO, Seat::ONE, Seat::THREE, Seat::TWO]).unwrap();
    let returned = return_card(&hands[Seat::ZERO.index()]);
    let plan = TributePlan::from_previous_round(&outcome, hands, Rank::Ten).unwrap();

    assert_eq!(plan.mode(), TributeMode::Single);
    let assignment = plan.assign_offers(&[(Seat::TWO, big_zero)], None).unwrap();
    assert_eq!(assignment.opening_player(), Seat::TWO);
    let resolution = assignment.apply_returns(&[(Seat::ZERO, returned)]).unwrap();

    assert!(resolution.hands()[Seat::ZERO.index()].contains(big_zero));
    assert!(resolution.hands()[Seat::TWO.index()].contains(returned));
    Round::from_deal(Rank::Ten, resolution.into_hands(), Seat::TWO).unwrap();
}

#[test]
fn two_big_jokers_resist_single_tribute() {
    let hands = deal_with([
        vec![],
        vec![],
        vec![
            Card::joker(0, Rank::BigJoker).unwrap(),
            Card::joker(1, Rank::BigJoker).unwrap(),
        ],
        vec![],
    ]);
    let outcome =
        RoundOutcome::from_finish_order([Seat::ZERO, Seat::ONE, Seat::THREE, Seat::TWO]).unwrap();
    let plan = TributePlan::from_previous_round(&outcome, hands, Rank::Two).unwrap();

    assert!(plan.is_resisted());
    assert!(plan.required_givers().is_empty());
    let assignment = plan.assign_offers(&[], None).unwrap();
    assert_eq!(assignment.opening_player(), Seat::ZERO);
    assert!(assignment.apply_returns(&[]).is_ok());
}

#[test]
fn equal_double_tribute_uses_winner_choice_and_heads_next_player_opens() {
    let small_zero = Card::joker(0, Rank::SmallJoker).unwrap();
    let small_one = Card::joker(1, Rank::SmallJoker).unwrap();
    let hands = deal_with([
        vec![Card::joker(0, Rank::BigJoker).unwrap()],
        vec![small_zero],
        vec![Card::joker(1, Rank::BigJoker).unwrap()],
        vec![small_one],
    ]);
    let returns = [
        (Seat::ZERO, return_card(&hands[Seat::ZERO.index()])),
        (Seat::TWO, return_card(&hands[Seat::TWO.index()])),
    ];
    let outcome =
        RoundOutcome::from_finish_order([Seat::ZERO, Seat::TWO, Seat::ONE, Seat::THREE]).unwrap();
    let plan = TributePlan::from_previous_round(&outcome, hands, Rank::Seven).unwrap();

    assert_eq!(plan.mode(), TributeMode::Double);
    let error = plan
        .clone()
        .assign_offers(&[(Seat::ONE, small_zero), (Seat::THREE, small_one)], None)
        .unwrap_err();
    assert_eq!(error, TributeError::EqualOfferChoiceRequired);

    let assignment = plan
        .assign_offers(
            &[(Seat::ONE, small_zero), (Seat::THREE, small_one)],
            Some(Seat::ONE),
        )
        .unwrap();
    assert_eq!(assignment.opening_player(), Seat::ONE);
    let resolution = assignment.apply_returns(&returns).unwrap();
    assert_eq!(resolution.tributes()[0].to, Seat::ZERO);
    assert_eq!(resolution.tributes()[1].to, Seat::TWO);
}

#[test]
fn double_tribute_is_resisted_when_losers_hold_both_big_jokers() {
    let hands = deal_with([
        vec![],
        vec![Card::joker(0, Rank::BigJoker).unwrap()],
        vec![],
        vec![Card::joker(1, Rank::BigJoker).unwrap()],
    ]);
    let outcome =
        RoundOutcome::from_finish_order([Seat::ZERO, Seat::TWO, Seat::ONE, Seat::THREE]).unwrap();
    let plan = TributePlan::from_previous_round(&outcome, hands, Rank::Two).unwrap();

    assert_eq!(plan.mode(), TributeMode::Double);
    assert!(plan.is_resisted());
}
