use ddz_core::{
    decode_state, encode_state, CardPlayState, DealPlan, DealState, DeckOrder, DoublingState,
    GameState, LandlordSelectionState, Phase, Rank, RankCounts, RevealState, Seat, SeatMap,
    SeatOrder, SeatSet, StakeState, DEAL_ROUNDS,
};

fn post_bid_state() -> GameState {
    let plan = DealPlan::new(DeckOrder::identity());
    let landlord = Seat::ZERO;
    let mut hands = plan.final_hands();
    hands[landlord] = hands[landlord]
        .checked_add(plan.bottom_counts())
        .expect("bottom cards fit landlord hand");

    GameState {
        rule_config_id: 1,
        phase: Phase::CardPlay,
        current_player: Some(landlord),
        deal: DealState {
            attempt: 0,
            plan,
            rounds_dealt: DEAL_ROUNDS,
        },
        hands,
        reveal: RevealState::hidden(),
        landlord_selection: LandlordSelectionState::PostBid { landlord },
        doubling: DoublingState::Disabled,
        stake: StakeState::default(),
        card_play: CardPlayState::empty(),
        history: Vec::new(),
        outcome: None,
    }
}

#[test]
fn seats_are_validated_and_rotate_without_integer_indexing() {
    assert_eq!(Seat::ZERO.next(), Seat::ONE);
    assert_eq!(Seat::ONE.next(), Seat::TWO);
    assert_eq!(Seat::TWO.next(), Seat::ZERO);
    assert_eq!(Seat::TWO.relative_to(Seat::ONE), 1);
    assert!(Seat::new(3).is_err());
}

#[test]
fn seat_collections_reject_invalid_structure() {
    let order = SeatOrder::new([Seat::ONE, Seat::TWO]).expect("unique order");
    assert_eq!(order.as_slice(), &[Seat::ONE, Seat::TWO]);
    assert!(SeatOrder::new([Seat::ONE, Seat::ONE]).is_err());

    let set = SeatSet::singleton(Seat::ZERO).union(SeatSet::singleton(Seat::TWO));
    assert!(set.contains(Seat::ZERO));
    assert!(!set.contains(Seat::ONE));
    assert!(set.contains(Seat::TWO));
}

#[test]
fn rank_counts_enforce_physical_capacity() {
    let mut raw = [0; 15];
    raw[Rank::Three.index()] = 4;
    raw[Rank::SmallJoker.index()] = 1;
    assert!(RankCounts::new(raw).is_ok());

    raw[Rank::SmallJoker.index()] = 2;
    assert!(RankCounts::new(raw).is_err());
}

#[test]
fn deal_plan_is_round_robin_and_keeps_three_bottom_cards() {
    let plan = DealPlan::new(DeckOrder::identity());
    assert_eq!(plan.card_for(Seat::ZERO, 0).expect("card").value(), 0);
    assert_eq!(plan.card_for(Seat::ONE, 0).expect("card").value(), 1);
    assert_eq!(plan.card_for(Seat::TWO, 0).expect("card").value(), 2);
    assert_eq!(
        plan.bottom_cards().map(|card| card.value()),
        [51, 52, 53]
    );
    assert_eq!(
        plan.final_hands()
            .iter()
            .map(|(_, hand)| hand.card_count())
            .collect::<Vec<_>>(),
        vec![17, 17, 17]
    );
}

#[test]
fn authoritative_state_roundtrips_through_versioned_envelope() {
    let state = post_bid_state();
    state.validate().expect("valid state");
    let bytes = encode_state(&state).expect("encode");
    let restored = decode_state(&bytes).expect("decode");
    assert_eq!(restored, state);
}

#[test]
fn seat_map_is_strongly_indexed() {
    let mut values = SeatMap::new([10, 20, 30]);
    values[Seat::ONE] = 25;
    assert_eq!(values[Seat::ZERO], 10);
    assert_eq!(values[Seat::ONE], 25);
    assert_eq!(values[Seat::TWO], 30);
}
