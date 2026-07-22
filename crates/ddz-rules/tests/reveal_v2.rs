use ddz_rules::{
    GameActionV2, HuanleMatchV2, MatchError, PhaseV2, RevealDecisionV2, RuleConfigV2,
    PRE_DEAL_REVEAL_ORDER_ALGORITHM,
};

const HUANLE_V2_FIXTURE: &str =
    include_str!("../../../tests/rules/huanle_classic_v1/parser_fixture_v2.yaml");

fn rules() -> RuleConfigV2 {
    RuleConfigV2::from_yaml_str(HUANLE_V2_FIXTURE)
        .expect("fully explicit Huanle parser fixture must remain valid")
}

fn apply_pre_deal_decisions(game: &mut HuanleMatchV2, decisions: [RevealDecisionV2; 3]) {
    while game.phase() == PhaseV2::PreDealReveal {
        let actor = game
            .reveal_observation(0)
            .unwrap()
            .pre_deal_reveal_actor
            .unwrap();
        game.apply_pre_deal_reveal(actor, decisions[usize::from(actor)])
            .unwrap();
    }
}

fn respond_to_current_dealing_round(game: &mut HuanleMatchV2, decisions: [RevealDecisionV2; 3]) {
    assert_eq!(game.phase(), PhaseV2::DealingReveal);
    let pending = game
        .reveal_observation(0)
        .unwrap()
        .pending_during_deal_reveal;
    for (actor, is_pending) in pending.iter().copied().enumerate() {
        if is_pending {
            game.apply_during_deal_reveal(
                u8::try_from(actor).expect("Huanle seat index fits in u8"),
                decisions[actor],
            )
            .unwrap();
        }
    }
}

fn finish_dealing_with_declines(game: &mut HuanleMatchV2) {
    while game.phase() != PhaseV2::Calling {
        match game.phase() {
            PhaseV2::PreDealReveal => {
                apply_pre_deal_decisions(game, [RevealDecisionV2::Decline; 3]);
            }
            PhaseV2::DealingReveal => {
                respond_to_current_dealing_round(game, [RevealDecisionV2::Decline; 3]);
            }
            PhaseV2::Calling => unreachable!("loop condition excludes calling"),
        }
    }
}

#[test]
// HCV1-REVEAL-001, HCV1-REVEAL-003, HCV1-BOTTOM-001.
fn pre_deal_reveal_uses_configured_x5_and_never_leaks_hidden_cards() {
    let mut game = HuanleMatchV2::new(101, &rules()).unwrap();
    let initial = game.reveal_observation(0).unwrap();
    assert_eq!(initial.phase, PhaseV2::PreDealReveal);
    assert_eq!(
        game.state().current_attempt.pre_deal_reveal_order_algorithm,
        PRE_DEAL_REVEAL_ORDER_ALGORITHM
    );
    let mut declaration_order = game.state().current_attempt.pre_deal_reveal_order;
    declaration_order.sort_unstable();
    assert_eq!(declaration_order, [0, 1, 2]);
    assert!(initial.own_hand.is_empty());
    assert!(initial.public_revealed_hands.iter().all(Vec::is_empty));
    assert!(!initial.bottom_visible);

    let order = game.state().current_attempt.pre_deal_reveal_order;
    let revealer = order[1];
    let observer = (revealer + 1) % 3;
    let mut decisions = [RevealDecisionV2::Decline; 3];
    decisions[usize::from(revealer)] = RevealDecisionV2::Reveal;
    apply_pre_deal_decisions(&mut game, decisions);

    let state = game.state();
    assert_eq!(state.current_attempt.reveal.first_revealer, Some(revealer));
    assert_eq!(state.current_attempt.reveal.first_reveal_sequence, Some(1));
    assert_eq!(
        state.current_attempt.reveal.reveal_factor_by_seat[usize::from(revealer)],
        rules().reveal.before_deal_factor
    );
    assert_eq!(state.current_attempt.reveal.maximum_factor, 5);

    let observation = game.reveal_observation(observer).unwrap();
    assert_eq!(observation.cards_received, [1, 1, 1]);
    assert_eq!(observation.own_hand.len(), 1);
    assert_eq!(
        observation.public_revealed_hands[usize::from(revealer)].len(),
        1
    );
    for seat in 0..3 {
        if seat != usize::from(revealer) {
            assert!(observation.public_revealed_hands[seat].is_empty());
        }
    }
    let json = serde_json::to_value(&observation).unwrap();
    assert_eq!(json.get("bottom_visible").unwrap(), false);
    assert!(json.get("bottom_cards").is_none());
    assert!(json.get("deck").is_none());
}

#[test]
// HCV1-REVEAL-002, HCV1-REVEAL-003, HCV1-CALL-002.
fn during_deal_factor_schedule_and_multi_reveal_maximum_are_configuration_driven() {
    let mut configured = rules();
    configured.reveal.factor_by_cards_received = [4; 18];
    configured.reveal.factor_by_cards_received[1] = 3;

    let mut game = HuanleMatchV2::new(202, &configured).unwrap();
    let seeded_candidate = game.state().current_attempt.first_caller_candidate;
    let first_revealer = (seeded_candidate + 1) % 3;
    let second_revealer = (first_revealer + 1) % 3;
    apply_pre_deal_decisions(&mut game, [RevealDecisionV2::Decline; 3]);

    let mut first_round = [RevealDecisionV2::Decline; 3];
    first_round[usize::from(first_revealer)] = RevealDecisionV2::Reveal;
    respond_to_current_dealing_round(&mut game, first_round);
    assert_eq!(
        game.state().current_attempt.reveal.reveal_factor_by_seat[usize::from(first_revealer)],
        3
    );

    let mut second_round = [RevealDecisionV2::Decline; 3];
    second_round[usize::from(second_revealer)] = RevealDecisionV2::Reveal;
    respond_to_current_dealing_round(&mut game, second_round);
    assert_eq!(
        game.state().current_attempt.reveal.reveal_factor_by_seat[usize::from(second_revealer)],
        4
    );
    assert_eq!(game.state().current_attempt.reveal.maximum_factor, 4);
    assert_eq!(
        game.state().current_attempt.reveal.first_revealer,
        Some(first_revealer)
    );

    finish_dealing_with_declines(&mut game);
    let observation = game.reveal_observation(2).unwrap();
    assert_eq!(observation.phase, PhaseV2::Calling);
    assert_eq!(observation.first_caller, Some(first_revealer));
    assert_ne!(first_revealer, seeded_candidate);
    assert_eq!(
        observation.public_revealed_hands[usize::from(first_revealer)].len(),
        17
    );
    assert_eq!(
        observation.public_revealed_hands[usize::from(second_revealer)].len(),
        17
    );
    let still_hidden = (0..3)
        .find(|seat| *seat != usize::from(first_revealer) && *seat != usize::from(second_revealer))
        .unwrap();
    assert!(observation.public_revealed_hands[still_hidden].is_empty());
}

#[test]
// HCV1-DEAL-001, HCV1-REVEAL-001, HCV1-REVEAL-003.
fn authoritative_dealing_preserves_the_full_physical_partition_and_replays_exactly() {
    let mut game = HuanleMatchV2::new(303, &rules()).unwrap();
    apply_pre_deal_decisions(&mut game, [RevealDecisionV2::Reveal; 3]);

    assert_eq!(game.phase(), PhaseV2::Calling);
    let state = game.state();
    assert_eq!(state.current_attempt.cards_received, [17, 17, 17]);
    assert_eq!(state.current_attempt.reveal.maximum_factor, 5);
    let observation = game.reveal_observation(0).unwrap();
    assert!(observation.revealed.iter().all(|revealed| *revealed));
    assert!(observation
        .public_revealed_hands
        .iter()
        .all(|hand| hand.len() == 17));
    assert!(!observation.bottom_visible);

    let mut physical_cards = observation
        .public_revealed_hands
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    physical_cards.extend_from_slice(&state.current_attempt.deck[51..]);
    physical_cards.sort_unstable();
    assert_eq!(physical_cards, (0_u8..54).collect::<Vec<_>>());

    let replay = HuanleMatchV2::replay(303, &rules(), game.decision_events()).unwrap();
    assert_eq!(replay, game);
}

#[test]
// HCV1-REVEAL-001, HCV1-REVEAL-002.
fn reveal_actions_are_phase_checked_irrevocable_and_transactional() {
    let mut game = HuanleMatchV2::new(404, &rules()).unwrap();
    let initial = game.clone();
    let order = game.state().current_attempt.pre_deal_reveal_order;
    let expected = order[0];
    let wrong_actor = order[1];

    assert!(matches!(
        game.apply_pre_deal_reveal(wrong_actor, RevealDecisionV2::Decline),
        Err(MatchError::PreDealRevealOutOfTurn {
            expected: actual_expected,
            actual
        }) if actual_expected == expected && actual == wrong_actor
    ));
    assert_eq!(game, initial);
    assert!(matches!(
        game.record_accepted_action(
            expected,
            GameActionV2::PreDealReveal(RevealDecisionV2::Reveal)
        ),
        Err(MatchError::RevealActionRequiresStateMachine)
    ));
    assert_eq!(game, initial);

    game.apply_pre_deal_reveal(expected, RevealDecisionV2::Reveal)
        .unwrap();
    let after_reveal = game.clone();
    assert!(matches!(
        game.apply_during_deal_reveal(expected, RevealDecisionV2::Reveal),
        Err(MatchError::UnexpectedPhase {
            expected: PhaseV2::DealingReveal,
            actual: PhaseV2::PreDealReveal
        })
    ));
    assert_eq!(game, after_reveal);

    let mut remaining = [RevealDecisionV2::Decline; 3];
    remaining[usize::from(expected)] = RevealDecisionV2::Reveal;
    apply_pre_deal_decisions(&mut game, remaining);
    assert_eq!(game.phase(), PhaseV2::DealingReveal);
    let before_duplicate = game.clone();
    assert!(matches!(
        game.apply_during_deal_reveal(expected, RevealDecisionV2::Reveal),
        Err(MatchError::DuringDealRevealNotPending { seat }) if seat == expected
    ));
    assert_eq!(game, before_duplicate);

    let legal = game.legal_reveal_actions(expected).unwrap();
    assert!(legal.is_empty());
}
