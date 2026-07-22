use ddz_rules::{
    derive_attempt_seed, AttemptActionRecordV2, AttemptCompletionReasonV2, AttemptStatusV2,
    CallDecisionV2, DoubleDecisionV2, GameActionV2, HuanleMatchV2, MatchDecisionEventV2,
    MatchError, RevealDecisionV2, RuleConfigV2, SystemEventV2, ATTEMPT_SEED_DERIVATION_ALGORITHM,
    SHUFFLE_ALGORITHM,
};

const HUANLE_V2_FIXTURE: &str =
    include_str!("../../../tests/rules/huanle_classic_v1/parser_fixture_v2.yaml");

fn rules() -> RuleConfigV2 {
    RuleConfigV2::from_yaml_str(HUANLE_V2_FIXTURE)
        .expect("fully explicit Huanle parser fixture must remain valid")
}

fn record_all_pass_calls(game: &mut HuanleMatchV2) {
    for actor in 0..3 {
        game.record_accepted_action(actor, GameActionV2::Call(CallDecisionV2::PassCall))
            .unwrap();
    }
}

#[test]
fn initial_attempt_is_seeded_deterministic_and_retains_its_physical_deck() {
    let first = HuanleMatchV2::new(0xA11C_E5ED, &rules()).unwrap();
    let second = HuanleMatchV2::new(0xA11C_E5ED, &rules()).unwrap();
    let state = first.state();

    assert_eq!(first, second);
    assert_eq!(
        state.attempt_seed_derivation_algorithm,
        ATTEMPT_SEED_DERIVATION_ALGORITHM
    );
    assert_eq!(state.current_attempt.shuffle_algorithm, SHUFFLE_ALGORITHM);
    assert_eq!(
        state.current_attempt.deal_seed,
        derive_attempt_seed(state.match_seed, 0)
    );
    assert_eq!(state.current_attempt.deck.len(), 54);
    let mut sorted_deck = state.current_attempt.deck.clone();
    sorted_deck.sort_unstable();
    assert_eq!(sorted_deck, (0_u8..54).collect::<Vec<_>>());
    assert_eq!(
        first.system_events(),
        &[ddz_rules::SystemEventRecordV2 {
            sequence: 0,
            event: SystemEventV2::AttemptStarted {
                attempt_index: 0,
                deal_seed: derive_attempt_seed(state.match_seed, 0),
                first_caller_candidate: state.current_attempt.first_caller_candidate,
            },
        }]
    );
}

#[test]
fn repeated_all_passes_preserve_every_attempt_and_can_reach_a_terminal_match_lifecycle() {
    let mut game = HuanleMatchV2::new(91, &rules()).unwrap();

    record_all_pass_calls(&mut game);
    game.resolve_no_reveal_all_pass().unwrap();
    record_all_pass_calls(&mut game);
    game.resolve_no_reveal_all_pass().unwrap();

    assert_eq!(game.state().attempt_index, 2);
    assert_eq!(game.state().completed_attempts.len(), 2);
    assert_eq!(game.state().total_accepted_action_count, 6);
    for (index, summary) in game.state().completed_attempts.iter().enumerate() {
        assert_eq!(summary.attempt_index, u32::try_from(index).unwrap());
        assert_eq!(summary.accepted_action_count, 3);
        assert_eq!(summary.action_history.len(), 3);
        assert_eq!(
            summary.completion_reason,
            AttemptCompletionReasonV2::AllPass
        );
        assert_eq!(
            summary.deal_seed,
            derive_attempt_seed(91, summary.attempt_index)
        );
    }
    assert_eq!(
        game.state().current_attempt.deal_seed,
        derive_attempt_seed(91, game.state().attempt_index)
    );

    game.record_accepted_action(1, GameActionV2::Call(CallDecisionV2::CallLandlord))
        .unwrap();
    game.record_landlord_resolution(1).unwrap();
    game.record_accepted_action(2, GameActionV2::Double(DoubleDecisionV2::Decline))
        .unwrap();
    game.complete_after_authoritative_card_play(2).unwrap();

    assert!(game.state().terminal);
    assert_eq!(game.state().total_accepted_action_count, 8);
    assert_eq!(
        game.state().current_attempt.status,
        AttemptStatusV2::LandlordResolved { landlord: 1 }
    );
    assert_eq!(game.state().final_result.unwrap().winner, 2);
    assert_eq!(game.decision_events().len(), 12);
    assert_eq!(game.system_events().len(), 7);
}

#[test]
fn deterministic_decision_replay_reconstructs_all_attempts_and_action_budgets_exactly() {
    let mut original = HuanleMatchV2::new(4242, &rules()).unwrap();
    record_all_pass_calls(&mut original);
    original.resolve_no_reveal_all_pass().unwrap();
    original
        .record_accepted_action(0, GameActionV2::Call(CallDecisionV2::CallLandlord))
        .unwrap();
    original.record_landlord_resolution(0).unwrap();
    original
        .record_accepted_action(1, GameActionV2::Double(DoubleDecisionV2::Decline))
        .unwrap();
    original.complete_after_authoritative_card_play(0).unwrap();

    let replay = HuanleMatchV2::replay(4242, &rules(), original.decision_events()).unwrap();

    assert_eq!(replay, original);
    assert_eq!(replay.state().total_accepted_action_count, 5);
}

#[test]
fn invalid_lifecycle_transitions_are_transactional() {
    let mut game = HuanleMatchV2::new(12, &rules()).unwrap();
    let initial = game.clone();

    assert!(matches!(
        game.resolve_no_reveal_all_pass(),
        Err(MatchError::AllPassWithoutAcceptedActions)
    ));
    assert_eq!(game, initial);
    assert!(matches!(
        game.record_accepted_action(3, GameActionV2::Call(CallDecisionV2::PassCall)),
        Err(MatchError::InvalidSeat { seat: 3 })
    ));
    assert_eq!(game, initial);
    assert!(matches!(
        game.complete_after_authoritative_card_play(0),
        Err(MatchError::LandlordNotResolved)
    ));
    assert_eq!(game, initial);

    let mut revealed = HuanleMatchV2::new(13, &rules()).unwrap();
    revealed
        .record_accepted_action(0, GameActionV2::PreDealReveal(RevealDecisionV2::Reveal))
        .unwrap();
    let before_invalid_all_pass = revealed.clone();
    assert!(matches!(
        revealed.resolve_no_reveal_all_pass(),
        Err(MatchError::AllPassAfterReveal)
    ));
    assert_eq!(revealed, before_invalid_all_pass);

    game.record_accepted_action(0, GameActionV2::Call(CallDecisionV2::CallLandlord))
        .unwrap();
    game.record_landlord_resolution(0).unwrap();
    let resolved = game.clone();
    assert!(matches!(
        game.resolve_no_reveal_all_pass(),
        Err(MatchError::AttemptAlreadyHasLandlord { landlord: 0 })
    ));
    assert_eq!(game, resolved);
}

#[test]
fn replay_rejects_events_that_target_the_wrong_attempt_or_tamper_with_sequence() {
    let wrong_attempt = [MatchDecisionEventV2::PlayerActionAccepted {
        sequence: 0,
        attempt_index: 1,
        action: AttemptActionRecordV2 {
            sequence: 0,
            actor: 0,
            action: GameActionV2::Call(CallDecisionV2::PassCall),
        },
    }];
    assert!(matches!(
        HuanleMatchV2::replay(8, &rules(), &wrong_attempt),
        Err(MatchError::ReplayAttemptMismatch {
            expected: 0,
            actual: 1,
        })
    ));

    let tampered_sequence = [MatchDecisionEventV2::PlayerActionAccepted {
        sequence: 7,
        attempt_index: 0,
        action: AttemptActionRecordV2 {
            sequence: 0,
            actor: 0,
            action: GameActionV2::Call(CallDecisionV2::PassCall),
        },
    }];
    assert!(matches!(
        HuanleMatchV2::replay(8, &rules(), &tampered_sequence),
        Err(MatchError::ReplayEventMismatch { sequence: 7 })
    ));
}
