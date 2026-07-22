use ddz_rules::{
    derive_attempt_seed, AttemptActionRecordV2, AttemptCompletionReasonV2, AttemptStatusV2,
    CallDecisionV2, GameActionV2, HuanleMatchV2, MatchDecisionEventV2, MatchError, PhaseV2,
    RevealDecisionV2, RobDecisionV2, RuleConfigV2, SystemEventV2,
    ATTEMPT_SEED_DERIVATION_ALGORITHM, SHUFFLE_ALGORITHM,
};

const HUANLE_V2_FIXTURE: &str =
    include_str!("../../../tests/rules/huanle_classic_v1/parser_fixture_v2.yaml");

fn rules() -> RuleConfigV2 {
    RuleConfigV2::from_yaml_str(HUANLE_V2_FIXTURE)
        .expect("fully explicit Huanle parser fixture must remain valid")
}

fn resolve_no_reveal_all_pass_through_call_state(game: &mut HuanleMatchV2) {
    drive_to_calling_without_reveal(game);
    while game.phase() == PhaseV2::Calling {
        let actor = game.state().current_attempt.call.unwrap().current_player;
        game.apply_call(actor, CallDecisionV2::PassCall).unwrap();
    }
    assert_eq!(game.phase(), PhaseV2::PreDealReveal);
}

fn drive_to_calling_without_reveal(game: &mut HuanleMatchV2) {
    while game.phase() != PhaseV2::Calling {
        match game.phase() {
            PhaseV2::PreDealReveal => {
                let actor = game
                    .reveal_observation(0)
                    .unwrap()
                    .pre_deal_reveal_actor
                    .unwrap();
                game.apply_pre_deal_reveal(actor, RevealDecisionV2::Decline)
                    .unwrap();
            }
            PhaseV2::DealingReveal => {
                let pending = game
                    .reveal_observation(0)
                    .unwrap()
                    .pending_during_deal_reveal;
                for (actor, is_pending) in pending.iter().copied().enumerate() {
                    if is_pending {
                        game.apply_during_deal_reveal(
                            u8::try_from(actor).expect("Huanle seat index fits in u8"),
                            RevealDecisionV2::Decline,
                        )
                        .unwrap();
                    }
                }
            }
            PhaseV2::Calling => unreachable!("loop condition excludes calling"),
            PhaseV2::Robbing | PhaseV2::BottomReveal => {
                unreachable!("R004 helper must stop before later phase boundaries")
            }
        }
    }
}

fn resolve_robbing_with_passes(game: &mut HuanleMatchV2) {
    while game.phase() == PhaseV2::Robbing {
        let actor = *game
            .state()
            .current_attempt
            .rob
            .as_ref()
            .unwrap()
            .queue
            .front()
            .unwrap();
        game.apply_rob(actor, RobDecisionV2::PassRob).unwrap();
    }
    assert_eq!(game.phase(), PhaseV2::BottomReveal);
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

    resolve_no_reveal_all_pass_through_call_state(&mut game);
    resolve_no_reveal_all_pass_through_call_state(&mut game);

    assert_eq!(game.state().attempt_index, 2);
    assert_eq!(game.state().completed_attempts.len(), 2);
    assert_eq!(game.state().total_accepted_action_count, 114);
    for (index, summary) in game.state().completed_attempts.iter().enumerate() {
        assert_eq!(summary.attempt_index, u32::try_from(index).unwrap());
        assert_eq!(summary.accepted_action_count, 57);
        assert_eq!(summary.action_history.len(), 57);
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

    drive_to_calling_without_reveal(&mut game);
    let caller = game.state().current_attempt.call.unwrap().current_player;
    game.apply_call(caller, CallDecisionV2::CallLandlord)
        .unwrap();
    resolve_robbing_with_passes(&mut game);
    game.complete_after_authoritative_card_play(2).unwrap();

    assert!(game.state().terminal);
    assert_eq!(game.state().total_accepted_action_count, 171);
    assert_eq!(
        game.state().current_attempt.status,
        AttemptStatusV2::LandlordResolved { landlord: caller }
    );
    assert_eq!(game.state().final_result.unwrap().winner, 2);
    assert_eq!(game.decision_events().len(), 175);
    assert_eq!(game.system_events().len(), 64);
}

#[test]
fn deterministic_decision_replay_reconstructs_all_attempts_and_action_budgets_exactly() {
    let mut original = HuanleMatchV2::new(4242, &rules()).unwrap();
    resolve_no_reveal_all_pass_through_call_state(&mut original);
    drive_to_calling_without_reveal(&mut original);
    let caller = original
        .state()
        .current_attempt
        .call
        .unwrap()
        .current_player;
    original
        .apply_call(caller, CallDecisionV2::CallLandlord)
        .unwrap();
    resolve_robbing_with_passes(&mut original);
    original.complete_after_authoritative_card_play(0).unwrap();

    let replay = HuanleMatchV2::replay(4242, &rules(), original.decision_events()).unwrap();

    assert_eq!(replay, original);
    assert_eq!(replay.state().total_accepted_action_count, 114);
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
        game.record_accepted_action(3, &GameActionV2::Call(CallDecisionV2::PassCall)),
        Err(MatchError::InvalidSeat { seat: 3 })
    ));
    assert_eq!(game, initial);
    assert!(matches!(
        game.complete_after_authoritative_card_play(0),
        Err(MatchError::LandlordNotResolved)
    ));
    assert_eq!(game, initial);

    let mut revealed = HuanleMatchV2::new(13, &rules()).unwrap();
    let first_actor = revealed
        .reveal_observation(0)
        .unwrap()
        .pre_deal_reveal_actor
        .unwrap();
    revealed
        .apply_pre_deal_reveal(first_actor, RevealDecisionV2::Reveal)
        .unwrap();
    let before_invalid_all_pass = revealed.clone();
    assert!(matches!(
        revealed.resolve_no_reveal_all_pass(),
        Err(MatchError::AllPassAfterReveal)
    ));
    assert_eq!(revealed, before_invalid_all_pass);

    drive_to_calling_without_reveal(&mut game);
    let caller = game.state().current_attempt.call.unwrap().current_player;
    game.apply_call(caller, CallDecisionV2::CallLandlord)
        .unwrap();
    resolve_robbing_with_passes(&mut game);
    let resolved = game.clone();
    assert!(matches!(
        game.resolve_no_reveal_all_pass(),
        Err(MatchError::AttemptAlreadyHasLandlord { landlord }) if landlord == caller
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

    let pre_deal_actor = HuanleMatchV2::new(8, &rules())
        .unwrap()
        .state()
        .current_attempt
        .pre_deal_reveal_order[0];
    let tampered_sequence = [MatchDecisionEventV2::PlayerActionAccepted {
        sequence: 7,
        attempt_index: 0,
        action: AttemptActionRecordV2 {
            sequence: 0,
            actor: pre_deal_actor,
            action: GameActionV2::PreDealReveal(RevealDecisionV2::Decline),
        },
    }];
    assert!(matches!(
        HuanleMatchV2::replay(8, &rules(), &tampered_sequence),
        Err(MatchError::ReplayEventMismatch { sequence: 7 })
    ));
}
