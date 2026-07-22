use ddz_rules::{
    AttemptStatusV2, CallDecisionV2, GameActionV2, HuanleMatchV2, MatchDecisionEventV2, MatchError,
    PhaseV2, RevealDecisionV2, RuleConfigV2, SystemEventV2,
};

const HUANLE_V2_FIXTURE: &str =
    include_str!("../../../tests/rules/huanle_classic_v1/parser_fixture_v2.yaml");

fn rules() -> RuleConfigV2 {
    RuleConfigV2::from_yaml_str(HUANLE_V2_FIXTURE)
        .expect("fully explicit Huanle parser fixture must remain valid")
}

fn advance_to_calling(game: &mut HuanleMatchV2, pre_deal_revealer: Option<u8>) {
    while game.phase() != PhaseV2::Calling {
        match game.phase() {
            PhaseV2::PreDealReveal => {
                let actor = game
                    .reveal_observation(0)
                    .unwrap()
                    .pre_deal_reveal_actor
                    .unwrap();
                let decision = if Some(actor) == pre_deal_revealer {
                    RevealDecisionV2::Reveal
                } else {
                    RevealDecisionV2::Decline
                };
                game.apply_pre_deal_reveal(actor, decision).unwrap();
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
                unreachable!("call helper must stop at the Calling boundary")
            }
        }
    }
}

fn pass_every_initial_call(game: &mut HuanleMatchV2) {
    while game.phase() == PhaseV2::Calling {
        let actor = game.state().current_attempt.call.unwrap().current_player;
        game.apply_call(actor, CallDecisionV2::PassCall).unwrap();
    }
}

#[test]
// HCV1-CALL-001.
fn initial_call_actions_are_exactly_call_or_pass_and_are_turn_bound() {
    let mut game = HuanleMatchV2::new(505, &rules()).unwrap();
    advance_to_calling(&mut game, None);
    let call = game.state().current_attempt.call.unwrap();
    assert_eq!(
        call.first_caller,
        game.state().current_attempt.first_caller.unwrap()
    );
    assert_eq!(call.current_player, call.first_caller);
    assert_eq!(
        game.legal_call_actions(call.current_player).unwrap(),
        vec![
            GameActionV2::Call(CallDecisionV2::CallLandlord),
            GameActionV2::Call(CallDecisionV2::PassCall),
        ]
    );
    assert!(game
        .legal_call_actions((call.current_player + 1) % 3)
        .unwrap()
        .is_empty());

    let before_invalid = game.clone();
    let wrong_actor = (call.current_player + 1) % 3;
    assert!(matches!(
        game.apply_call(wrong_actor, CallDecisionV2::PassCall),
        Err(MatchError::CallOutOfTurn { expected, actual })
            if expected == call.current_player && actual == wrong_actor
    ));
    assert_eq!(game, before_invalid);
    assert!(matches!(
        game.record_accepted_action(
            call.current_player,
            &GameActionV2::Call(CallDecisionV2::PassCall)
        ),
        Err(MatchError::CallActionRequiresStateMachine)
    ));
    assert_eq!(game, before_invalid);
    assert!(matches!(
        game.record_landlord_resolution(call.current_player),
        Err(MatchError::LandlordResolutionRequiresCallOutcome)
    ));
    assert_eq!(game, before_invalid);
}

#[test]
// HCV1-CALL-001.
fn first_positive_call_immediately_opens_the_r006_robbing_boundary() {
    let mut game = HuanleMatchV2::new(606, &rules()).unwrap();
    advance_to_calling(&mut game, None);
    let caller = game.state().current_attempt.call.unwrap().current_player;

    game.apply_call(caller, CallDecisionV2::CallLandlord)
        .unwrap();

    assert_eq!(game.phase(), PhaseV2::Robbing);
    let call = game.state().current_attempt.call.unwrap();
    assert_eq!(call.caller, Some(caller));
    assert!(call.acted[usize::from(caller)]);
    assert!(!call.declined[usize::from(caller)]);
    assert_eq!(
        game.state().current_attempt.status,
        AttemptStatusV2::Unresolved
    );
    assert!(game.legal_call_actions(caller).unwrap().is_empty());
    assert!(game.system_events().iter().any(|record| {
        record.event
            == SystemEventV2::CallingEndedWithCall {
                attempt_index: game.state().attempt_index,
                caller,
            }
    }));

    let before_second_call = game.clone();
    assert!(matches!(
        game.apply_call((caller + 1) % 3, CallDecisionV2::CallLandlord),
        Err(MatchError::UnexpectedPhase {
            expected: PhaseV2::Calling,
            actual: PhaseV2::Robbing
        })
    ));
    assert_eq!(game, before_second_call);
}

#[test]
// HCV1-CALL-001, HCV1-ROB-001.
fn passed_initial_call_is_retained_for_r006_eligibility() {
    let mut game = HuanleMatchV2::new(707, &rules()).unwrap();
    advance_to_calling(&mut game, None);
    let first = game.state().current_attempt.call.unwrap().current_player;
    game.apply_call(first, CallDecisionV2::PassCall).unwrap();

    let second = game.state().current_attempt.call.unwrap().current_player;
    assert_eq!(second, (first + 1) % 3);
    game.apply_call(second, CallDecisionV2::CallLandlord)
        .unwrap();

    let call = game.state().current_attempt.call.unwrap();
    assert_eq!(game.phase(), PhaseV2::Robbing);
    assert_eq!(call.caller, Some(second));
    assert_eq!(call.declined, [first == 0, first == 1, first == 2]);
    let r006_eligible = call.declined.map(|declined| !declined);
    assert!(!r006_eligible[usize::from(first)]);
    assert!(r006_eligible[usize::from(second)]);
}

#[test]
// HCV1-CALL-003.
fn all_initial_passes_without_a_revealer_redeal_the_same_match_and_replay_exactly() {
    let mut game = HuanleMatchV2::new(808, &rules()).unwrap();
    advance_to_calling(&mut game, None);
    pass_every_initial_call(&mut game);

    assert_eq!(game.phase(), PhaseV2::PreDealReveal);
    assert_eq!(game.state().attempt_index, 1);
    assert!(!game.state().terminal);
    assert_eq!(game.state().completed_attempts.len(), 1);
    let summary = &game.state().completed_attempts[0];
    let call = summary.call.unwrap();
    assert!(call.acted.iter().all(|acted| *acted));
    assert!(call.declined.iter().all(|declined| *declined));
    assert_eq!(call.caller, None);
    assert_eq!(summary.first_caller, Some(call.first_caller));
    assert_eq!(summary.accepted_action_count, 57);
    assert!(matches!(
        game.decision_events().last(),
        Some(MatchDecisionEventV2::AllPass {
            attempt_index: 0,
            ..
        })
    ));

    let replay = HuanleMatchV2::replay(808, &rules(), game.decision_events()).unwrap();
    assert_eq!(replay, game);
}

#[test]
// HCV1-CALL-002, HCV1-CALL-003.
fn all_initial_passes_with_a_revealer_assign_that_first_revealer_without_redeal() {
    let mut game = HuanleMatchV2::new(909, &rules()).unwrap();
    let first_revealer = game.state().current_attempt.pre_deal_reveal_order[1];
    advance_to_calling(&mut game, Some(first_revealer));
    assert_eq!(
        game.state().current_attempt.first_caller,
        Some(first_revealer)
    );

    pass_every_initial_call(&mut game);

    assert_eq!(game.phase(), PhaseV2::BottomReveal);
    assert_eq!(game.state().attempt_index, 0);
    assert!(game.state().completed_attempts.is_empty());
    assert!(!game.state().terminal);
    assert_eq!(
        game.state().current_attempt.status,
        AttemptStatusV2::LandlordResolved {
            landlord: first_revealer
        }
    );
    let call = game.state().current_attempt.call.unwrap();
    assert!(call.acted.iter().all(|acted| *acted));
    assert!(call.declined.iter().all(|declined| *declined));
    assert_eq!(call.caller, None);
    assert!(game.system_events().iter().any(|record| {
        record.event
            == SystemEventV2::CallingAllPassAssignedLandlord {
                attempt_index: 0,
                landlord: first_revealer,
            }
    }));
    assert!(matches!(
        game.decision_events().last(),
        Some(MatchDecisionEventV2::LandlordResolved {
            landlord,
            ..
        }) if *landlord == first_revealer
    ));

    let replay = HuanleMatchV2::replay(909, &rules(), game.decision_events()).unwrap();
    assert_eq!(replay, game);

    let mut missing_automatic_resolution = game.decision_events().to_vec();
    assert!(matches!(
        missing_automatic_resolution.pop(),
        Some(MatchDecisionEventV2::LandlordResolved { .. })
    ));
    assert!(matches!(
        HuanleMatchV2::replay(909, &rules(), &missing_automatic_resolution),
        Err(MatchError::ReplayEventMismatch { .. })
    ));
}
