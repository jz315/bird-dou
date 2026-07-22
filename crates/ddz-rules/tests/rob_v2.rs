use ddz_rules::{
    AttemptStatusV2, CallDecisionV2, GameActionV2, HuanleMatchV2, MatchDecisionEventV2, MatchError,
    PhaseV2, RevealDecisionV2, RobDecisionV2, RuleConfigV2, SystemEventV2,
};

const HUANLE_V2_FIXTURE: &str =
    include_str!("../../../tests/rules/huanle_classic_v1/parser_fixture_v2.yaml");

fn rules() -> RuleConfigV2 {
    RuleConfigV2::from_yaml_str(HUANLE_V2_FIXTURE)
        .expect("fully explicit Huanle parser fixture must remain valid")
}

fn next_seat(seat: u8) -> u8 {
    (seat + 1) % 3
}

fn advance_to_calling(game: &mut HuanleMatchV2) {
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
                unreachable!("rob helper must stop at the Calling boundary")
            }
        }
    }
}

fn rob_front(game: &HuanleMatchV2) -> u8 {
    *game
        .state()
        .current_attempt
        .rob
        .as_ref()
        .unwrap()
        .queue
        .front()
        .unwrap()
}

#[test]
// HCV1-ROB-001, HCV1-ROB-003.
fn positive_call_with_no_robs_resolves_the_original_caller_at_factor_one() {
    let mut game = HuanleMatchV2::new(6_006, &rules()).unwrap();
    advance_to_calling(&mut game);
    let caller = game.state().current_attempt.call.unwrap().current_player;
    let second = next_seat(caller);
    let third = next_seat(second);

    game.apply_call(caller, CallDecisionV2::CallLandlord)
        .unwrap();
    let rob = game.state().current_attempt.rob.as_ref().unwrap();
    assert_eq!(game.phase(), PhaseV2::Robbing);
    assert_eq!(rob.caller, caller);
    assert_eq!(rob.candidate, caller);
    assert_eq!(rob.eligible, [true; 3]);
    assert_eq!(
        rob.queue.iter().copied().collect::<Vec<_>>(),
        vec![second, third, caller]
    );
    assert_eq!(rob.successful_rob_count, 0);
    assert_eq!(rob.rob_factor, 1);
    assert!(game.legal_rob_actions(caller).unwrap().is_empty());
    assert_eq!(
        game.legal_rob_actions(second).unwrap(),
        vec![
            GameActionV2::Rob(RobDecisionV2::Rob),
            GameActionV2::Rob(RobDecisionV2::PassRob),
        ]
    );

    game.apply_rob(second, RobDecisionV2::PassRob).unwrap();
    assert_eq!(rob_front(&game), third);
    game.apply_rob(third, RobDecisionV2::PassRob).unwrap();

    let rob = game.state().current_attempt.rob.as_ref().unwrap();
    assert_eq!(game.phase(), PhaseV2::BottomReveal);
    assert!(rob.queue.is_empty());
    assert_eq!(rob.candidate, caller);
    assert_eq!(rob.successful_rob_count, 0);
    assert_eq!(rob.rob_factor, 1);
    assert_eq!(
        game.state().current_attempt.status,
        AttemptStatusV2::LandlordResolved { landlord: caller }
    );
    assert!(game.system_events().iter().any(|record| {
        record.event
            == SystemEventV2::RobbingResolved {
                attempt_index: 0,
                landlord: caller,
                successful_rob_count: 0,
                rob_factor: 1,
            }
    }));

    let replay = HuanleMatchV2::replay(6_006, &rules(), game.decision_events()).unwrap();
    assert_eq!(replay, game);
}

#[test]
// HCV1-ROB-001, HCV1-ROB-002, HCV1-ROB-003.
fn caller_can_reclaim_once_after_another_eligible_seat_robs() {
    let mut configured_rules = rules();
    configured_rules.robbing.caller_can_reclaim = true;
    let mut game = HuanleMatchV2::new(6_007, &configured_rules).unwrap();
    advance_to_calling(&mut game);
    let caller = game.state().current_attempt.call.unwrap().current_player;
    let second = next_seat(caller);
    let third = next_seat(second);

    game.apply_call(caller, CallDecisionV2::CallLandlord)
        .unwrap();
    assert_eq!(
        game.state()
            .current_attempt
            .rob
            .as_ref()
            .unwrap()
            .queue
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![second, third, caller]
    );

    game.apply_rob(second, RobDecisionV2::Rob).unwrap();
    let rob = game.state().current_attempt.rob.as_ref().unwrap();
    assert_eq!(rob.candidate, second);
    assert_eq!(rob.successful_rob_count, 1);
    assert_eq!(rob.rob_factor, 2);
    assert_eq!(rob_front(&game), third);

    game.apply_rob(third, RobDecisionV2::PassRob).unwrap();
    assert_eq!(rob_front(&game), caller);
    game.apply_rob(caller, RobDecisionV2::Rob).unwrap();

    let rob = game.state().current_attempt.rob.as_ref().unwrap();
    assert_eq!(game.phase(), PhaseV2::BottomReveal);
    assert_eq!(rob.candidate, caller);
    assert_eq!(rob.successful_rob_count, 2);
    assert_eq!(rob.rob_factor, 4);
    assert_eq!(rob.acted, [true, true, true]);
    assert_eq!(
        game.state().current_attempt.status,
        AttemptStatusV2::LandlordResolved { landlord: caller }
    );

    let replay = HuanleMatchV2::replay(6_007, &configured_rules, game.decision_events()).unwrap();
    assert_eq!(replay, game);
}

#[test]
// HCV1-ROB-001, HCV1-ROB-002.
fn passed_initial_caller_is_never_queued_but_the_positive_caller_can_reclaim() {
    let mut configured_rules = rules();
    configured_rules.robbing.caller_can_reclaim = true;
    let mut game = HuanleMatchV2::new(6_008, &configured_rules).unwrap();
    advance_to_calling(&mut game);
    let first = game.state().current_attempt.call.unwrap().current_player;
    game.apply_call(first, CallDecisionV2::PassCall).unwrap();
    let caller = game.state().current_attempt.call.unwrap().current_player;
    let robber = next_seat(caller);
    assert_eq!(next_seat(robber), first);

    game.apply_call(caller, CallDecisionV2::CallLandlord)
        .unwrap();
    let rob = game.state().current_attempt.rob.as_ref().unwrap();
    assert!(!rob.eligible[usize::from(first)]);
    assert_eq!(
        rob.queue.iter().copied().collect::<Vec<_>>(),
        vec![robber, caller]
    );
    assert!(game.legal_rob_actions(first).unwrap().is_empty());

    game.apply_rob(robber, RobDecisionV2::Rob).unwrap();
    game.apply_rob(caller, RobDecisionV2::Rob).unwrap();

    let rob = game.state().current_attempt.rob.as_ref().unwrap();
    assert_eq!(game.phase(), PhaseV2::BottomReveal);
    assert_eq!(rob.candidate, caller);
    assert_eq!(rob.successful_rob_count, 2);
    assert_eq!(rob.rob_factor, 4);
    assert!(!rob.acted[usize::from(first)]);
    assert!(game
        .state()
        .current_attempt
        .action_history
        .iter()
        .all(|record| { record.actor != first || !matches!(record.action, GameActionV2::Rob(_)) }));
}

#[test]
// HCV1-ROB-001, HCV1-ROB-003.
fn only_remaining_eligible_caller_resolves_without_a_self_rob() {
    let mut game = HuanleMatchV2::new(6_009, &rules()).unwrap();
    advance_to_calling(&mut game);
    let first = game.state().current_attempt.call.unwrap().current_player;
    game.apply_call(first, CallDecisionV2::PassCall).unwrap();
    let second = game.state().current_attempt.call.unwrap().current_player;
    game.apply_call(second, CallDecisionV2::PassCall).unwrap();
    let caller = game.state().current_attempt.call.unwrap().current_player;

    game.apply_call(caller, CallDecisionV2::CallLandlord)
        .unwrap();

    let rob = game.state().current_attempt.rob.as_ref().unwrap();
    assert_eq!(game.phase(), PhaseV2::BottomReveal);
    assert!(!rob.eligible[usize::from(first)]);
    assert!(!rob.eligible[usize::from(second)]);
    assert!(rob.eligible[usize::from(caller)]);
    assert!(rob.queue.is_empty());
    assert_eq!(rob.candidate, caller);
    assert_eq!(rob.successful_rob_count, 0);
    assert_eq!(rob.rob_factor, 1);
    assert_eq!(
        game.state().current_attempt.status,
        AttemptStatusV2::LandlordResolved { landlord: caller }
    );
    assert!(game.legal_rob_actions(caller).unwrap().is_empty());

    let replay = HuanleMatchV2::replay(6_009, &rules(), game.decision_events()).unwrap();
    assert_eq!(replay, game);

    let mut missing_automatic_resolution = game.decision_events().to_vec();
    assert!(matches!(
        missing_automatic_resolution.pop(),
        Some(MatchDecisionEventV2::LandlordResolved { .. })
    ));
    assert!(matches!(
        HuanleMatchV2::replay(6_009, &rules(), &missing_automatic_resolution),
        Err(MatchError::ReplayEventMismatch { .. })
    ));
}

#[test]
// HCV1-ROB-001.
fn rob_actions_are_turn_bound_transactional_and_cannot_bypass_the_queue() {
    let mut game = HuanleMatchV2::new(6_010, &rules()).unwrap();
    advance_to_calling(&mut game);
    let caller = game.state().current_attempt.call.unwrap().current_player;
    let first_robber = next_seat(caller);
    let second_robber = next_seat(first_robber);
    game.apply_call(caller, CallDecisionV2::CallLandlord)
        .unwrap();

    let before_wrong_actor = game.clone();
    assert!(matches!(
        game.apply_rob(second_robber, RobDecisionV2::PassRob),
        Err(MatchError::RobOutOfTurn { expected, actual })
            if expected == first_robber && actual == second_robber
    ));
    assert_eq!(game, before_wrong_actor);
    assert!(matches!(
        game.record_accepted_action(first_robber, &GameActionV2::Rob(RobDecisionV2::Rob)),
        Err(MatchError::ActionRequiresOwningStateMachine { phase: "robbing" })
    ));
    assert_eq!(game, before_wrong_actor);

    game.apply_rob(first_robber, RobDecisionV2::PassRob)
        .unwrap();
    let before_reused_seat = game.clone();
    assert!(matches!(
        game.apply_rob(first_robber, RobDecisionV2::Rob),
        Err(MatchError::RobOutOfTurn { expected, actual })
            if expected == second_robber && actual == first_robber
    ));
    assert_eq!(game, before_reused_seat);
}
