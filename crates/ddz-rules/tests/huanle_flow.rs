mod common;

use ddz_core::{
    CallAction, GameAction, GameEventKind, LandlordSelectionState, Phase, RevealAction, RobAction,
    SystemEvent,
};

use common::{continue_predeal, current, huanle_rules, new_huanle};

#[test]
fn all_pass_redeals_inside_the_same_match_and_keeps_history() {
    let mut game = new_huanle(huanle_rules(10), 42);
    continue_predeal(&mut game);
    for _ in 0..3 {
        let actor = current(&game);
        game.step(actor, GameAction::Call(CallAction::Pass)).unwrap();
    }
    assert_eq!(game.state().deal.attempt, 1);
    assert_eq!(game.state().phase, Phase::PreDeal);
    assert!(game.state().outcome.is_none());
    assert!(game.state().history.iter().any(|event| matches!(
        event.kind,
        GameEventKind::System(SystemEvent::Redeal {
            from_attempt: 0,
            to_attempt: 1
        })
    )));
}

#[test]
fn first_revealer_becomes_landlord_when_everyone_passes() {
    let mut game = new_huanle(huanle_rules(11), 7);
    let first = current(&game);
    game.step(first, GameAction::Reveal(RevealAction::Reveal))
        .unwrap();
    while game.state().phase == Phase::PreDeal {
        let actor = current(&game);
        game.step(actor, GameAction::Reveal(RevealAction::Continue))
            .unwrap();
    }
    assert_eq!(current(&game), first);
    for _ in 0..3 {
        let actor = current(&game);
        game.step(actor, GameAction::Call(CallAction::Pass)).unwrap();
    }
    assert_eq!(game.state().deal.attempt, 0);
    assert_eq!(game.state().landlord(), Some(first));
    assert_eq!(game.state().stake.reveal_factor, 5);
    assert!(matches!(
        &game.state().landlord_selection,
        LandlordSelectionState::Resolved(_)
    ));
}

#[test]
fn call_rob_pass_reclaim_produces_two_rob_doublings() {
    let mut game = new_huanle(huanle_rules(12), 99);
    continue_predeal(&mut game);
    let caller = current(&game);
    game.step(caller, GameAction::Call(CallAction::CallLandlord))
        .unwrap();
    let first_robber = current(&game);
    game.step(first_robber, GameAction::Rob(RobAction::RobLandlord))
        .unwrap();
    assert_eq!(game.state().stake.rob_exponent, 1);
    let second_robber = current(&game);
    game.step(second_robber, GameAction::Rob(RobAction::Pass))
        .unwrap();
    assert_eq!(current(&game), caller);
    game.step(caller, GameAction::Rob(RobAction::RobLandlord))
        .unwrap();
    assert_eq!(game.state().landlord(), Some(caller));
    assert_eq!(game.state().landlord_selection.successful_robs(), 2);
    assert_eq!(game.state().stake.rob_exponent, 2);
    assert_eq!(game.state().phase, Phase::PostBottomReveal);
}

#[test]
fn illegal_phase_action_is_transactional() {
    let mut game = new_huanle(huanle_rules(13), 123);
    let before = game.state().clone();
    let actor = current(&game);
    assert!(game
        .step(actor, GameAction::Call(CallAction::CallLandlord))
        .is_err());
    assert_eq!(game.state(), &before);
}
