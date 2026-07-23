mod common;

use ddz_core::{
    CallAction, DoubleAction, GameAction, GameEventKind, Phase, PlayerEvent, RevealAction,
    RobAction, SeatMap,
};
use ddz_rules::{EconomyContext, Game, RuleConfig};

use common::{continue_predeal, current};

fn reach_post_bottom(mut game: Game) -> Game {
    continue_predeal(&mut game);
    let caller = current(&game);
    game.step(caller, GameAction::Call(CallAction::CallLandlord))
        .expect("call landlord");
    while game.state().phase == Phase::Robbing {
        let actor = current(&game);
        game.step(actor, GameAction::Rob(RobAction::Pass))
            .expect("pass rob");
    }
    assert_eq!(game.state().phase, Phase::PostBottomReveal);
    let landlord = current(&game);
    game.step(
        landlord,
        GameAction::Reveal(RevealAction::Continue),
    )
    .expect("decline post-bottom reveal");
    game
}

#[test]
fn in_progress_double_choices_are_hidden_until_everyone_finishes() {
    let rules = RuleConfig::huanle_classic(50, [0; 18]);
    let game = Game::new_huanle(rules, 55, EconomyContext::unlimited()).expect("game");
    let mut game = reach_post_bottom(game);
    assert_eq!(game.state().phase, Phase::Doubling);

    let first = current(&game);
    game.step(first, GameAction::Double(DoubleAction::Double))
        .expect("first double");
    let during = game.observe(first.next()).expect("public doubling view");
    assert!(during.history.iter().all(|event| !matches!(
        event.kind,
        GameEventKind::Player(PlayerEvent {
            action: GameAction::Double(_),
            ..
        })
    )));

    while game.state().phase == Phase::Doubling {
        let actor = current(&game);
        game.step(actor, GameAction::Double(DoubleAction::Decline))
            .expect("finish doubling");
    }
    let after = game.observe(first.next()).expect("resolved doubling view");
    assert!(after.history.iter().any(|event| matches!(
        event.kind,
        GameEventKind::Player(PlayerEvent {
            action: GameAction::Double(_),
            ..
        })
    )));
}

#[test]
fn balance_rules_expose_only_eligible_double_decisions() {
    let rules = RuleConfig::huanle_classic(51, [0; 18]);
    let provisional = Game::new_huanle(rules.clone(), 88, EconomyContext::unlimited())
        .expect("provisional game");
    let provisional = reach_post_bottom(provisional);
    let landlord = provisional.state().landlord().expect("landlord");
    let eligible_farmer = landlord.next();
    let ineligible_farmer = eligible_farmer.next();

    let mut balances = SeatMap::new([0_u64; 3]);
    balances[landlord] = 100;
    balances[eligible_farmer] = 100;
    balances[ineligible_farmer] = 0;
    let mut restricted_rules = rules;
    restricted_rules.doubling.minimum_balance_exclusive = 50;

    let fresh = Game::new_huanle(
        restricted_rules,
        88,
        EconomyContext::new(balances),
    )
    .expect("restricted game");
    let game = reach_post_bottom(fresh);
    assert_eq!(game.state().phase, Phase::Doubling);
    assert_eq!(game.state().current_player, Some(eligible_farmer));
    let actions = game.legal_actions().expect("eligible actions");
    assert_eq!(actions.len(), 2);
    assert!(actions.contains(&GameAction::Double(DoubleAction::Double)));
}
