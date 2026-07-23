mod common;

use ddz_core::{GameAction, RevealAction};
use ddz_rules::{EconomyContext, Game, GameRestoreError};

use common::{current, huanle_rules, new_huanle};

#[test]
fn apply_with_undo_restores_the_exact_state() {
    let mut game = new_huanle(huanle_rules(60), 1234);
    let before = game.state().clone();
    let actor = current(&game);
    let (_, token) = game
        .apply_with_undo(actor, GameAction::Reveal(RevealAction::Continue))
        .expect("valid action");
    assert_ne!(game.state(), &before);
    game.undo(token).expect("matching undo token");
    assert_eq!(game.state(), &before);
}

#[test]
fn restore_rejects_a_different_match_seed() {
    let rules = huanle_rules(61);
    let game = new_huanle(rules.clone(), 321);
    let state = game.into_state();
    Game::restore(
        rules.clone(),
        321,
        EconomyContext::unlimited(),
        state.clone(),
    )
    .expect("same seed restores");
    assert!(matches!(
        Game::restore(rules, 322, EconomyContext::unlimited(), state),
        Err(GameRestoreError::DealPlanMismatch { .. })
    ));
}

#[test]
fn undo_token_is_bound_to_its_match_identity() {
    let mut first = new_huanle(huanle_rules(62), 1);
    let actor = current(&first);
    let (_, token) = first
        .apply_with_undo(actor, GameAction::Reveal(RevealAction::Continue))
        .expect("valid action");

    let mut second = new_huanle(huanle_rules(62), 2);
    assert!(second.undo(token).is_err());
}

#[test]
fn restore_rejects_a_state_from_the_other_rule_flow() {
    use ddz_core::Seat;
    use ddz_rules::RewardMode;

    let post_bid_rules = ddz_rules::RuleConfig::douzero_post_bid(
        63,
        RewardMode::WinPercentage,
    );
    let state = Game::new_post_bid(post_bid_rules, 7, Seat::ZERO)
        .expect("post-bid game")
        .into_state();
    let huanle = huanle_rules(63);
    assert!(matches!(
        Game::restore(huanle, 7, EconomyContext::unlimited(), state),
        Err(GameRestoreError::ProfileStateMismatch { .. })
    ));
}

#[test]
fn restore_rejects_stake_counters_that_drift_from_public_state() {
    let rules = huanle_rules(64);
    let game = new_huanle(rules.clone(), 44);
    let mut state = game.into_state();
    state.stake.rob_exponent = 1;
    assert!(matches!(
        Game::restore(rules, 44, EconomyContext::unlimited(), state),
        Err(GameRestoreError::RuleState(_))
    ));
}

#[test]
fn restore_rejects_a_zero_sum_but_incorrect_terminal_payoff() {
    use ddz_core::{GameAction, Seat};
    use ddz_rules::{RewardMode, RuleConfig};

    let rules = RuleConfig::douzero_post_bid(65, RewardMode::WinPercentage);
    let mut game = Game::new_post_bid(rules.clone(), 99, Seat::ZERO).expect("post-bid game");
    for _ in 0..1_024 {
        if game.state().is_terminal() {
            break;
        }
        let actor = game.state().current_player.expect("current player");
        let actions = game.legal_actions().expect("legal actions");
        let action = actions
            .iter()
            .copied()
            .filter_map(|action| match action {
                GameAction::Play(movement) if !movement.is_pass() => {
                    Some((movement.total_cards(), action))
                }
                GameAction::Play(_) => None,
                _ => None,
            })
            .max_by_key(|(count, _)| *count)
            .map(|(_, action)| action)
            .unwrap_or_else(|| actions[0]);
        game.step(actor, action).expect("legal play");
    }
    assert!(game.state().is_terminal(), "policy must finish within 1,024 decisions");
    let mut state = game.into_state();
    let landlord = state.landlord().expect("landlord");
    let farmer = landlord.next();
    let outcome = state.outcome.as_mut().expect("terminal outcome");
    outcome.payoff[landlord] += 1;
    outcome.payoff[farmer] -= 1;
    state.validate().expect("tampered payoff remains structurally zero-sum");
    assert!(matches!(
        Game::restore(rules, 99, EconomyContext::unlimited(), state),
        Err(GameRestoreError::RuleState(_))
    ));
}
