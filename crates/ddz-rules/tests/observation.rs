mod common;

use ddz_core::{GameAction, Phase, RevealAction, Seat};

use common::{current, huanle_rules_with_round_one_reveal, new_huanle};

#[test]
fn predeal_observation_accounts_for_the_full_unseen_deck() {
    let game = new_huanle(huanle_rules_with_round_one_reveal(20), 9);
    let observer = Seat::ZERO;
    let view = game.observe(observer).expect("predeal observation");
    assert_eq!(view.phase, Phase::PreDeal);
    assert_eq!(view.own_hand.card_count(), 0);
    assert_eq!(view.unknown_pool.card_count(), 54);
    view.validate().expect("information-set invariant");
}

#[test]
fn a_revealed_hand_is_removed_from_the_unknown_pool() {
    let mut game = new_huanle(huanle_rules_with_round_one_reveal(21), 17);

    while game.state().phase == Phase::PreDeal {
        let actor = current(&game);
        game.step(actor, GameAction::Reveal(RevealAction::Continue))
            .expect("finish predeal reveal round");
    }
    assert_eq!(game.state().phase, Phase::Dealing);
    assert_eq!(game.state().deal.rounds_dealt, 1);

    let revealer = current(&game);
    game.step(revealer, GameAction::Reveal(RevealAction::Reveal))
        .expect("during-deal reveal");

    let observer = revealer.next();
    let view = game.observe(observer).expect("observation after reveal");
    assert_eq!(view.own_hand.card_count(), 1);
    assert_eq!(view.revealed_hands[revealer].unwrap().card_count(), 1);
    assert_eq!(view.unknown_pool.card_count(), 52);
    view.validate().expect("information-set invariant");
}

#[test]
fn history_free_observation_preserves_the_public_state() {
    let mut game = new_huanle(huanle_rules_with_round_one_reveal(22), 25);
    let actor = current(&game);
    game.step(actor, GameAction::Reveal(RevealAction::Continue))
        .expect("predeal action");

    let with_history = game.observe(Seat::ZERO).expect("observation with history");
    let without_history = game
        .observe_without_history(Seat::ZERO)
        .expect("observation without history");
    assert!(!with_history.history.is_empty());
    assert!(without_history.history.is_empty());

    let mut expected = with_history;
    expected.history.clear();
    assert_eq!(without_history, expected);
}
