use ddz_core::{GameAction, Seat};
use ddz_rules::{Game, RewardMode, RuleConfig};

#[test]
fn deterministic_legal_policy_finishes_a_post_bid_game() {
    let rules = RuleConfig::douzero_post_bid(40, RewardMode::AverageDifferencePoints);
    let mut game = Game::new_post_bid(rules, 1001, Seat::ZERO).expect("post-bid game");

    for _ in 0..1_024 {
        if game.state().is_terminal() {
            let payoff = &game.state().outcome.as_ref().expect("outcome").payoff;
            assert_eq!(payoff.iter().map(|(_, value)| *value).sum::<i64>(), 0);
            game.state().validate().expect("terminal invariant");
            return;
        }
        let actor = game.state().current_player.expect("current player");
        let actions = game.legal_actions().expect("legal actions");
        let selected = actions
            .iter()
            .copied()
            .filter_map(|action| match action {
                GameAction::Play(movement) if !movement.is_pass() => {
                    Some((movement.total_cards(), action))
                }
                GameAction::Play(_) => None,
                _ => unreachable!("post-bid game only exposes play actions"),
            })
            .max_by_key(|(cards, _)| *cards)
            .map(|(_, action)| action)
            .unwrap_or_else(|| {
                actions
                    .iter()
                    .copied()
                    .find(|action| matches!(action, GameAction::Play(movement) if movement.is_pass()))
                    .expect("follow position always allows pass")
            });
        game.step(actor, selected).expect("legal transition");
    }
    panic!("deterministic legal policy did not finish within 1,024 decisions");
}
