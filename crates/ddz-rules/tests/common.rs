#![allow(dead_code)]

use ddz_core::{GameAction, Phase, RevealAction, Seat};
use ddz_rules::{EconomyContext, Game, RuleConfig};

pub fn huanle_rules(rule_config_id: u32) -> RuleConfig {
    RuleConfig::huanle_classic(rule_config_id, [0; 18])
}

pub fn huanle_rules_with_round_one_reveal(rule_config_id: u32) -> RuleConfig {
    let mut schedule = [0; 18];
    schedule[1] = 4;
    RuleConfig::huanle_classic(rule_config_id, schedule)
}

pub fn continue_predeal(game: &mut Game) {
    while game.state().phase == Phase::PreDeal {
        let actor = current(game);
        game.step(actor, GameAction::Reveal(RevealAction::Continue))
            .expect("predeal continue");
    }
}

pub fn new_huanle(rules: RuleConfig, seed: u64) -> Game {
    Game::new_huanle(rules, seed, EconomyContext::unlimited()).expect("valid game")
}

pub fn current(game: &Game) -> Seat {
    game.state().current_player.expect("current player")
}
