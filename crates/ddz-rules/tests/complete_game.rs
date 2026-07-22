use ddz_core::{BidAction, BidState, DoubleAction, GameAction, Phase, Role};
use ddz_rules::{deal_complete, BiddingMode, PostBidGame, RuleConfig};

const CANONICAL_FULL_YAML: &str = include_str!("../../../configs/rules/canonical_full.yaml");

fn rules() -> RuleConfig {
    RuleConfig::from_yaml_str(CANONICAL_FULL_YAML).expect("canonical profile must be valid")
}

fn assert_round_trip(game: &PostBidGame) {
    let bytes = game.serialize_state().unwrap();
    let restored = PostBidGame::deserialize_state(&bytes, *game.rules()).unwrap();
    assert_eq!(restored, *game);
    assert_eq!(restored.serialize_state().unwrap(), bytes);
    assert_eq!(
        restored.legal_actions().unwrap(),
        game.legal_actions().unwrap()
    );
}

#[test]
fn score_bidding_hides_bottom_resolves_landlord_and_enters_doubling() {
    let mut game = deal_complete(40, rules()).unwrap();
    assert_eq!(game.state().phase, Phase::Bidding);
    assert_eq!(game.state().current_player, 1);
    assert_eq!(game.state().landlord, None);
    assert_eq!(game.state().cards_left, [17, 17, 17]);
    assert_eq!(game.state().hands.iter().flatten().sum::<u8>(), 51);
    let observation = game.observe(1).unwrap();
    assert_eq!(observation.role, Role::Unassigned);
    assert_eq!(observation.public_bottom_cards.iter().sum::<u8>(), 0);
    assert_eq!(observation.unknown_pool.iter().sum::<u8>(), 37);
    assert_eq!(
        game.legal_actions().unwrap(),
        vec![
            GameAction::Bid(BidAction::Pass),
            GameAction::Bid(BidAction::Score(1)),
            GameAction::Bid(BidAction::Score(2)),
            GameAction::Bid(BidAction::Score(3)),
        ]
    );
    assert_round_trip(&game);

    let first = game
        .step_with_undo(&GameAction::Bid(BidAction::Score(2)))
        .unwrap();
    assert_eq!(game.state().current_player, 2);
    game.undo(&first.1).unwrap();
    assert_eq!(game.state().current_player, 1);
    game.step(&GameAction::Bid(BidAction::Score(2))).unwrap();
    game.step(&GameAction::Bid(BidAction::Pass)).unwrap();
    game.step(&GameAction::Bid(BidAction::Pass)).unwrap();
    assert_eq!(game.state().phase, Phase::Doubling);
    assert_eq!(game.state().landlord, Some(1));
    assert_eq!(game.state().cards_left, [17, 20, 17]);
    assert_eq!(game.state().hands.iter().flatten().sum::<u8>(), 54);
    assert_eq!(
        game.observe(0).unwrap().public_bottom_cards,
        game.state().bottom_cards
    );
    assert_eq!(game.observe(2).unwrap().bid_history.len(), 3);
    assert_eq!(
        game.legal_actions().unwrap(),
        vec![
            GameAction::Double(DoubleAction::Decline),
            GameAction::Double(DoubleAction::Double),
        ]
    );
    assert_round_trip(&game);

    game.step(&GameAction::Double(DoubleAction::Double))
        .unwrap();
    game.step(&GameAction::Double(DoubleAction::Decline))
        .unwrap();
    game.step(&GameAction::Double(DoubleAction::Double))
        .unwrap();
    assert_eq!(game.state().phase, Phase::CardPlay);
    assert_eq!(game.state().current_player, 1);
    assert_eq!(game.state().multiplier_exp, 2);
    assert_round_trip(&game);
}

#[test]
fn all_pass_is_an_explicit_zero_payoff_redeal_terminal() {
    let mut game = deal_complete(41, rules()).unwrap();
    for _ in 0..3 {
        game.step(&GameAction::Bid(BidAction::Pass)).unwrap();
    }
    assert!(game.is_terminal());
    assert_eq!(game.state().phase, Phase::Terminal);
    assert_eq!(game.state().landlord, None);
    assert_eq!(game.state().raw_payoff, [0, 0, 0]);
    assert_eq!(game.state().bid_state, BidState::AllPass);
    assert!(game.legal_actions().unwrap().is_empty());
    assert_round_trip(&game);
}

#[test]
fn rob_mode_and_hidden_bottom_profile_are_executable() {
    let mut rob_rules = rules();
    rob_rules.bidding.mode = BiddingMode::Rob;
    rob_rules.bidding.max_bid = None;
    rob_rules.bottom_cards_public = false;
    rob_rules.validate().unwrap();
    let mut game = deal_complete(42, rob_rules).unwrap();
    let caller = game.state().current_player;
    assert_eq!(
        game.legal_actions().unwrap(),
        vec![
            GameAction::Bid(BidAction::Pass),
            GameAction::Bid(BidAction::Call),
        ]
    );
    game.step(&GameAction::Bid(BidAction::Call)).unwrap();
    assert_eq!(
        game.legal_actions().unwrap(),
        vec![
            GameAction::Bid(BidAction::Pass),
            GameAction::Bid(BidAction::Rob),
        ]
    );
    game.step(&GameAction::Bid(BidAction::Pass)).unwrap();
    game.step(&GameAction::Bid(BidAction::Pass)).unwrap();
    assert_eq!(game.state().landlord, Some(caller));
    assert_eq!(game.state().phase, Phase::Doubling);
    assert_eq!(
        game.observe((caller + 1) % 3)
            .unwrap()
            .public_bottom_cards
            .iter()
            .sum::<u8>(),
        0
    );
    assert_round_trip(&game);
}

#[test]
fn complete_games_terminate_with_bid_score_and_zero_sum_scoring() {
    for seed in 50..60 {
        let mut game = deal_complete(seed, rules()).unwrap();
        let maximum = game
            .legal_actions()
            .unwrap()
            .into_iter()
            .find(|action| *action == GameAction::Bid(BidAction::Score(3)))
            .unwrap();
        game.step(&maximum).unwrap();
        while game.state().phase == Phase::Doubling {
            game.step(&GameAction::Double(DoubleAction::Decline))
                .unwrap();
        }
        while !game.is_terminal() {
            let actions = game.legal_actions().unwrap();
            game.step(actions.last().unwrap()).unwrap();
        }
        let landlord = usize::from(game.state().landlord.unwrap());
        let unit = 2_i32.pow(u32::from(game.state().multiplier_exp)) * 3;
        let unsigned_unit = u32::try_from(unit).unwrap();
        assert_eq!(game.state().raw_payoff.iter().sum::<i32>(), 0);
        assert_eq!(
            game.state().raw_payoff[landlord].unsigned_abs(),
            2 * unsigned_unit
        );
        for seat in 0..3 {
            if seat != landlord {
                assert_eq!(game.state().raw_payoff[seat].unsigned_abs(), unsigned_unit);
            }
        }
        assert_round_trip(&game);
    }
}
