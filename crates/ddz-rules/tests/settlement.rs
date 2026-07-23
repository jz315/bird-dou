use ddz_core::{DoublingState, RankCounts, Seat, SeatSet};
use ddz_rules::{settle_game, Game, RewardMode, RuleConfig};

#[test]
fn pairwise_doubling_can_give_the_two_farmers_different_scores() {
    let landlord = Seat::ZERO;
    let first_farmer = Seat::ONE;
    let second_farmer = Seat::TWO;

    let post_bid = RuleConfig::douzero_post_bid(30, RewardMode::WinPercentage);
    let mut state = Game::new_post_bid(post_bid, 3, landlord)
        .expect("post-bid game")
        .into_state();

    let mut rules = RuleConfig::huanle_classic(31, [0; 18]);
    rules.settlement.spring.landlord_spring_enabled = false;
    rules.settlement.spring.farmer_spring_enabled = false;
    rules.settlement.spring.factor = 1;
    rules.validate().expect("test rules");

    state.rule_config_id = rules.rule_config_id;
    let mut doubled = SeatSet::empty();
    doubled.insert(landlord);
    doubled.insert(first_farmer);
    state.doubling = DoublingState::Resolved {
        eligible: SeatSet::all(),
        doubled,
    };

    let winning_hand = state.hands[landlord];
    state.hands[landlord] = RankCounts::empty();
    state.card_play.played_cards[landlord] = winning_hand;
    state.card_play.non_pass_plays[landlord] = 1;

    settle_game(&mut state, landlord, &rules).expect("settlement");
    let payoff = state.outcome.expect("outcome").payoff;
    assert_eq!(payoff[landlord], 6);
    assert_eq!(payoff[first_farmer], -4);
    assert_eq!(payoff[second_farmer], -2);
    assert_eq!(payoff.iter().map(|(_, value)| *value).sum::<i64>(), 0);
}
