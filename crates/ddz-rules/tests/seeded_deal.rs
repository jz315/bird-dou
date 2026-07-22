use ddz_rules::{deal_post_bid, RuleConfig, POST_BID_LANDLORD, SHUFFLE_ALGORITHM};

const DOUZERO_POST_BID_YAML: &str = include_str!("../../../configs/rules/douzero_post_bid.yaml");

fn rules() -> RuleConfig {
    RuleConfig::from_yaml_str(DOUZERO_POST_BID_YAML).expect("checked-in profile must parse")
}

#[test]
fn seeded_deals_are_reproducible_and_complete() {
    let first = deal_post_bid(7, rules()).expect("seeded deal must initialize");
    let second = deal_post_bid(7, rules()).expect("same seeded deal must initialize");
    let other = deal_post_bid(8, rules()).expect("different seeded deal must initialize");

    assert_eq!(SHUFFLE_ALGORITHM, "splitmix64_fisher_yates_v1");
    assert_eq!(first.state(), second.state());
    assert_ne!(first.state(), other.state());
    assert_eq!(first.state().landlord, Some(POST_BID_LANDLORD));
    assert_eq!(first.state().cards_left, [20, 17, 17]);
}

#[test]
fn seed_seven_pins_the_physical_shuffle_contract() {
    let game = deal_post_bid(7, rules()).expect("seeded deal must initialize");

    assert_eq!(
        game.state().hands,
        [
            [1, 0, 2, 1, 1, 2, 2, 2, 0, 4, 2, 3, 0, 0, 0],
            [3, 1, 1, 2, 1, 1, 2, 0, 3, 0, 1, 0, 1, 0, 1],
            [0, 3, 1, 1, 2, 1, 0, 2, 1, 0, 1, 1, 3, 1, 0],
        ]
    );
    assert_eq!(
        game.state().bottom_cards,
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    );
}
