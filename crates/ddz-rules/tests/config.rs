use ddz_rules::{RewardMode, RuleConfig, RuleProfile};

#[test]
fn constructors_produce_valid_profiles_and_stable_hashes() {
    let douzero = RuleConfig::douzero_post_bid(50, RewardMode::AverageDifferencePoints);
    douzero.validate().expect("douzero rules");
    assert_eq!(douzero.profile, RuleProfile::DouzeroPostBid);
    assert_eq!(douzero.rules_hash().unwrap(), douzero.rules_hash().unwrap());

    let huanle = RuleConfig::huanle_classic(51, [0; 18]);
    huanle.validate().expect("huanle rules");
    assert_eq!(huanle.profile, RuleProfile::HuanleClassic);
    assert_ne!(douzero.rules_hash().unwrap(), huanle.rules_hash().unwrap());
}

#[test]
fn unknown_yaml_fields_are_rejected() {
    let yaml = format!(
        "{}\nunexpected: true\n",
        include_str!("../examples/douzero_post_bid.yaml")
    );
    assert!(RuleConfig::from_yaml_str(&yaml).is_err());
}

#[test]
fn shipped_yaml_examples_parse() {
    let douzero = RuleConfig::from_yaml_str(include_str!("../examples/douzero_post_bid.yaml"))
        .expect("douzero example");
    assert_eq!(douzero.profile, RuleProfile::DouzeroPostBid);

    let huanle = RuleConfig::from_yaml_str(include_str!(
        "../examples/huanle_classic.template.yaml"
    ))
    .expect("huanle template");
    assert_eq!(huanle.profile, RuleProfile::HuanleClassic);
}

#[test]
fn huanle_during_deal_schedule_is_explicit_and_monotone() {
    let mut at_zero = [0_u32; 18];
    at_zero[0] = 4;
    assert!(RuleConfig::huanle_classic(52, at_zero).validate().is_err());

    let mut wrong_factor = [0_u32; 18];
    wrong_factor[4] = 2;
    assert!(RuleConfig::huanle_classic(53, wrong_factor)
        .validate()
        .is_err());

    let mut increasing = [0_u32; 18];
    increasing[4] = 3;
    increasing[8] = 4;
    assert!(RuleConfig::huanle_classic(54, increasing)
        .validate()
        .is_err());

    let mut valid = [0_u32; 18];
    valid[4] = 4;
    valid[8] = 3;
    RuleConfig::huanle_classic(55, valid)
        .validate()
        .expect("4 then 3 is valid");
}
