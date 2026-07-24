use guandan_rules::{
    report_duty, Rank, ReportDuty, RuleConfig, RuleProfile, RULE_CONFIG_SCHEMA_VERSION,
};

#[test]
fn repository_profile_parses() {
    let yaml = include_str!("../../../configs/rules/guandan_two_deck.yaml");
    let config = RuleConfig::from_yaml_str(yaml).unwrap();

    assert_eq!(config.schema_version, RULE_CONFIG_SCHEMA_VERSION);
    assert_eq!(config.profile, RuleProfile::GuandanTwoDeck);
    assert_eq!(config.starting_level, Rank::Two);
}

#[test]
fn report_thresholds_match_the_tournament_profile() {
    let config = RuleConfig::tournament();

    assert_eq!(report_duty(11, &config), ReportDuty::None);
    assert_eq!(report_duty(10, &config), ReportDuty::AnswerOnRequest);
    assert_eq!(report_duty(6, &config), ReportDuty::AnnounceImmediately);
    assert_eq!(report_duty(0, &config), ReportDuty::None);
}
