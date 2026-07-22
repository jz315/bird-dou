use ddz_rules::{
    AirplaneRules, AttachmentMultiplicity, BiddingMode, FourWithTwoRules, RewardMode, RuleConfig,
    RuleConfigError, RuleProfile,
};

const DOUZERO_POST_BID_YAML: &str = include_str!("../../../configs/rules/douzero_post_bid.yaml");
const CANONICAL_FULL_YAML: &str = include_str!("../../../configs/rules/canonical_full.yaml");

#[test]
fn douzero_profile_parses_with_exact_compatibility_settings() {
    let config = RuleConfig::from_yaml_str(DOUZERO_POST_BID_YAML)
        .expect("the checked-in DouZero profile must be valid");

    assert_eq!(config.profile, RuleProfile::DouzeroPostBid);
    assert_eq!(config.bidding.mode, BiddingMode::Disabled);
    assert_eq!(config.bidding.max_bid, None);
    assert!(config.landlord_plays_first);
    assert!(config.bottom_cards_public);
    assert!(!config.doubling_enabled);
    assert_eq!(config.bomb_multiplier, 2);
    assert_eq!(config.rocket_multiplier, 2);
    assert_eq!(
        config.four_with_two,
        FourWithTwoRules {
            two_singles_enabled: true,
            two_pairs_enabled: true,
            single_attachments: AttachmentMultiplicity::MayShareRank,
            pair_attachments: AttachmentMultiplicity::DistinctRanks,
        }
    );
    assert_eq!(
        config.airplane,
        AirplaneRules {
            single_attachments: AttachmentMultiplicity::MayShareRank,
            pair_attachments: AttachmentMultiplicity::DistinctRanks,
        }
    );
    assert_eq!(config.reward_mode, RewardMode::AverageDifferencePoints);
}

#[test]
fn canonical_profile_makes_every_required_platform_choice_explicit() {
    let config = RuleConfig::from_yaml_str(CANONICAL_FULL_YAML)
        .expect("the checked-in canonical profile must be valid");

    assert_eq!(config.profile, RuleProfile::CanonicalFullLegacyV1);
    assert_eq!(config.bidding.mode, BiddingMode::Score);
    assert_eq!(config.bidding.max_bid, Some(3));
    assert!(config.landlord_plays_first);
    assert!(config.bottom_cards_public);
    assert!(config.doubling_enabled);
    assert_eq!(config.bomb_multiplier, 2);
    assert_eq!(config.rocket_multiplier, 2);
    assert!(config.spring.landlord_spring_enabled);
    assert!(config.spring.anti_spring_enabled);
    assert_eq!(config.spring.multiplier, 2);
    assert_eq!(
        config.four_with_two,
        FourWithTwoRules {
            two_singles_enabled: true,
            two_pairs_enabled: true,
            single_attachments: AttachmentMultiplicity::MayShareRank,
            pair_attachments: AttachmentMultiplicity::DistinctRanks,
        }
    );
    assert_eq!(
        config.airplane,
        AirplaneRules {
            single_attachments: AttachmentMultiplicity::DistinctRanks,
            pair_attachments: AttachmentMultiplicity::DistinctRanks,
        }
    );
    assert!(config.bidding.redeal_on_all_pass);
    assert_eq!(config.score_cap, None);
    assert_eq!(config.reward_mode, RewardMode::RawScore);
}

#[test]
fn yaml_round_trip_preserves_a_valid_profile() {
    let config = RuleConfig::from_yaml_str(CANONICAL_FULL_YAML).expect("profile is valid");
    let serialized = serde_yaml_ng::to_string(&config).expect("configuration must serialize");
    let reparsed = RuleConfig::from_yaml_str(&serialized).expect("round trip must stay valid");

    assert_eq!(reparsed, config);
}

#[test]
fn unknown_yaml_fields_are_rejected() {
    let yaml = format!("{CANONICAL_FULL_YAML}\nunknown_platform_default: true\n");

    assert!(matches!(
        RuleConfig::from_yaml_str(&yaml),
        Err(RuleConfigError::Yaml(_))
    ));
}

#[test]
fn unsupported_schema_versions_are_rejected() {
    let mut config = RuleConfig::from_yaml_str(CANONICAL_FULL_YAML).expect("profile is valid");
    config.schema_version = 2;

    assert!(matches!(
        config.validate(),
        Err(RuleConfigError::UnsupportedSchemaVersion {
            expected: 1,
            actual: 2
        })
    ));
}

#[test]
fn zero_configuration_ids_are_rejected() {
    let mut config = RuleConfig::from_yaml_str(CANONICAL_FULL_YAML).expect("profile is valid");
    config.rule_config_id = 0;

    assert!(matches!(
        config.validate(),
        Err(RuleConfigError::InvalidField {
            field: "rule_config_id",
            ..
        })
    ));
}

#[test]
fn bidding_field_relationships_are_validated() {
    let mut config = RuleConfig::from_yaml_str(CANONICAL_FULL_YAML).expect("profile is valid");
    config.bidding.max_bid = None;

    assert!(matches!(
        config.validate(),
        Err(RuleConfigError::InvalidField {
            field: "bidding.max_bid",
            ..
        })
    ));
}

#[test]
fn invalid_multipliers_and_score_caps_are_rejected() {
    let mut config = RuleConfig::from_yaml_str(CANONICAL_FULL_YAML).expect("profile is valid");
    config.bomb_multiplier = 0;
    assert!(matches!(
        config.validate(),
        Err(RuleConfigError::InvalidField {
            field: "bomb_multiplier",
            ..
        })
    ));

    config.bomb_multiplier = 2;
    config.score_cap = Some(0);
    assert!(matches!(
        config.validate(),
        Err(RuleConfigError::InvalidField {
            field: "score_cap",
            ..
        })
    ));
}

#[test]
fn spring_flags_and_multiplier_must_agree() {
    let mut config = RuleConfig::from_yaml_str(CANONICAL_FULL_YAML).expect("profile is valid");
    config.spring.multiplier = 1;
    assert!(matches!(
        config.validate(),
        Err(RuleConfigError::InvalidField {
            field: "spring.multiplier",
            ..
        })
    ));

    config.spring.landlord_spring_enabled = false;
    config.spring.anti_spring_enabled = false;
    config.spring.multiplier = 2;
    assert!(matches!(
        config.validate(),
        Err(RuleConfigError::InvalidField {
            field: "spring.multiplier",
            ..
        })
    ));
}

#[test]
fn douzero_profile_drift_is_rejected() {
    let mut config = RuleConfig::from_yaml_str(DOUZERO_POST_BID_YAML).expect("profile is valid");
    config.doubling_enabled = true;

    assert!(matches!(
        config.validate(),
        Err(RuleConfigError::IncompatibleProfile {
            profile: RuleProfile::DouzeroPostBid,
            field: "doubling_enabled",
            ..
        })
    ));
}

#[test]
fn all_three_douzero_reward_modes_are_selectable() {
    let mut config = RuleConfig::from_yaml_str(DOUZERO_POST_BID_YAML).expect("profile is valid");

    for reward_mode in [
        RewardMode::WinPercentage,
        RewardMode::AverageDifferencePoints,
        RewardMode::LogAverageDifferencePoints,
    ] {
        config.reward_mode = reward_mode;
        assert!(config.validate().is_ok());
    }

    config.reward_mode = RewardMode::RawScore;
    assert!(matches!(
        config.validate(),
        Err(RuleConfigError::IncompatibleProfile {
            field: "reward_mode",
            ..
        })
    ));
}
