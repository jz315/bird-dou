use ddz_rules::{
    RuleConfig, RuleConfigError, RuleConfigV1, RuleConfigV2, RuleProfile, VersionedRuleConfig,
    RULE_CONFIG_V1_SCHEMA_VERSION, RULE_CONFIG_V2_SCHEMA_VERSION,
};

const DOUZERO_POST_BID_YAML: &str = include_str!("../../../configs/rules/douzero_post_bid.yaml");
const CANONICAL_FULL_YAML: &str = include_str!("../../../configs/rules/canonical_full.yaml");
const HUANLE_V2_FIXTURE: &str =
    include_str!("../../../tests/rules/huanle_classic_v1/parser_fixture_v2.yaml");

#[test]
fn versioned_reader_preserves_both_legacy_profiles_byte_for_byte() {
    for yaml in [DOUZERO_POST_BID_YAML, CANONICAL_FULL_YAML] {
        let legacy = RuleConfig::from_yaml_str(yaml).expect("checked-in v1 profile must parse");
        let versioned = VersionedRuleConfig::from_yaml_str(yaml)
            .expect("versioned reader must preserve checked-in v1 profile");

        assert_eq!(versioned, VersionedRuleConfig::V1(legacy));
        assert_eq!(
            versioned.as_v1().expect("legacy profile remains available"),
            &legacy
        );
        assert_eq!(
            legacy.rules_hash().unwrap(),
            versioned.rules_hash().unwrap()
        );
        assert_eq!(
            serde_yaml_ng::to_string(&legacy).unwrap(),
            serde_yaml_ng::to_string(versioned.as_v1().unwrap()).unwrap()
        );
    }
}

#[test]
fn legacy_canonical_serialization_stays_canonical_full() {
    let config = RuleConfigV1::from_yaml_str(CANONICAL_FULL_YAML).unwrap();
    assert_eq!(config.profile, RuleProfile::CanonicalFullLegacyV1);

    let serialized = serde_yaml_ng::to_string(&config).unwrap();
    assert!(serialized.contains("profile: canonical_full"));
    assert!(!serialized.contains("canonical_full_legacy_v1"));
}

#[test]
fn explicit_huanle_v2_fixture_parses_and_hashes_stably() {
    let config = RuleConfigV2::from_yaml_str(HUANLE_V2_FIXTURE)
        .expect("fully explicit structural v2 fixture must parse");
    let versioned = VersionedRuleConfig::from_yaml_str(HUANLE_V2_FIXTURE).unwrap();

    assert_eq!(config.schema_version, RULE_CONFIG_V2_SCHEMA_VERSION);
    assert_eq!(config.profile, RuleProfile::HuanleClassicV1);
    assert_eq!(versioned, VersionedRuleConfig::V2(config));
    assert_eq!(versioned.as_v2().unwrap(), &config);
    assert_eq!(config.reveal.factor_by_cards_received.len(), 18);
    assert_eq!(config.rules_hash().unwrap().len(), 64);
    assert_eq!(
        config.rules_hash().unwrap(),
        versioned.rules_hash().unwrap()
    );

    let serialized = serde_yaml_ng::to_string(&config).unwrap();
    let reparsed = RuleConfigV2::from_yaml_str(&serialized).unwrap();
    assert_eq!(reparsed, config);
    assert_eq!(reparsed.rules_hash().unwrap(), config.rules_hash().unwrap());
}

#[test]
fn direct_v1_and_v2_readers_reject_the_other_schema_before_deserialization() {
    assert!(matches!(
        RuleConfigV1::from_yaml_str(HUANLE_V2_FIXTURE),
        Err(RuleConfigError::UnsupportedSchemaVersion {
            expected: RULE_CONFIG_V1_SCHEMA_VERSION,
            actual: RULE_CONFIG_V2_SCHEMA_VERSION,
        })
    ));
    assert!(matches!(
        RuleConfigV2::from_yaml_str(CANONICAL_FULL_YAML),
        Err(RuleConfigError::UnsupportedSchemaVersion {
            expected: RULE_CONFIG_V2_SCHEMA_VERSION,
            actual: RULE_CONFIG_V1_SCHEMA_VERSION,
        })
    ));

    let v2 = VersionedRuleConfig::from_yaml_str(HUANLE_V2_FIXTURE).unwrap();
    assert!(matches!(
        v2.as_v1(),
        Err(RuleConfigError::UnsupportedByLegacyEngine {
            schema_version: RULE_CONFIG_V2_SCHEMA_VERSION,
            profile: RuleProfile::HuanleClassicV1,
        })
    ));
    let v1 = VersionedRuleConfig::from_yaml_str(CANONICAL_FULL_YAML).unwrap();
    assert!(matches!(
        v1.as_v2(),
        Err(RuleConfigError::UnsupportedByV2Engine {
            schema_version: RULE_CONFIG_V1_SCHEMA_VERSION,
            profile: RuleProfile::CanonicalFullLegacyV1,
        })
    ));
}

#[test]
fn v2_rejects_legacy_profiles_and_v1_rejects_huanle_profile() {
    let legacy_in_v2 =
        HUANLE_V2_FIXTURE.replace("profile: huanle_classic_v1", "profile: canonical_full");
    assert!(matches!(
        RuleConfigV2::from_yaml_str(&legacy_in_v2),
        Err(RuleConfigError::IncompatibleProfile {
            profile: RuleProfile::CanonicalFullLegacyV1,
            field: "profile",
            ..
        })
    ));

    let huanle_in_v1 =
        CANONICAL_FULL_YAML.replace("profile: canonical_full", "profile: huanle_classic_v1");
    assert!(matches!(
        RuleConfigV1::from_yaml_str(&huanle_in_v1),
        Err(RuleConfigError::IncompatibleProfile {
            profile: RuleProfile::HuanleClassicV1,
            field: "schema_version",
            ..
        })
    ));
}

#[test]
fn every_previously_unresolved_v2_choice_is_required_without_a_default() {
    let missing_cases = [
        (
            "HCV1-CONFIG-DEAL-REVEAL-SCHEDULE",
            HUANLE_V2_FIXTURE.replace(
                "  factor_by_cards_received: [4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3]\n",
                "",
            ),
        ),
        (
            "HCV1-CONFIG-AIRPLANE-SINGLE-WINGS",
            HUANLE_V2_FIXTURE.replacen(
                "  airplane:\n    single_attachments: may_share_rank\n",
                "  airplane:\n",
                1,
            ),
        ),
        (
            "HCV1-CONFIG-FOUR-TWO-SINGLE-WINGS",
            HUANLE_V2_FIXTURE.replacen(
                "  four_with_two:\n    two_singles_enabled: true\n    two_pairs_enabled: true\n    single_attachments: may_share_rank\n",
                "  four_with_two:\n    two_singles_enabled: true\n    two_pairs_enabled: true\n",
                1,
            ),
        ),
        (
            "HCV1-CONFIG-SPRING",
            HUANLE_V2_FIXTURE.replacen(
                "  spring:\n    landlord_spring_enabled: false\n    anti_spring_enabled: false\n    multiplier: 1\n",
                "",
                1,
            ),
        ),
        (
            "HCV1-CONFIG-SCORE-CAP",
            HUANLE_V2_FIXTURE.replacen("  score_cap: null\n", "", 1),
        ),
        (
            "HCV1-CONFIG-CALLER-RECLAIM",
            HUANLE_V2_FIXTURE.replacen("  caller_can_reclaim: false\n", "", 1),
        ),
    ];

    for (case_id, missing) in missing_cases {
        assert!(
            matches!(
                RuleConfigV2::from_yaml_str(&missing),
                Err(RuleConfigError::Yaml(_))
            ),
            "{case_id} must not receive an implicit default"
        );
    }
}

#[test]
fn rules_hash_is_sensitive_to_explicit_unresolved_choices() {
    let config = RuleConfigV2::from_yaml_str(HUANLE_V2_FIXTURE).unwrap();
    let mut changed = config;
    changed.robbing.caller_can_reclaim = !changed.robbing.caller_can_reclaim;
    changed.validate().unwrap();

    assert_ne!(config.rules_hash().unwrap(), changed.rules_hash().unwrap());
}
