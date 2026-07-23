use serde::{Deserialize, Serialize};

use super::{hash, RuleConfigError};

/// Current serialized rule-config schema. Runtime domain types are intentionally not version-suffixed.
pub const RULE_CONFIG_SCHEMA_VERSION: u32 = 1;

/// The only two supported game profiles.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleProfile {
    /// Exact post-bid environment used for DouZero comparison.
    DouzeroPostBid,
    /// Huanle-style reveal, call, rob, double, and card-play flow.
    HuanleClassic,
}

/// Whether attachment units inside one move must use distinct ranks.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentMultiplicity {
    DistinctRanks,
    MayShareRank,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FourWithTwoRules {
    pub two_singles_enabled: bool,
    pub two_pairs_enabled: bool,
    pub single_attachments: AttachmentMultiplicity,
    pub pair_attachments: AttachmentMultiplicity,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AirplaneRules {
    pub single_attachments: AttachmentMultiplicity,
    pub pair_attachments: AttachmentMultiplicity,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MoveRules {
    pub four_with_two: FourWithTwoRules,
    pub airplane: AirplaneRules,
}

/// Reveal schedule. A zero during-deal factor disables revealing at that received-card count.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RevealRules {
    pub before_deal_enabled: bool,
    pub before_deal_factor: u32,
    pub during_deal_factors: [u32; 18],
    pub after_bottom_enabled: bool,
    pub after_bottom_factor: u32,
}

impl RevealRules {
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            before_deal_enabled: false,
            before_deal_factor: 1,
            during_deal_factors: [0; 18],
            after_bottom_enabled: false,
            after_bottom_factor: 1,
        }
    }

    #[must_use]
    pub fn factor_during_deal(self, cards_received: u8) -> Option<u32> {
        self.during_deal_factors
            .get(usize::from(cards_received))
            .copied()
            .filter(|factor| *factor != 0)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CallingRules {
    pub enabled: bool,
    pub redeal_on_all_pass: bool,
    pub first_revealer_becomes_landlord_on_all_pass: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RobbingRules {
    pub enabled: bool,
    pub caller_can_reclaim: bool,
    pub factor_per_successful_rob: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DoublingRules {
    pub enabled: bool,
    pub factor: u32,
    /// A player is eligible only when every required balance is strictly greater than this value.
    pub minimum_balance_exclusive: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpringRules {
    pub landlord_spring_enabled: bool,
    pub farmer_spring_enabled: bool,
    pub factor: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SettlementRules {
    pub base_unit: u32,
    pub spring: SpringRules,
    /// Optional cap applied independently to each landlord-farmer transfer.
    pub pair_score_cap: Option<u64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardMode {
    WinPercentage,
    AverageDifferencePoints,
    LogAverageDifferencePoints,
    RawScore,
}

/// One complete immutable ruleset.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuleConfig {
    pub schema_version: u32,
    pub rule_config_id: u32,
    pub profile: RuleProfile,
    pub bottom_cards_public: bool,
    pub reveal: RevealRules,
    pub calling: CallingRules,
    pub robbing: RobbingRules,
    pub doubling: DoublingRules,
    pub moves: MoveRules,
    pub settlement: SettlementRules,
    pub reward_mode: RewardMode,
}

impl RuleConfig {
    pub fn from_yaml_str(yaml: &str) -> Result<Self, RuleConfigError> {
        let value: Self = serde_yaml_ng::from_str(yaml).map_err(RuleConfigError::Yaml)?;
        value.validate()?;
        Ok(value)
    }

    pub fn rules_hash(&self) -> Result<String, RuleConfigError> {
        hash::rules_hash(self)
    }

    pub fn validate(&self) -> Result<(), RuleConfigError> {
        super::validate::validate(self)
    }

    #[must_use]
    pub fn douzero_post_bid(rule_config_id: u32, reward_mode: RewardMode) -> Self {
        Self {
            schema_version: RULE_CONFIG_SCHEMA_VERSION,
            rule_config_id,
            profile: RuleProfile::DouzeroPostBid,
            bottom_cards_public: true,
            reveal: RevealRules::disabled(),
            calling: CallingRules {
                enabled: false,
                redeal_on_all_pass: false,
                first_revealer_becomes_landlord_on_all_pass: false,
            },
            robbing: RobbingRules {
                enabled: false,
                caller_can_reclaim: false,
                factor_per_successful_rob: 2,
            },
            doubling: DoublingRules {
                enabled: false,
                factor: 2,
                minimum_balance_exclusive: 0,
            },
            moves: MoveRules {
                four_with_two: FourWithTwoRules {
                    two_singles_enabled: true,
                    two_pairs_enabled: true,
                    single_attachments: AttachmentMultiplicity::MayShareRank,
                    pair_attachments: AttachmentMultiplicity::DistinctRanks,
                },
                airplane: AirplaneRules {
                    single_attachments: AttachmentMultiplicity::MayShareRank,
                    pair_attachments: AttachmentMultiplicity::DistinctRanks,
                },
            },
            settlement: SettlementRules {
                base_unit: 1,
                spring: SpringRules {
                    landlord_spring_enabled: false,
                    farmer_spring_enabled: false,
                    factor: 1,
                },
                pair_score_cap: None,
            },
            reward_mode,
        }
    }

    #[must_use]
    pub fn huanle_classic(rule_config_id: u32, during_deal_factors: [u32; 18]) -> Self {
        Self {
            schema_version: RULE_CONFIG_SCHEMA_VERSION,
            rule_config_id,
            profile: RuleProfile::HuanleClassic,
            bottom_cards_public: true,
            reveal: RevealRules {
                before_deal_enabled: true,
                before_deal_factor: 5,
                during_deal_factors,
                after_bottom_enabled: true,
                after_bottom_factor: 2,
            },
            calling: CallingRules {
                enabled: true,
                redeal_on_all_pass: true,
                first_revealer_becomes_landlord_on_all_pass: true,
            },
            robbing: RobbingRules {
                enabled: true,
                caller_can_reclaim: true,
                factor_per_successful_rob: 2,
            },
            doubling: DoublingRules {
                enabled: true,
                factor: 2,
                minimum_balance_exclusive: 0,
            },
            moves: MoveRules {
                four_with_two: FourWithTwoRules {
                    two_singles_enabled: true,
                    two_pairs_enabled: true,
                    single_attachments: AttachmentMultiplicity::MayShareRank,
                    pair_attachments: AttachmentMultiplicity::DistinctRanks,
                },
                airplane: AirplaneRules {
                    single_attachments: AttachmentMultiplicity::MayShareRank,
                    pair_attachments: AttachmentMultiplicity::DistinctRanks,
                },
            },
            settlement: SettlementRules {
                base_unit: 1,
                spring: SpringRules {
                    landlord_spring_enabled: false,
                    farmer_spring_enabled: false,
                    factor: 1,
                },
                pair_score_cap: None,
            },
            reward_mode: RewardMode::RawScore,
        }
    }
}
