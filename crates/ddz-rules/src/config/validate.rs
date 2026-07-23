use std::error::Error;
use std::fmt::{Display, Formatter};

use super::{RewardMode, RuleConfig, RuleProfile, RULE_CONFIG_SCHEMA_VERSION};

pub(super) fn validate(config: &RuleConfig) -> Result<(), RuleConfigError> {
    if config.schema_version != RULE_CONFIG_SCHEMA_VERSION {
        return Err(RuleConfigError::UnsupportedSchemaVersion {
            expected: RULE_CONFIG_SCHEMA_VERSION,
            actual: config.schema_version,
        });
    }
    require(config.rule_config_id != 0, "rule_config_id", "must be non-zero")?;
    validate_reveal(config)?;
    require(
        config.robbing.factor_per_successful_rob == 2,
        "robbing.factor_per_successful_rob",
        "must be two because StakeState stores a base-two exponent",
    )?;
    require(
        config.doubling.factor == 2,
        "doubling.factor",
        "must be two because pairwise doubling is binary",
    )?;
    require(
        config.settlement.base_unit > 0,
        "settlement.base_unit",
        "must be positive",
    )?;
    if let Some(cap) = config.settlement.pair_score_cap {
        require(cap > 0, "settlement.pair_score_cap", "must be positive")?;
    }
    let spring = config.settlement.spring;
    if spring.landlord_spring_enabled || spring.farmer_spring_enabled {
        require(
            spring.factor == 2,
            "settlement.spring.factor",
            "must be two because StakeState stores spring as one base-two factor",
        )?;
    } else {
        require(
            spring.factor == 1,
            "settlement.spring.factor",
            "must be one when spring is disabled",
        )?;
    }

    match config.profile {
        RuleProfile::DouzeroPostBid => validate_douzero(config),
        RuleProfile::HuanleClassic => validate_huanle(config),
    }
}

fn validate_reveal(config: &RuleConfig) -> Result<(), RuleConfigError> {
    let rules = config.reveal;
    require(
        rules.during_deal_factors[0] == 0,
        "reveal.during_deal_factors[0]",
        "must be zero because no card has been dealt yet",
    )?;
    if rules.before_deal_enabled {
        require(
            rules.before_deal_factor >= 2,
            "reveal.before_deal_factor",
            "must be at least two when enabled",
        )?;
    } else {
        require(
            rules.before_deal_factor == 1,
            "reveal.before_deal_factor",
            "must be one when disabled",
        )?;
    }
    for (cards_received, factor) in rules.during_deal_factors.into_iter().enumerate() {
        if factor != 0 && factor < 2 {
            return Err(RuleConfigError::InvalidField {
                field: "reveal.during_deal_factors",
                reason: format!(
                    "entry {cards_received} is {factor}; enabled entries must be at least two"
                ),
            });
        }
    }
    if rules.after_bottom_enabled {
        require(
            rules.after_bottom_factor >= 2,
            "reveal.after_bottom_factor",
            "must be at least two when enabled",
        )?;
    } else {
        require(
            rules.after_bottom_factor == 1,
            "reveal.after_bottom_factor",
            "must be one when disabled",
        )?;
    }
    Ok(())
}

fn validate_douzero(config: &RuleConfig) -> Result<(), RuleConfigError> {
    require(config.bottom_cards_public, "bottom_cards_public", "must be true")?;
    require(!config.calling.enabled, "calling.enabled", "must be false")?;
    require(!config.robbing.enabled, "robbing.enabled", "must be false")?;
    require(!config.doubling.enabled, "doubling.enabled", "must be false")?;
    require(
        config.reveal == super::RevealRules::disabled(),
        "reveal",
        "must be fully disabled",
    )?;
    require(
        !config.settlement.spring.landlord_spring_enabled
            && !config.settlement.spring.farmer_spring_enabled,
        "settlement.spring",
        "must be disabled",
    )?;
    require(
        config.reward_mode != RewardMode::RawScore,
        "reward_mode",
        "must be a DouZero training reward rather than raw_score",
    )
}

fn validate_huanle(config: &RuleConfig) -> Result<(), RuleConfigError> {
    require(config.bottom_cards_public, "bottom_cards_public", "must be true")?;
    require(
        config.reveal.before_deal_enabled,
        "reveal.before_deal_enabled",
        "must be true",
    )?;
    require(
        config.reveal.before_deal_factor == 5,
        "reveal.before_deal_factor",
        "must be five",
    )?;
    require(
        config.reveal.after_bottom_enabled,
        "reveal.after_bottom_enabled",
        "must be true",
    )?;
    require(
        config.reveal.after_bottom_factor == 2,
        "reveal.after_bottom_factor",
        "must be two",
    )?;
    require(config.calling.enabled, "calling.enabled", "must be true")?;
    require(
        config.calling.redeal_on_all_pass,
        "calling.redeal_on_all_pass",
        "must be true",
    )?;
    require(
        config.calling.first_revealer_becomes_landlord_on_all_pass,
        "calling.first_revealer_becomes_landlord_on_all_pass",
        "must be true",
    )?;
    require(config.robbing.enabled, "robbing.enabled", "must be true")?;
    require(
        config.robbing.caller_can_reclaim,
        "robbing.caller_can_reclaim",
        "must be true for the selected Huanle call/rob interpretation",
    )?;
    require(config.doubling.enabled, "doubling.enabled", "must be true")?;
    let mut previous = u32::MAX;
    for factor in config
        .reveal
        .during_deal_factors
        .into_iter()
        .filter(|factor| *factor != 0)
    {
        require(
            matches!(factor, 3 | 4),
            "reveal.during_deal_factors",
            "Huanle during-deal reveal factors must be three or four",
        )?;
        require(
            factor <= previous,
            "reveal.during_deal_factors",
            "enabled factors must not increase as more cards are received",
        )?;
        previous = factor;
    }
    require(
        config.reward_mode == RewardMode::RawScore,
        "reward_mode",
        "must be raw_score for the complete Huanle environment",
    )
}

fn require(
    condition: bool,
    field: &'static str,
    reason: &'static str,
) -> Result<(), RuleConfigError> {
    if condition {
        Ok(())
    } else {
        Err(RuleConfigError::InvalidField {
            field,
            reason: reason.to_owned(),
        })
    }
}

#[derive(Debug)]
pub enum RuleConfigError {
    UnsupportedSchemaVersion { expected: u32, actual: u32 },
    InvalidField { field: &'static str, reason: String },
    Yaml(serde_yaml_ng::Error),
    Json(serde_json::Error),
}

impl Display for RuleConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { expected, actual } => write!(
                formatter,
                "rule-config schema {actual} is unsupported; expected {expected}"
            ),
            Self::InvalidField { field, reason } => write!(formatter, "invalid {field}: {reason}"),
            Self::Yaml(error) => Display::fmt(error, formatter),
            Self::Json(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for RuleConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Yaml(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::UnsupportedSchemaVersion { .. } | Self::InvalidField { .. } => None,
        }
    }
}
