use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::Rank;

pub const RULE_CONFIG_SCHEMA_VERSION: u32 = 1;
pub const RULE_SOURCE_URL: &str = "https://xxgk.seu.edu.cn/_upload/article/files/44/c8/f455e1d04d2a998e40454931740a/4f853bb4-29b9-45dc-9c56-7627ed4c9726.pdf";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleProfile {
    GuandanTwoDeck,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuleConfig {
    pub schema_version: u32,
    pub profile: RuleProfile,
    pub starting_level: Rank,
    pub report_remaining_six: bool,
    pub report_remaining_ten_on_request: bool,
}

impl RuleConfig {
    pub const fn tournament() -> Self {
        Self {
            schema_version: RULE_CONFIG_SCHEMA_VERSION,
            profile: RuleProfile::GuandanTwoDeck,
            starting_level: Rank::Two,
            report_remaining_six: true,
            report_remaining_ten_on_request: true,
        }
    }

    pub fn from_yaml_str(yaml: &str) -> Result<Self, RuleConfigError> {
        let value: Self = serde_yaml_ng::from_str(yaml)
            .map_err(|error| RuleConfigError::Yaml(error.to_string()))?;
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), RuleConfigError> {
        if self.schema_version != RULE_CONFIG_SCHEMA_VERSION {
            return Err(RuleConfigError::SchemaVersion {
                actual: self.schema_version,
                expected: RULE_CONFIG_SCHEMA_VERSION,
            });
        }
        if !self.starting_level.is_standard() {
            return Err(RuleConfigError::InvalidStartingLevel(self.starting_level));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuleConfigError {
    Yaml(String),
    SchemaVersion { actual: u32, expected: u32 },
    InvalidStartingLevel(Rank),
}

impl Display for RuleConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Yaml(error) => write!(formatter, "invalid Guandan YAML: {error}"),
            Self::SchemaVersion { actual, expected } => {
                write!(
                    formatter,
                    "schema version {actual} does not match {expected}"
                )
            }
            Self::InvalidStartingLevel(rank) => {
                write!(formatter, "starting level {rank:?} is not a standard rank")
            }
        }
    }
}

impl Error for RuleConfigError {}
