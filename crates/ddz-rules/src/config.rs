//! Versioned rule configuration and validation.

use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

/// Schema version understood by this crate.
pub const RULE_CONFIG_SCHEMA_VERSION: u32 = 1;

/// Named environment profiles required by the implementation plan.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleProfile {
    /// DouZero-compatible environment beginning after the landlord is fixed.
    DouzeroPostBid,
    /// Project-defined complete game including bidding and raw scoring.
    CanonicalFull,
}

/// Supported landlord-selection mechanisms.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BiddingMode {
    /// The landlord is supplied before environment reset.
    Disabled,
    /// Players bid integer scores up to a configured maximum.
    Score,
    /// Players call and rob the landlord rather than bidding a score.
    Rob,
}

/// Bidding configuration.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BiddingRules {
    /// Landlord-selection mechanism.
    pub mode: BiddingMode,
    /// Inclusive maximum bid for [`BiddingMode::Score`]; absent for other modes.
    pub max_bid: Option<u8>,
    /// Whether a bidding round with no caller causes a redeal.
    pub redeal_on_all_pass: bool,
}

/// Spring and anti-spring scoring configuration.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpringRules {
    /// Whether a landlord win before either farmer plays cards earns a spring.
    pub landlord_spring_enabled: bool,
    /// Whether a farmer win after only one landlord play earns an anti-spring.
    pub anti_spring_enabled: bool,
    /// Score factor applied when either enabled event occurs.
    pub multiplier: u32,
}

/// Whether attachments within one move may reuse a rank.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentMultiplicity {
    /// Every attachment must have a different rank.
    DistinctRanks,
    /// Physical cards may share a rank, so a pair can supply two single wings.
    MayShareRank,
}

/// Configurable boundaries for four-with-two moves.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FourWithTwoRules {
    /// Whether four cards may carry two individual cards.
    pub two_singles_enabled: bool,
    /// Whether four cards may carry two pairs.
    pub two_pairs_enabled: bool,
    /// Rank multiplicity allowed among the two individual attachments.
    pub single_attachments: AttachmentMultiplicity,
    /// Rank multiplicity allowed among the two pair attachments.
    pub pair_attachments: AttachmentMultiplicity,
}

/// Configurable airplane attachment boundaries.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AirplaneRules {
    /// Rank multiplicity allowed among individual wings.
    pub single_attachments: AttachmentMultiplicity,
    /// Rank multiplicity allowed among pair wings.
    pub pair_attachments: AttachmentMultiplicity,
}

/// Terminal reward exposed by the environment wrapper.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardMode {
    /// Win percentage objective: terminal reward is `+1` or `-1`.
    WinPercentage,
    /// `DouZero` ADP objective: each bomb or rocket doubles reward magnitude.
    AverageDifferencePoints,
    /// `DouZero` logADP objective: signed bomb count plus one.
    LogAverageDifferencePoints,
    /// Complete-game platform score before any training transform.
    RawScore,
}

/// Complete, versioned rule selection for one environment instance.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuleConfig {
    /// Serialization schema version, currently [`RULE_CONFIG_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Stable non-zero identifier recorded in game states and replays.
    pub rule_config_id: u32,
    /// Named compatibility profile.
    pub profile: RuleProfile,
    /// Whether the landlord takes the first card-play turn.
    pub landlord_plays_first: bool,
    /// Landlord-selection rules.
    pub bidding: BiddingRules,
    /// Whether the three bottom cards become public after landlord selection.
    pub bottom_cards_public: bool,
    /// Whether the optional standard two-times doubling phase is enabled.
    pub doubling_enabled: bool,
    /// Score multiplier for each ordinary bomb.
    pub bomb_multiplier: u32,
    /// Score multiplier for the rocket.
    pub rocket_multiplier: u32,
    /// Spring and anti-spring scoring rules.
    pub spring: SpringRules,
    /// Four-with-two move boundaries.
    pub four_with_two: FourWithTwoRules,
    /// Airplane attachment boundaries.
    pub airplane: AirplaneRules,
    /// Optional absolute cap applied to each seat's raw terminal score.
    pub score_cap: Option<u32>,
    /// Terminal reward representation returned by the environment wrapper.
    pub reward_mode: RewardMode,
}

impl RuleConfig {
    /// Parse and validate one YAML rule configuration.
    ///
    /// # Errors
    ///
    /// Returns [`RuleConfigError::Yaml`] for malformed or unknown fields and a
    /// validation error for inconsistent values or profile drift.
    pub fn from_yaml_str(yaml: &str) -> Result<Self, RuleConfigError> {
        let config = serde_yaml_ng::from_str(yaml).map_err(RuleConfigError::Yaml)?;
        Self::validate(&config)?;
        Ok(config)
    }

    /// Validate schema, field relationships, and compatibility-profile invariants.
    ///
    /// # Errors
    ///
    /// Returns a descriptive [`RuleConfigError`] for the first invalid field.
    pub fn validate(&self) -> Result<(), RuleConfigError> {
        if self.schema_version != RULE_CONFIG_SCHEMA_VERSION {
            return Err(RuleConfigError::UnsupportedSchemaVersion {
                expected: RULE_CONFIG_SCHEMA_VERSION,
                actual: self.schema_version,
            });
        }
        if self.rule_config_id == 0 {
            return Err(Self::invalid("rule_config_id", "must be non-zero"));
        }

        self.validate_bidding()?;
        if !self.bomb_multiplier.is_power_of_two() {
            return Err(Self::invalid(
                "bomb_multiplier",
                "must be a positive power of two",
            ));
        }
        if !self.rocket_multiplier.is_power_of_two() {
            return Err(Self::invalid(
                "rocket_multiplier",
                "must be a positive power of two",
            ));
        }
        if self.spring.landlord_spring_enabled || self.spring.anti_spring_enabled {
            if self.spring.multiplier < 2 || !self.spring.multiplier.is_power_of_two() {
                return Err(Self::invalid(
                    "spring.multiplier",
                    "must be a power of two of at least two when spring is enabled",
                ));
            }
        } else if self.spring.multiplier != 1 {
            return Err(Self::invalid(
                "spring.multiplier",
                "must be one when both spring rules are disabled",
            ));
        }
        if self.score_cap == Some(0) {
            return Err(Self::invalid("score_cap", "must be positive when present"));
        }

        match self.profile {
            RuleProfile::DouzeroPostBid => self.validate_douzero_profile(),
            RuleProfile::CanonicalFull => {
                if self.bidding.mode == BiddingMode::Disabled {
                    Err(RuleConfigError::IncompatibleProfile {
                        profile: self.profile,
                        field: "bidding.mode",
                        expected: "score or rob",
                    })
                } else {
                    Ok(())
                }
            }
        }
    }

    fn validate_bidding(&self) -> Result<(), RuleConfigError> {
        match (self.bidding.mode, self.bidding.max_bid) {
            (BiddingMode::Disabled | BiddingMode::Rob, None) => Ok(()),
            (BiddingMode::Score, Some(max_bid)) if max_bid > 0 => Ok(()),
            (BiddingMode::Score, _) => Err(Self::invalid(
                "bidding.max_bid",
                "must be positive in score-bidding mode",
            )),
            (BiddingMode::Disabled | BiddingMode::Rob, Some(_)) => Err(Self::invalid(
                "bidding.max_bid",
                "must be absent unless score-bidding mode is selected",
            )),
        }
    }

    fn validate_douzero_profile(&self) -> Result<(), RuleConfigError> {
        self.require_douzero(self.landlord_plays_first, "landlord_plays_first", "true")?;
        self.require_douzero(
            self.bidding.mode == BiddingMode::Disabled,
            "bidding.mode",
            "disabled",
        )?;
        self.require_douzero(self.bottom_cards_public, "bottom_cards_public", "true")?;
        self.require_douzero(!self.doubling_enabled, "doubling_enabled", "false")?;
        self.require_douzero(self.bomb_multiplier == 2, "bomb_multiplier", "2")?;
        self.require_douzero(self.rocket_multiplier == 2, "rocket_multiplier", "2")?;
        self.require_douzero(
            !self.spring.landlord_spring_enabled
                && !self.spring.anti_spring_enabled
                && self.spring.multiplier == 1,
            "spring",
            "disabled with multiplier 1",
        )?;
        self.require_douzero(
            self.four_with_two
                == (FourWithTwoRules {
                    two_singles_enabled: true,
                    two_pairs_enabled: true,
                    single_attachments: AttachmentMultiplicity::MayShareRank,
                    pair_attachments: AttachmentMultiplicity::DistinctRanks,
                }),
            "four_with_two",
            "DouZero attachment semantics",
        )?;
        self.require_douzero(
            self.airplane
                == (AirplaneRules {
                    single_attachments: AttachmentMultiplicity::MayShareRank,
                    pair_attachments: AttachmentMultiplicity::DistinctRanks,
                }),
            "airplane",
            "DouZero attachment semantics",
        )?;
        self.require_douzero(
            !self.bidding.redeal_on_all_pass,
            "bidding.redeal_on_all_pass",
            "false",
        )?;
        self.require_douzero(self.score_cap.is_none(), "score_cap", "null")?;
        self.require_douzero(
            self.reward_mode != RewardMode::RawScore,
            "reward_mode",
            "win_percentage, average_difference_points, or log_average_difference_points",
        )
    }

    fn require_douzero(
        &self,
        condition: bool,
        field: &'static str,
        expected: &'static str,
    ) -> Result<(), RuleConfigError> {
        if condition {
            Ok(())
        } else {
            Err(RuleConfigError::IncompatibleProfile {
                profile: self.profile,
                field,
                expected,
            })
        }
    }

    const fn invalid(field: &'static str, reason: &'static str) -> RuleConfigError {
        RuleConfigError::InvalidField { field, reason }
    }
}

/// Errors returned while parsing or validating a rule configuration.
#[derive(Debug)]
pub enum RuleConfigError {
    /// YAML syntax, type, missing-field, or unknown-field error.
    Yaml(serde_yaml_ng::Error),
    /// The serialized schema cannot be interpreted by this crate version.
    UnsupportedSchemaVersion {
        /// Version supported by this crate.
        expected: u32,
        /// Version found in the configuration.
        actual: u32,
    },
    /// A field violates a local range or cross-field constraint.
    InvalidField {
        /// Dotted field path.
        field: &'static str,
        /// Human-readable constraint.
        reason: &'static str,
    },
    /// A valid general setting would break a named compatibility profile.
    IncompatibleProfile {
        /// Profile whose invariant would be broken.
        profile: RuleProfile,
        /// Dotted field path.
        field: &'static str,
        /// Required value or semantic behavior.
        expected: &'static str,
    },
}

impl Display for RuleConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Yaml(error) => write!(formatter, "invalid rule configuration YAML: {error}"),
            Self::UnsupportedSchemaVersion { expected, actual } => write!(
                formatter,
                "unsupported rule schema version {actual}; expected {expected}"
            ),
            Self::InvalidField { field, reason } => {
                write!(formatter, "invalid rule field `{field}`: {reason}")
            }
            Self::IncompatibleProfile {
                profile,
                field,
                expected,
            } => write!(
                formatter,
                "field `{field}` is incompatible with profile {profile:?}; expected {expected}"
            ),
        }
    }
}

impl Error for RuleConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Yaml(error) => Some(error),
            Self::UnsupportedSchemaVersion { .. }
            | Self::InvalidField { .. }
            | Self::IncompatibleProfile { .. } => None,
        }
    }
}
