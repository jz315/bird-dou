//! Versioned rule configuration and validation.

use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Schema version for the pre-Huanle rule configuration.
pub const RULE_CONFIG_V1_SCHEMA_VERSION: u32 = 1;
/// Schema version for the Huanle-capable rule configuration.
pub const RULE_CONFIG_V2_SCHEMA_VERSION: u32 = 2;
/// Backwards-compatible name for the legacy v1 schema version.
pub const RULE_CONFIG_SCHEMA_VERSION: u32 = RULE_CONFIG_V1_SCHEMA_VERSION;

/// Named environment profiles required by the implementation plan.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleProfile {
    /// DouZero-compatible environment beginning after the landlord is fixed.
    DouzeroPostBid,
    /// Project-defined complete game retained for old replays and checkpoints.
    #[serde(rename = "canonical_full")]
    CanonicalFullLegacyV1,
    /// Strictly versioned Huanle Dou Dizhu profile.
    #[serde(rename = "huanle_classic_v1")]
    HuanleClassicV1,
}

impl RuleProfile {
    /// Historical Rust spelling retained for downstream source compatibility.
    #[allow(non_upper_case_globals)]
    #[deprecated(note = "use RuleProfile::CanonicalFullLegacyV1")]
    pub const CanonicalFull: Self = Self::CanonicalFullLegacyV1;
}

/// Supported landlord-selection mechanisms in the legacy schema.
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

/// Bidding configuration in the legacy schema.
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

/// Complete legacy v1 rule selection for one environment instance.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuleConfigV1 {
    /// Serialization schema version, always [`RULE_CONFIG_V1_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Stable non-zero identifier recorded in legacy game states and replays.
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

/// Backwards-compatible legacy rule configuration name.
pub type RuleConfig = RuleConfigV1;

impl RuleConfigV1 {
    /// Parse and validate one v1 YAML rule configuration.
    ///
    /// # Errors
    ///
    /// Returns [`RuleConfigError::Yaml`] for malformed or unknown fields and a
    /// validation error for inconsistent values, schema mismatch, or profile drift.
    pub fn from_yaml_str(yaml: &str) -> Result<Self, RuleConfigError> {
        require_schema_version(yaml, RULE_CONFIG_V1_SCHEMA_VERSION)?;
        let config = serde_yaml_ng::from_str(yaml).map_err(RuleConfigError::Yaml)?;
        Self::validate(&config)?;
        Ok(config)
    }

    /// Return the stable SHA-256 identity of this complete v1 configuration.
    ///
    /// The hash includes a domain separator, every serialized field, and the schema version.
    ///
    /// # Errors
    ///
    /// Returns [`RuleConfigError::Json`] only if canonical serialization unexpectedly fails.
    pub fn rules_hash(&self) -> Result<String, RuleConfigError> {
        rule_config_hash(self)
    }

    /// Validate schema, field relationships, and compatibility-profile invariants.
    ///
    /// # Errors
    ///
    /// Returns a descriptive [`RuleConfigError`] for the first invalid field.
    pub fn validate(&self) -> Result<(), RuleConfigError> {
        if self.schema_version != RULE_CONFIG_V1_SCHEMA_VERSION {
            return Err(RuleConfigError::UnsupportedSchemaVersion {
                expected: RULE_CONFIG_V1_SCHEMA_VERSION,
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
        validate_spring(self.spring, "spring.multiplier")?;
        if self.score_cap == Some(0) {
            return Err(Self::invalid("score_cap", "must be positive when present"));
        }

        match self.profile {
            RuleProfile::DouzeroPostBid => self.validate_douzero_profile(),
            RuleProfile::CanonicalFullLegacyV1 => {
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
            RuleProfile::HuanleClassicV1 => Err(RuleConfigError::IncompatibleProfile {
                profile: self.profile,
                field: "schema_version",
                expected: "2 for huanle_classic_v1",
            }),
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

/// Fixed deck and turn-order rules for the v2 Huanle profile.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DealRules {
    /// Number of players in the deal.
    pub player_count: u8,
    /// Cards initially dealt to each player.
    pub cards_per_player: u8,
    /// Cards reserved as the bottom.
    pub bottom_card_count: u8,
    /// Whether bottom cards are public before a landlord exists.
    pub bottom_visible_before_landlord: bool,
    /// Whether bottom cards are public after landlord resolution.
    pub bottom_visible_after_landlord: bool,
    /// Whether the resolved landlord leads card play.
    pub landlord_plays_first: bool,
}

/// Roles permitted to reveal their post-bottom hand.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RevealRoleEligibility {
    /// Only the landlord, who has received the bottom, may reveal.
    LandlordOnly,
    /// Any player may reveal under a future profile.
    AnyPlayer,
}

/// Explicit factors and visibility policies for all Huanle reveal opportunities.
///
/// The independent booleans are deliberate wire-level choices required by the frozen profile;
/// collapsing them would reintroduce implicit defaults.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RevealRules {
    /// Whether a player may reveal before cards are dealt.
    pub before_deal_enabled: bool,
    /// Factor applied to a valid pre-deal reveal.
    pub before_deal_factor: u32,
    /// Whether a player may reveal while receiving cards.
    pub during_deal_enabled: bool,
    /// Factor for each possible received-card count from zero through seventeen.
    pub factor_by_cards_received: [u32; 18],
    /// Whether the landlord may reveal after receiving the bottom.
    pub after_bottom_enabled: bool,
    /// Factor applied to a valid post-bottom reveal.
    pub after_bottom_factor: u32,
    /// Whether multiple reveal events use their maximum factor instead of multiplication.
    pub maximum_factor_only: bool,
    /// Roles allowed to make the post-bottom reveal.
    pub after_bottom_eligible_roles: RevealRoleEligibility,
}

/// How the first call candidate is selected.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FirstCallerPolicy {
    /// The first revealer calls first; otherwise the deterministic deal seed supplies a seat.
    FirstRevealerElseSeededSeat,
    /// The deterministic deal seed supplies a seat regardless of reveal history.
    SeededSeat,
}

/// Resolution after every player declines to call.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AllPassPolicy {
    /// Start a new deal attempt.
    Redeal,
    /// Assign the first revealer as landlord; invalid if no player has revealed.
    FirstRevealerBecomesLandlord,
    /// Assign the first revealer when present, otherwise start a new deal attempt.
    FirstRevealerBecomesLandlordElseRedeal,
}

/// Huanle calling state-machine configuration.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CallingRules {
    /// First eligible caller selection.
    pub first_caller_policy: FirstCallerPolicy,
    /// Whether the first positive call resolves the calling phase immediately.
    pub call_ends_immediately: bool,
    /// Resolution after all candidates pass.
    pub all_pass_policy: AllPassPolicy,
    /// Whether a player who passes the call loses its rob chance.
    pub passed_call_loses_rob_eligibility: bool,
}

/// Huanle robbing state-machine configuration.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RobbingRules {
    /// Whether robbing follows a successful call.
    pub enabled: bool,
    /// Public factor applied for each successful rob.
    pub factor_per_successful_rob: u32,
    /// Whether each eligible seat receives at most one rob opportunity.
    pub each_eligible_player_once: bool,
    /// Whether the original caller can reclaim after another player robs.
    pub caller_can_reclaim: bool,
}

/// Eligibility policy for the per-seat Huanle doubling decision.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DoubleEligibilityMode {
    /// Every seat is eligible without an account-balance check.
    AllPlayers,
    /// A seat is eligible only when its room balance meets the configured threshold.
    RoomBalanceThreshold,
}

/// Huanle doubling configuration.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DoublingRules {
    /// Whether every eligible seat receives a doubling decision.
    pub enabled: bool,
    /// Factor applied to one eligible player's pairwise stake on a double.
    pub factor: u32,
    /// Eligibility computation for each seat.
    pub eligibility_mode: DoubleEligibilityMode,
    /// Minimum room balance when [`DoubleEligibilityMode::RoomBalanceThreshold`] is selected.
    pub room_balance_threshold: u64,
}

/// Move legality boundaries unique to the v2 Huanle profile.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CardPlayRules {
    /// Whether wildcard cards are active; Huanle classic v1 forbids them.
    pub wildcards_enabled: bool,
    /// Four-with-two attachment policy.
    pub four_with_two: FourWithTwoRules,
    /// Airplane attachment policy.
    pub airplane: AirplaneRules,
}

/// How platform bean loss limits are applied during settlement.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BeanCapPolicy {
    /// Do not cap by a seat's bean balance.
    None,
    /// Cap a losing seat's payment at its available balance.
    AvailableBalance,
}

/// Pairwise Huanle settlement configuration.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SettlementRules {
    /// Base stake before public and pairwise factors.
    pub base_unit: i64,
    /// Whether landlord-versus-farmer payoffs are tracked independently per pair.
    pub pairwise_landlord_farmer: bool,
    /// Spring and anti-spring configuration.
    pub spring: SpringRules,
    /// Optional absolute cap on an individual pairwise score.
    #[serde(deserialize_with = "deserialize_required_option")]
    pub score_cap: Option<i64>,
    /// Bean-balance cap policy.
    pub bean_cap_policy: BeanCapPolicy,
}

/// Reward representation requested after the authoritative v2 settlement is complete.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RewardRules {
    /// Reward representation.
    pub mode: RewardMode,
}

/// Complete v2 Huanle rule selection.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuleConfigV2 {
    /// Serialization schema version, always [`RULE_CONFIG_V2_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Stable non-zero identifier for v2 states, replays, and checkpoints.
    pub rule_config_id: u32,
    /// The only v2 profile currently supported by this crate.
    pub profile: RuleProfile,
    /// Fixed card-dealing and turn-order choices.
    pub deal: DealRules,
    /// Reveal phases and factors.
    pub reveal: RevealRules,
    /// Calling phase rules.
    pub calling: CallingRules,
    /// Robbing phase rules.
    pub robbing: RobbingRules,
    /// Per-seat doubling rules.
    pub doubling: DoublingRules,
    /// Card-play legality choices.
    pub card_play: CardPlayRules,
    /// Pairwise terminal settlement choices.
    pub settlement: SettlementRules,
    /// Training-facing reward conversion.
    pub reward: RewardRules,
}

impl RuleConfigV2 {
    /// Parse and validate one v2 YAML rule configuration.
    ///
    /// This accepts only `huanle_classic_v1`; legacy profiles must remain in v1.
    ///
    /// # Errors
    ///
    /// Returns [`RuleConfigError::Yaml`] for malformed, incomplete, or unknown fields and a
    /// validation error for a schema mismatch or incompatible Huanle setting.
    pub fn from_yaml_str(yaml: &str) -> Result<Self, RuleConfigError> {
        require_schema_version(yaml, RULE_CONFIG_V2_SCHEMA_VERSION)?;
        let config = serde_yaml_ng::from_str(yaml).map_err(RuleConfigError::Yaml)?;
        Self::validate(&config)?;
        Ok(config)
    }

    /// Return the stable SHA-256 identity of this complete v2 configuration.
    ///
    /// The hash includes a domain separator, every serialized field, and the schema version.
    ///
    /// # Errors
    ///
    /// Returns [`RuleConfigError::Json`] only if canonical serialization unexpectedly fails.
    pub fn rules_hash(&self) -> Result<String, RuleConfigError> {
        rule_config_hash(self)
    }

    /// Validate every Huanle-specific field without supplying defaults.
    ///
    /// # Errors
    ///
    /// Returns the first schema, profile, range, or Huanle compatibility violation.
    pub fn validate(&self) -> Result<(), RuleConfigError> {
        self.validate_identity()?;
        self.validate_deal()?;
        self.validate_reveal()?;
        self.validate_calling()?;
        self.validate_robbing()?;
        self.validate_doubling()?;
        self.validate_card_play()?;
        self.validate_settlement()
    }

    fn validate_identity(&self) -> Result<(), RuleConfigError> {
        if self.schema_version != RULE_CONFIG_V2_SCHEMA_VERSION {
            return Err(RuleConfigError::UnsupportedSchemaVersion {
                expected: RULE_CONFIG_V2_SCHEMA_VERSION,
                actual: self.schema_version,
            });
        }
        if self.rule_config_id == 0 {
            return Err(Self::invalid("rule_config_id", "must be non-zero"));
        }
        if self.profile != RuleProfile::HuanleClassicV1 {
            return Err(RuleConfigError::IncompatibleProfile {
                profile: self.profile,
                field: "profile",
                expected: "huanle_classic_v1 for schema version 2",
            });
        }
        Ok(())
    }

    fn validate_deal(&self) -> Result<(), RuleConfigError> {
        self.require(
            self.deal.player_count == 3,
            "deal.player_count",
            "3 for huanle_classic_v1",
        )?;
        self.require(
            self.deal.cards_per_player == 17,
            "deal.cards_per_player",
            "17 for huanle_classic_v1",
        )?;
        self.require(
            self.deal.bottom_card_count == 3,
            "deal.bottom_card_count",
            "3 for huanle_classic_v1",
        )?;
        self.require(
            !self.deal.bottom_visible_before_landlord,
            "deal.bottom_visible_before_landlord",
            "false",
        )?;
        self.require(
            self.deal.bottom_visible_after_landlord,
            "deal.bottom_visible_after_landlord",
            "true",
        )?;
        self.require(
            self.deal.landlord_plays_first,
            "deal.landlord_plays_first",
            "true",
        )?;
        Ok(())
    }

    fn validate_reveal(&self) -> Result<(), RuleConfigError> {
        self.require(
            self.reveal.before_deal_enabled,
            "reveal.before_deal_enabled",
            "true",
        )?;
        self.require(
            self.reveal.before_deal_factor == 5,
            "reveal.before_deal_factor",
            "5",
        )?;
        self.require(
            self.reveal.during_deal_enabled,
            "reveal.during_deal_enabled",
            "true",
        )?;
        self.validate_reveal_schedule()?;
        self.require(
            self.reveal.after_bottom_enabled,
            "reveal.after_bottom_enabled",
            "true",
        )?;
        self.require(
            self.reveal.after_bottom_factor == 2,
            "reveal.after_bottom_factor",
            "2",
        )?;
        self.require(
            self.reveal.maximum_factor_only,
            "reveal.maximum_factor_only",
            "true",
        )?;
        self.require(
            self.reveal.after_bottom_eligible_roles == RevealRoleEligibility::LandlordOnly,
            "reveal.after_bottom_eligible_roles",
            "landlord_only",
        )?;
        Ok(())
    }

    fn validate_calling(&self) -> Result<(), RuleConfigError> {
        self.require(
            self.calling.first_caller_policy == FirstCallerPolicy::FirstRevealerElseSeededSeat,
            "calling.first_caller_policy",
            "first_revealer_else_seeded_seat",
        )?;
        self.require(
            self.calling.call_ends_immediately,
            "calling.call_ends_immediately",
            "true",
        )?;
        self.require(
            self.calling.all_pass_policy == AllPassPolicy::FirstRevealerBecomesLandlordElseRedeal,
            "calling.all_pass_policy",
            "first_revealer_becomes_landlord_else_redeal",
        )?;
        self.require(
            self.calling.passed_call_loses_rob_eligibility,
            "calling.passed_call_loses_rob_eligibility",
            "true",
        )?;
        Ok(())
    }

    fn validate_robbing(&self) -> Result<(), RuleConfigError> {
        self.require(self.robbing.enabled, "robbing.enabled", "true")?;
        self.require(
            self.robbing.factor_per_successful_rob == 2,
            "robbing.factor_per_successful_rob",
            "2",
        )?;
        self.require(
            self.robbing.each_eligible_player_once,
            "robbing.each_eligible_player_once",
            "true",
        )?;
        Ok(())
    }

    fn validate_doubling(&self) -> Result<(), RuleConfigError> {
        self.require(self.doubling.enabled, "doubling.enabled", "true")?;
        self.require(self.doubling.factor == 2, "doubling.factor", "2")?;
        self.require(
            self.doubling.eligibility_mode == DoubleEligibilityMode::RoomBalanceThreshold,
            "doubling.eligibility_mode",
            "room_balance_threshold",
        )?;
        Ok(())
    }

    fn validate_card_play(&self) -> Result<(), RuleConfigError> {
        self.require(
            !self.card_play.wildcards_enabled,
            "card_play.wildcards_enabled",
            "false",
        )?;
        self.require(
            self.card_play.four_with_two.two_singles_enabled,
            "card_play.four_with_two.two_singles_enabled",
            "true",
        )?;
        self.require(
            self.card_play.four_with_two.two_pairs_enabled,
            "card_play.four_with_two.two_pairs_enabled",
            "true",
        )?;
        self.require(
            self.card_play.four_with_two.pair_attachments == AttachmentMultiplicity::DistinctRanks,
            "card_play.four_with_two.pair_attachments",
            "distinct_ranks",
        )?;
        self.require(
            self.card_play.airplane.pair_attachments == AttachmentMultiplicity::DistinctRanks,
            "card_play.airplane.pair_attachments",
            "distinct_ranks",
        )?;
        Ok(())
    }

    fn validate_settlement(&self) -> Result<(), RuleConfigError> {
        if self.settlement.base_unit <= 0 {
            return Err(Self::invalid("settlement.base_unit", "must be positive"));
        }
        self.require(
            self.settlement.pairwise_landlord_farmer,
            "settlement.pairwise_landlord_farmer",
            "true",
        )?;
        validate_spring(self.settlement.spring, "settlement.spring.multiplier")?;
        if self
            .settlement
            .score_cap
            .is_some_and(|score_cap| score_cap <= 0)
        {
            return Err(Self::invalid(
                "settlement.score_cap",
                "must be positive when present",
            ));
        }
        self.require(
            self.reward.mode == RewardMode::RawScore,
            "reward.mode",
            "raw_score",
        )
    }

    fn validate_reveal_schedule(&self) -> Result<(), RuleConfigError> {
        let schedule = self.reveal.factor_by_cards_received;
        if schedule.iter().any(|factor| !matches!(factor, 3 | 4)) {
            return Err(Self::invalid(
                "reveal.factor_by_cards_received",
                "must contain only the explicit x3 and x4 factors",
            ));
        }
        if !schedule.contains(&3) || !schedule.contains(&4) {
            return Err(Self::invalid(
                "reveal.factor_by_cards_received",
                "must explicitly contain both x3 and x4 factors",
            ));
        }
        Ok(())
    }

    fn require(
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

/// One parsed rule configuration, selected by its serialized schema version.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(untagged)]
pub enum VersionedRuleConfig {
    /// Legacy v1 configuration.
    V1(RuleConfigV1),
    /// Huanle-capable v2 configuration.
    V2(RuleConfigV2),
}

impl VersionedRuleConfig {
    /// Parse a configuration by first dispatching on `schema_version`.
    ///
    /// # Errors
    ///
    /// Returns a parse, validation, or unsupported-version error without trying another schema.
    pub fn from_yaml_str(yaml: &str) -> Result<Self, RuleConfigError> {
        match schema_version_from_yaml(yaml)? {
            RULE_CONFIG_V1_SCHEMA_VERSION => RuleConfigV1::from_yaml_str(yaml).map(Self::V1),
            RULE_CONFIG_V2_SCHEMA_VERSION => RuleConfigV2::from_yaml_str(yaml).map(Self::V2),
            actual => Err(RuleConfigError::UnsupportedVersionedSchemaVersion { actual }),
        }
    }

    /// Validate the configuration selected by its schema.
    ///
    /// # Errors
    ///
    /// Returns the selected v1 or v2 validation error.
    pub fn validate(&self) -> Result<(), RuleConfigError> {
        match self {
            Self::V1(config) => config.validate(),
            Self::V2(config) => config.validate(),
        }
    }

    /// Serialized schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        match self {
            Self::V1(config) => config.schema_version,
            Self::V2(config) => config.schema_version,
        }
    }

    /// Stable non-zero rule configuration identifier.
    #[must_use]
    pub const fn rule_config_id(&self) -> u32 {
        match self {
            Self::V1(config) => config.rule_config_id,
            Self::V2(config) => config.rule_config_id,
        }
    }

    /// Named rule profile.
    #[must_use]
    pub const fn profile(&self) -> RuleProfile {
        match self {
            Self::V1(config) => config.profile,
            Self::V2(config) => config.profile,
        }
    }

    /// Return the stable SHA-256 identity of the selected configuration.
    ///
    /// # Errors
    ///
    /// Returns [`RuleConfigError::Json`] only if canonical serialization unexpectedly fails.
    pub fn rules_hash(&self) -> Result<String, RuleConfigError> {
        match self {
            Self::V1(config) => config.rules_hash(),
            Self::V2(config) => config.rules_hash(),
        }
    }

    /// Borrow the configuration for the legacy v1 engine.
    ///
    /// # Errors
    ///
    /// A v2 Huanle configuration is intentionally rejected until the v2 engine exists.
    pub fn as_v1(&self) -> Result<&RuleConfigV1, RuleConfigError> {
        match self {
            Self::V1(config) => Ok(config),
            Self::V2(config) => Err(RuleConfigError::UnsupportedByLegacyEngine {
                schema_version: config.schema_version,
                profile: config.profile,
            }),
        }
    }

    /// Borrow the configuration for the future v2 engine.
    ///
    /// # Errors
    ///
    /// A legacy configuration must never be silently interpreted as a Huanle configuration.
    pub fn as_v2(&self) -> Result<&RuleConfigV2, RuleConfigError> {
        match self {
            Self::V1(config) => Err(RuleConfigError::UnsupportedByV2Engine {
                schema_version: config.schema_version,
                profile: config.profile,
            }),
            Self::V2(config) => Ok(config),
        }
    }

    /// Consume the selected configuration for the legacy v1 engine.
    ///
    /// # Errors
    ///
    /// A v2 Huanle configuration is intentionally rejected until the v2 engine exists.
    pub fn into_v1(self) -> Result<RuleConfigV1, RuleConfigError> {
        match self {
            Self::V1(config) => Ok(config),
            Self::V2(config) => Err(RuleConfigError::UnsupportedByLegacyEngine {
                schema_version: config.schema_version,
                profile: config.profile,
            }),
        }
    }
}

#[derive(Deserialize)]
struct SchemaVersionHeader {
    schema_version: u32,
}

fn schema_version_from_yaml(yaml: &str) -> Result<u32, RuleConfigError> {
    serde_yaml_ng::from_str::<SchemaVersionHeader>(yaml)
        .map(|header| header.schema_version)
        .map_err(RuleConfigError::Yaml)
}

fn require_schema_version(yaml: &str, expected: u32) -> Result<(), RuleConfigError> {
    let actual = schema_version_from_yaml(yaml)?;
    if actual == expected {
        Ok(())
    } else {
        Err(RuleConfigError::UnsupportedSchemaVersion { expected, actual })
    }
}

fn validate_spring(
    spring: SpringRules,
    multiplier_field: &'static str,
) -> Result<(), RuleConfigError> {
    if spring.landlord_spring_enabled || spring.anti_spring_enabled {
        if spring.multiplier < 2 || !spring.multiplier.is_power_of_two() {
            return Err(RuleConfigError::InvalidField {
                field: multiplier_field,
                reason: "must be a power of two of at least two when spring is enabled",
            });
        }
    } else if spring.multiplier != 1 {
        return Err(RuleConfigError::InvalidField {
            field: multiplier_field,
            reason: "must be one when both spring rules are disabled",
        });
    }
    Ok(())
}

fn rule_config_hash<T: Serialize>(config: &T) -> Result<String, RuleConfigError> {
    let canonical_json = serde_json::to_vec(config).map_err(RuleConfigError::Json)?;
    let mut hasher = Sha256::new();
    hasher.update(b"bird-dou/rule-config/");
    hasher.update(canonical_json);
    Ok(format!("{:x}", hasher.finalize()))
}

fn deserialize_required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

/// Errors returned while parsing, validating, or hashing a rule configuration.
#[derive(Debug)]
pub enum RuleConfigError {
    /// YAML syntax, type, missing-field, or unknown-field error.
    Yaml(serde_yaml_ng::Error),
    /// JSON serialization failed while constructing a stable configuration hash.
    Json(serde_json::Error),
    /// The serialized schema cannot be interpreted by one specific schema reader.
    UnsupportedSchemaVersion {
        /// Version supported by the reader.
        expected: u32,
        /// Version found in the configuration.
        actual: u32,
    },
    /// The serialized schema cannot be interpreted by the versioned reader.
    UnsupportedVersionedSchemaVersion {
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
    /// The legacy engine received a parsed v2 configuration.
    UnsupportedByLegacyEngine {
        /// Parsed configuration schema.
        schema_version: u32,
        /// Parsed configuration profile.
        profile: RuleProfile,
    },
    /// The future v2 engine received a parsed legacy configuration.
    UnsupportedByV2Engine {
        /// Parsed configuration schema.
        schema_version: u32,
        /// Parsed configuration profile.
        profile: RuleProfile,
    },
}

impl Display for RuleConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Yaml(error) => write!(formatter, "invalid rule configuration YAML: {error}"),
            Self::Json(error) => write!(formatter, "failed to serialize rule configuration: {error}"),
            Self::UnsupportedSchemaVersion { expected, actual } => write!(
                formatter,
                "unsupported rule schema version {actual}; expected {expected}"
            ),
            Self::UnsupportedVersionedSchemaVersion { actual } => write!(
                formatter,
                "unsupported rule schema version {actual}; supported versions are 1 and 2"
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
            Self::UnsupportedByLegacyEngine {
                schema_version,
                profile,
            } => write!(
                formatter,
                "legacy v1 engine cannot interpret rule schema {schema_version} profile {profile:?}"
            ),
            Self::UnsupportedByV2Engine {
                schema_version,
                profile,
            } => write!(
                formatter,
                "v2 Huanle engine cannot interpret legacy rule schema {schema_version} profile {profile:?}"
            ),
        }
    }
}

impl Error for RuleConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Yaml(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::UnsupportedSchemaVersion { .. }
            | Self::UnsupportedVersionedSchemaVersion { .. }
            | Self::InvalidField { .. }
            | Self::IncompatibleProfile { .. }
            | Self::UnsupportedByLegacyEngine { .. }
            | Self::UnsupportedByV2Engine { .. } => None,
        }
    }
}
