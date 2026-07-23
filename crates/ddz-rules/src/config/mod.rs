//! Rule profiles and their validated configuration.

mod hash;
mod model;
mod validate;

pub use model::{
    AirplaneRules, AttachmentMultiplicity, CallingRules, DoublingRules, FourWithTwoRules,
    MoveRules, RevealRules, RewardMode, RobbingRules, RuleConfig, RuleProfile, SettlementRules,
    SpringRules, RULE_CONFIG_SCHEMA_VERSION,
};
pub use validate::RuleConfigError;
