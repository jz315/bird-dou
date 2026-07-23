use sha2::{Digest, Sha256};

use super::{RuleConfig, RuleConfigError};

const HASH_DOMAIN: &[u8] = b"bird-dou/ddz-rules/rule-config/v1\0";

pub(super) fn rules_hash(config: &RuleConfig) -> Result<String, RuleConfigError> {
    let encoded = serde_json::to_vec(config).map_err(RuleConfigError::Json)?;
    let mut digest = Sha256::new();
    digest.update(HASH_DOMAIN);
    digest.update(encoded);
    Ok(format!("{:x}", digest.finalize()))
}
