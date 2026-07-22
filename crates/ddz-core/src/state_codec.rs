//! Versioned wire format for authoritative game states.

use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::GameState;

/// Wire schema understood by [`deserialize_game_state`].
pub const GAME_STATE_SCHEMA_VERSION: u32 = 1;

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct StateEnvelopeRef<'a> {
    schema_version: u32,
    state: &'a GameState,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStateEnvelope {
    schema_version: u32,
    state: serde_json::Value,
}

/// Serialize a state into the stable schema-versioned JSON byte representation.
///
/// Field order follows the checked-in Rust structs and output is compact UTF-8.
/// Deserializers must still treat the bytes as a schema rather than relying on
/// textual offsets.
///
/// # Errors
///
/// Returns [`StateCodecError::Json`] if serialization fails.
pub fn serialize_game_state(state: &GameState) -> Result<Vec<u8>, StateCodecError> {
    serde_json::to_vec(&StateEnvelopeRef {
        schema_version: GAME_STATE_SCHEMA_VERSION,
        state,
    })
    .map_err(StateCodecError::Json)
}

/// Decode a structural game state from the versioned wire representation.
///
/// This validates the envelope schema and all Serde-backed field and [`Move`](crate::Move)
/// invariants. Cross-field transition history is validated when the state is
/// restored into the rules engine.
///
/// # Errors
///
/// Returns [`StateCodecError::Json`] for malformed, unknown, missing, or invalid
/// fields and [`StateCodecError::UnsupportedSchemaVersion`] for another schema.
pub fn deserialize_game_state(bytes: &[u8]) -> Result<GameState, StateCodecError> {
    let envelope: RawStateEnvelope =
        serde_json::from_slice(bytes).map_err(StateCodecError::Json)?;
    if envelope.schema_version != GAME_STATE_SCHEMA_VERSION {
        return Err(StateCodecError::UnsupportedSchemaVersion {
            expected: GAME_STATE_SCHEMA_VERSION,
            actual: envelope.schema_version,
        });
    }
    serde_json::from_value(envelope.state).map_err(StateCodecError::Json)
}

/// Errors returned by the game-state wire codec.
#[derive(Debug)]
pub enum StateCodecError {
    /// JSON syntax, shape, unknown-field, or nested invariant failure.
    Json(serde_json::Error),
    /// The envelope uses an unsupported state schema.
    UnsupportedSchemaVersion {
        /// Version supported by this build.
        expected: u32,
        /// Version carried by the input.
        actual: u32,
    },
}

impl Display for StateCodecError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(error) => write!(formatter, "invalid serialized game state: {error}"),
            Self::UnsupportedSchemaVersion { expected, actual } => write!(
                formatter,
                "unsupported game-state schema version {actual}; expected {expected}"
            ),
        }
    }
}

impl Error for StateCodecError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::UnsupportedSchemaVersion { .. } => None,
        }
    }
}
