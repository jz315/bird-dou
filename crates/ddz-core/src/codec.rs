use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::{GameState, GameStateError, Observation, ObservationError};

pub const STATE_SCHEMA_VERSION: u32 = 2;
pub const OBSERVATION_SCHEMA_VERSION: u32 = 2;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StateEnvelope {
    schema_version: u32,
    state: GameState,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationEnvelope {
    schema_version: u32,
    observation: Observation,
}

pub fn encode_state(state: &GameState) -> Result<Vec<u8>, CodecError> {
    state.validate().map_err(CodecError::State)?;
    serde_json::to_vec(&StateEnvelope {
        schema_version: STATE_SCHEMA_VERSION,
        state: state.clone(),
    })
    .map_err(CodecError::Json)
}

pub fn decode_state(bytes: &[u8]) -> Result<GameState, CodecError> {
    let envelope: StateEnvelope = serde_json::from_slice(bytes).map_err(CodecError::Json)?;
    if envelope.schema_version != STATE_SCHEMA_VERSION {
        return Err(CodecError::UnsupportedStateSchema {
            actual: envelope.schema_version,
            expected: STATE_SCHEMA_VERSION,
        });
    }
    envelope.state.validate().map_err(CodecError::State)?;
    Ok(envelope.state)
}

pub fn encode_observation(observation: &Observation) -> Result<Vec<u8>, CodecError> {
    observation
        .validate()
        .map_err(CodecError::Observation)?;
    serde_json::to_vec(&ObservationEnvelope {
        schema_version: OBSERVATION_SCHEMA_VERSION,
        observation: observation.clone(),
    })
    .map_err(CodecError::Json)
}

pub fn decode_observation(bytes: &[u8]) -> Result<Observation, CodecError> {
    let envelope: ObservationEnvelope =
        serde_json::from_slice(bytes).map_err(CodecError::Json)?;
    if envelope.schema_version != OBSERVATION_SCHEMA_VERSION {
        return Err(CodecError::UnsupportedObservationSchema {
            actual: envelope.schema_version,
            expected: OBSERVATION_SCHEMA_VERSION,
        });
    }
    envelope
        .observation
        .validate()
        .map_err(CodecError::Observation)?;
    Ok(envelope.observation)
}

#[derive(Debug)]
pub enum CodecError {
    Json(serde_json::Error),
    State(GameStateError),
    Observation(ObservationError),
    UnsupportedStateSchema { actual: u32, expected: u32 },
    UnsupportedObservationSchema { actual: u32, expected: u32 },
}

impl Display for CodecError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(error) => write!(formatter, "invalid JSON wire payload: {error}"),
            Self::State(error) => Display::fmt(error, formatter),
            Self::Observation(error) => Display::fmt(error, formatter),
            Self::UnsupportedStateSchema { actual, expected } => write!(
                formatter,
                "state schema {actual} is unsupported; expected {expected}"
            ),
            Self::UnsupportedObservationSchema { actual, expected } => write!(
                formatter,
                "observation schema {actual} is unsupported; expected {expected}"
            ),
        }
    }
}

impl Error for CodecError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::State(error) => Some(error),
            Self::Observation(error) => Some(error),
            Self::UnsupportedStateSchema { .. }
            | Self::UnsupportedObservationSchema { .. } => None,
        }
    }
}
