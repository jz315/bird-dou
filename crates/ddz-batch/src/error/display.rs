use std::error::Error;
use std::fmt::{Display, Formatter};

use super::BatchError;

impl Display for BatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuleConfig(error) => Display::fmt(error, formatter),
            Self::Uninitialized => formatter.write_str("batch has not been reset or restored"),
            Self::UnsupportedSnapshotSchema { actual, expected } => write!(
                formatter,
                "batch snapshot schema {actual} is unsupported; expected {expected}"
            ),
            Self::InvalidSnapshotGeneration { slot } => write!(
                formatter,
                "batch snapshot slot {slot} uses reserved generation zero"
            ),
            Self::SnapshotRulesMismatch { expected, actual } => write!(
                formatter,
                "batch snapshot rules hash {actual} differs from current rules {expected}"
            ),
            Self::EmptyBatch => formatter.write_str("batch must contain at least one slot"),
            Self::BatchSizeMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "{field} has length {actual}; current batch size is {expected}"
            ),
            Self::SlotOutOfRange { slot, batch_size } => write!(
                formatter,
                "slot {slot} is outside batch size {batch_size}"
            ),
            Self::DuplicateSlot { slot } => {
                write!(formatter, "partial reset repeats slot {slot}")
            }
            Self::ResetProfileMismatch {
                slot,
                profile,
                expected,
            } => write!(
                formatter,
                "reset slot {slot} is incompatible with profile {profile:?}; expected {expected}"
            ),
            Self::Reset { slot, source } => {
                write!(formatter, "failed to reset slot {slot}: {source}")
            }
            Self::Decode { slot, source } => {
                write!(formatter, "failed to decode slot {slot}: {source}")
            }
            Self::Restore { slot, source } => {
                write!(formatter, "failed to restore slot {slot}: {source}")
            }
            Self::Encode { slot, source } => {
                write!(formatter, "failed to encode slot {slot}: {source}")
            }
            Self::Observe { slot, source } => {
                write!(formatter, "failed to observe slot {slot}: {source}")
            }
            Self::LegalActions { slot, source } => {
                write!(formatter, "failed to generate slot {slot} legal actions: {source}")
            }
            Self::InvalidActionIndex { slot, index } => write!(
                formatter,
                "slot {slot} action index {index} is invalid; use -1 to skip or a non-negative local index"
            ),
            Self::TerminalActionIndex { slot, index } => write!(
                formatter,
                "terminal slot {slot} received action index {index}; terminal slots must use -1"
            ),
            Self::ActionIndexOutOfRange {
                slot,
                index,
                legal_count,
            } => write!(
                formatter,
                "slot {slot} action index {index} is outside 0..{legal_count}"
            ),
            Self::StaleGeneration {
                slot,
                expected,
                actual,
            } => write!(
                formatter,
                "slot {slot} generation is {expected}, but response carries {actual}"
            ),
            Self::StaleRevision {
                slot,
                expected,
                actual,
            } => write!(
                formatter,
                "slot {slot} revision is {expected}, but response carries {actual}"
            ),
            Self::Step { slot, source } => {
                write!(formatter, "failed to step slot {slot}: {source}")
            }
            Self::Rollback { slot, source } => write!(
                formatter,
                "failed to roll back slot {slot}; discard the batch: {source}"
            ),
            Self::GenerationOverflow => formatter.write_str("slot generation counter overflowed"),
            Self::RevisionOverflow { slot } => {
                write!(formatter, "slot {slot} revision counter overflowed")
            }
            Self::PackStake { slot, source } => {
                write!(formatter, "failed to pack slot {slot} stake: {source}")
            }
            Self::BufferOverflow { field } => {
                write!(formatter, "packed buffer {field} exceeded its wire integer range")
            }
            Self::InternalInvariant(message) => formatter.write_str(message),
        }
    }
}

impl Error for BatchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RuleConfig(error) => Some(error),
            Self::Reset { source, .. } => Some(source),
            Self::Decode { source, .. } | Self::Encode { source, .. } => Some(source),
            Self::Restore { source, .. } => Some(source),
            Self::PackStake { source, .. } => Some(source),
            Self::Observe { source, .. }
            | Self::LegalActions { source, .. }
            | Self::Step { source, .. }
            | Self::Rollback { source, .. } => Some(source),
            _ => None,
        }
    }
}
