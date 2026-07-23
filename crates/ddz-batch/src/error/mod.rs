//! Batch-level errors with slot attribution.

use ddz_core::{CodecError, StakeError};
use ddz_rules::{GameError, GameRestoreError, RuleConfigError, RuleProfile};

/// Error returned by native batch operations.
#[derive(Debug)]
pub enum BatchError {
    /// Shared rules are invalid.
    RuleConfig(RuleConfigError),
    /// An operation requiring initialized slots was called before reset/restore.
    Uninitialized,
    /// A typed batch snapshot uses a different packed schema.
    UnsupportedSnapshotSchema {
        /// Snapshot schema.
        actual: u32,
        /// Required schema.
        expected: u32,
    },
    /// A typed snapshot contains the reserved zero generation.
    InvalidSnapshotGeneration {
        /// Failing slot.
        slot: usize,
    },
    /// A snapshot was created under a different immutable ruleset.
    SnapshotRulesMismatch {
        /// Hash of the current rules.
        expected: String,
        /// Hash recorded in the snapshot.
        actual: String,
    },
    /// A full reset/restore supplied no slots.
    EmptyBatch,
    /// Parallel input arrays disagree with the current batch size.
    BatchSizeMismatch {
        /// Input field name.
        field: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// A partial reset references a nonexistent slot.
    SlotOutOfRange {
        /// Requested slot.
        slot: usize,
        /// Current batch length.
        batch_size: usize,
    },
    /// One partial reset request mentions a slot more than once.
    DuplicateSlot {
        /// Repeated slot.
        slot: usize,
    },
    /// Reset constructor does not match the batch's rule profile.
    ResetProfileMismatch {
        /// Failing slot in the supplied reset list.
        slot: usize,
        /// Shared batch profile.
        profile: RuleProfile,
        /// Reset variant required by that profile.
        expected: &'static str,
    },
    /// One new game failed to initialize.
    Reset {
        /// Failing slot.
        slot: usize,
        /// Rule-engine error.
        source: GameError,
    },
    /// One encoded state failed to decode.
    Decode {
        /// Failing slot.
        slot: usize,
        /// Core codec error.
        source: CodecError,
    },
    /// One decoded state failed deterministic replay verification.
    Restore {
        /// Failing slot.
        slot: usize,
        /// Restore error.
        source: GameRestoreError,
    },
    /// One state failed serialization.
    Encode {
        /// Failing slot.
        slot: usize,
        /// Core codec error.
        source: CodecError,
    },
    /// One information-set observation failed.
    Observe {
        /// Failing slot.
        slot: usize,
        /// Rule-engine error.
        source: GameError,
    },
    /// One legal-action query failed.
    LegalActions {
        /// Failing slot.
        slot: usize,
        /// Rule-engine error.
        source: GameError,
    },
    /// Selected local index is below the only supported sentinel `-1`.
    InvalidActionIndex {
        /// Failing slot.
        slot: usize,
        /// Supplied index.
        index: i64,
    },
    /// A terminal slot received a real action rather than `-1`.
    TerminalActionIndex {
        /// Failing slot.
        slot: usize,
        /// Supplied index.
        index: i64,
    },
    /// Selected local index does not belong to the slot's legal range.
    ActionIndexOutOfRange {
        /// Failing slot.
        slot: usize,
        /// Supplied local index.
        index: i64,
        /// Number of legal actions.
        legal_count: usize,
    },
    /// An asynchronous response was produced before the slot was reset/restored.
    StaleGeneration {
        /// Failing slot.
        slot: usize,
        /// Current generation.
        expected: u64,
        /// Generation attached to the response.
        actual: u64,
    },
    /// An asynchronous response was produced for an older state in the same generation.
    StaleRevision {
        /// Failing slot.
        slot: usize,
        /// Current revision.
        expected: u64,
        /// Revision attached to the response.
        actual: u64,
    },
    /// One authoritative transition failed after all packed inputs were validated.
    Step {
        /// Failing slot.
        slot: usize,
        /// Rule-engine error.
        source: GameError,
    },
    /// Batch rollback failed; the caller must discard this object.
    Rollback {
        /// Failing slot.
        slot: usize,
        /// Rule-engine error.
        source: GameError,
    },
    /// Monotonic generation counter exhausted `u64`.
    GenerationOverflow,
    /// One slot revision exhausted `u64`.
    RevisionOverflow {
        /// Failing slot.
        slot: usize,
    },
    /// A validated state could not be represented by the packed stake protocol.
    PackStake {
        /// Failing slot.
        slot: usize,
        /// Stake arithmetic error.
        source: StakeError,
    },
    /// A packed offset or owner cannot fit its declared wire type.
    BufferOverflow {
        /// Logical buffer field.
        field: &'static str,
    },
    /// Internal length or cache contract was violated.
    InternalInvariant(&'static str),
}


mod display;
