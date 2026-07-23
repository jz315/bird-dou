#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::struct_field_names
)]
#![doc = "Native vectorized ownership, packed protocols, and transactional batch stepping for BIRD-Dou."]

mod batch;
mod cache;
mod error;
mod reset;
mod version;

pub mod protocol;

pub use batch::{BatchEnv, BatchSnapshot, SlotSnapshot};
pub use error::BatchError;
pub use protocol::{
    PackedActionData, PackedActions, PackedEvents, PackedObservation, PackedStepResult,
};
pub use reset::{EncodedRestoreSpec, ResetSpec, RestoreSpec, SlotReset};
pub use version::SlotVersion;

/// Current packed-buffer protocol version.
///
/// Runtime domain types are not version-suffixed. Only the wire/buffer contract is versioned.
pub const BATCH_SCHEMA_VERSION: u32 = 2;
/// Local-action index used to leave one slot unchanged during a batch step.
pub const SKIP_ACTION_INDEX: i64 = -1;
