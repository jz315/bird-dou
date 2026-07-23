use ddz_core::{decode_state, encode_state};
use ddz_rules::{EconomyContext, Game};

use crate::protocol::PackedObservation;
use crate::{
    BatchError, EncodedRestoreSpec, RestoreSpec, SlotVersion, BATCH_SCHEMA_VERSION,
};

use super::{observe, BatchEnv, Slot};

/// Exact typed snapshot of one slot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SlotSnapshot {
    /// Match/deal seed.
    pub match_seed: u64,
    /// Huanle balance context.
    pub economy: EconomyContext,
    /// Slot version at snapshot time.
    pub version: SlotVersion,
    /// Versioned `ddz-core` state envelope.
    pub state_bytes: Vec<u8>,
}

/// Exact typed snapshot of the initialized native batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchSnapshot {
    /// Packed protocol version.
    pub schema_version: u32,
    /// Immutable rule-config hash.
    pub rules_hash: String,
    /// Slots in stable batch order.
    pub slots: Vec<SlotSnapshot>,
}

impl BatchEnv {
    /// Restore a whole batch from decoded states, assigning fresh slot generations.
    pub fn restore_all(&mut self, specs: &[RestoreSpec]) -> Result<PackedObservation, BatchError> {
        if specs.is_empty() {
            return Err(BatchError::EmptyBatch);
        }
        let (generations, next_generation) = self.checked_generations(specs.len())?;
        let slots = specs
            .iter()
            .zip(generations)
            .enumerate()
            .map(|(slot, (spec, generation))| {
                Game::restore(
                    self.rules.clone(),
                    spec.match_seed,
                    spec.economy,
                    spec.state.clone(),
                )
                .map(|game| Slot {
                    game,
                    version: SlotVersion::initial(generation),
                })
                .map_err(|source| BatchError::Restore { slot, source })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let observation = observe::pack_current(&slots)?;
        self.slots = slots;
        self.legal_cache.reset(self.slots.len());
        self.next_generation = next_generation;
        Ok(observation)
    }

    /// Decode and restore a whole batch, assigning fresh slot generations.
    pub fn restore_encoded_all(
        &mut self,
        specs: &[EncodedRestoreSpec],
    ) -> Result<PackedObservation, BatchError> {
        let decoded = specs
            .iter()
            .enumerate()
            .map(|(slot, spec)| {
                decode_state(&spec.state_bytes)
                    .map(|state| RestoreSpec {
                        match_seed: spec.match_seed,
                        economy: spec.economy,
                        state,
                    })
                    .map_err(|source| BatchError::Decode { slot, source })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.restore_all(&decoded)
    }

    /// Create an exact typed checkpoint, including slot versions.
    pub fn snapshot(&self) -> Result<BatchSnapshot, BatchError> {
        self.require_initialized()?;
        let rules_hash = self.rules.rules_hash().map_err(BatchError::RuleConfig)?;
        let slots = self
            .slots
            .iter()
            .enumerate()
            .map(|(slot, value)| {
                encode_state(value.game.state())
                    .map(|state_bytes| SlotSnapshot {
                        match_seed: value.game.match_seed(),
                        economy: value.game.economy(),
                        version: value.version,
                        state_bytes,
                    })
                    .map_err(|source| BatchError::Encode { slot, source })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BatchSnapshot {
            schema_version: BATCH_SCHEMA_VERSION,
            rules_hash,
            slots,
        })
    }

    /// Restore an exact typed checkpoint, preserving slot versions.
    pub fn restore_snapshot(
        &mut self,
        snapshot: &BatchSnapshot,
    ) -> Result<PackedObservation, BatchError> {
        if snapshot.schema_version != BATCH_SCHEMA_VERSION {
            return Err(BatchError::UnsupportedSnapshotSchema {
                actual: snapshot.schema_version,
                expected: BATCH_SCHEMA_VERSION,
            });
        }
        if snapshot.slots.is_empty() {
            return Err(BatchError::EmptyBatch);
        }
        let expected_hash = self.rules.rules_hash().map_err(BatchError::RuleConfig)?;
        if snapshot.rules_hash != expected_hash {
            return Err(BatchError::SnapshotRulesMismatch {
                expected: expected_hash,
                actual: snapshot.rules_hash.clone(),
            });
        }

        for (slot, value) in snapshot.slots.iter().enumerate() {
            if value.version.generation == 0 {
                return Err(BatchError::InvalidSnapshotGeneration { slot });
            }
        }

        let slots = snapshot
            .slots
            .iter()
            .enumerate()
            .map(|(slot, value)| {
                let state = decode_state(&value.state_bytes)
                    .map_err(|source| BatchError::Decode { slot, source })?;
                Game::restore(
                    self.rules.clone(),
                    value.match_seed,
                    value.economy,
                    state,
                )
                .map(|game| Slot {
                    game,
                    version: value.version,
                })
                .map_err(|source| BatchError::Restore { slot, source })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let next_generation = slots
            .iter()
            .map(|slot| slot.version.generation)
            .max()
            .ok_or(BatchError::EmptyBatch)?
            .checked_add(1)
            .ok_or(BatchError::GenerationOverflow)?;
        let observation = observe::pack_current(&slots)?;
        self.slots = slots;
        self.legal_cache.reset(self.slots.len());
        self.next_generation = next_generation;
        Ok(observation)
    }

    /// Encode only authoritative states in stable slot order.
    pub fn encoded_states(&self) -> Result<Vec<Vec<u8>>, BatchError> {
        self.require_initialized()?;
        self.slots
            .iter()
            .enumerate()
            .map(|(slot, value)| {
                encode_state(value.game.state())
                    .map_err(|source| BatchError::Encode { slot, source })
            })
            .collect()
    }
}
