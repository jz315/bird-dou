use ddz_core::GameAction;

use crate::protocol::PackedActions;
use crate::BatchError;

use super::BatchEnv;

impl BatchEnv {
    /// Generate stable ragged legal-action ranges for all slots.
    ///
    /// Results are cached per slot and invalidated only when that slot advances or resets.
    pub fn legal_actions_packed(&mut self) -> Result<PackedActions, BatchError> {
        self.require_initialized()?;
        for slot in 0..self.slots.len() {
            self.ensure_legal_slot(slot)?;
        }
        let rows = (0..self.slots.len())
            .map(|slot| {
                let version = self.slots[slot].version;
                let actions = self
                    .legal_cache
                    .get(slot, version)
                    .ok_or(BatchError::InternalInvariant(
                        "legal cache was not populated for a current slot version",
                    ))?;
                Ok((version, actions))
            })
            .collect::<Result<Vec<_>, BatchError>>()?;
        PackedActions::from_rows(rows)
    }

    /// Read one cached/generated legal action by local slot index.
    pub fn legal_action(
        &mut self,
        slot: usize,
        local_index: usize,
    ) -> Result<Option<GameAction>, BatchError> {
        self.require_initialized()?;
        if slot >= self.slots.len() {
            return Err(BatchError::SlotOutOfRange {
                slot,
                batch_size: self.slots.len(),
            });
        }
        self.ensure_legal_slot(slot)?;
        Ok(self
            .legal_cache
            .get(slot, self.slots[slot].version)
            .and_then(|actions| actions.get(local_index))
            .copied())
    }

    pub(crate) fn ensure_legal_slot(&mut self, slot: usize) -> Result<(), BatchError> {
        let version = self.slots[slot].version;
        if self.legal_cache.get(slot, version).is_none() {
            let actions = self.slots[slot]
                .game
                .legal_actions()
                .map_err(|source| BatchError::LegalActions { slot, source })?;
            self.legal_cache.insert(slot, version, actions);
        }
        Ok(())
    }
}
